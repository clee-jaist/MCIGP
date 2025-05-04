import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import argparse
import pyrealsense2 as rs
import torch.utils.data
import gc
import time

from hardware.camera import RealSenseCamera
from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.visualisation.plot import plot_results, save_results_f
from skimage.feature import peak_local_max
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from xarm.wrapper import XArmAPI
from method import *

logging.basicConfig(level=logging.INFO)
sys.path.append("..")

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate network')
    parser.add_argument('--network', type=str, default='GRconvnet_RGBD_epoch_40_iou_0.52',
                        help='Path to saved network to evaluate')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for evaluation (1/0)')
    parser.add_argument('--n-grasps', type=int, default=1,
                        help='Number of grasps to consider per image')
    parser.add_argument("--gpu", default="0", type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    torch.cuda.empty_cache()

    # Initialize the robot and gripper
    ip = '192.168.1.232'
    arm = XArmAPI(ip, is_radian=True)
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(state=0)
    arm.set_position(x=231.6, y=0, z=198,
                          roll=180, pitch=0, yaw=0, speed=200,
                          is_radian=False)
    arm.set_position(x=404, y=0, z=198,
                          roll=180, pitch=0, yaw=0, speed=200,
                          is_radian=False)

    code = arm.set_gripper_enable(True)
    code = arm.set_gripper_mode(0)

# ----------------------------------------------------------------------------------------------------------------------------#
                                                   # MVA part
# ----------------------------------------------------------------------------------------------------------------------------#

    # Connect to Camera
    logging.info('Connecting to camera...')
    cam = RealSenseCamera(device_id='943222070907')
    color_intrinsics = cam.connect()
    cam_data = CameraData(include_depth=args.use_depth, include_rgb=args.use_rgb)

    # Load Network
    logging.info('Loading model...')
    net = torch.load(args.network)
    logging.info('Done')

    # Get the compute device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    gpu = torch.cuda.current_device()
    device = get_device(gpu)

    try:
        fig = plt.figure(figsize=(10, 10))
        is_grasping = True  # # Indicate whether crawling is in progress
        exit_requested = False  # Indicate whether to request to exit
        count = 0  # Initialize the buffer frame counter
        move = 0  # Initialize the center movement frame counter
        rotation = 0  # Initialize the image rotation angle

        while True:
            image_bundle = cam.get_image_bundle()
            rgb = image_bundle['rgb']
            depth = image_bundle['aligned_depth']
            aligned_depth_frame = image_bundle['aligned_depth_frame']

            # Buffer frames to prevent poor prediction
            if count < 24:
                x, depth_img, rgb_img = cam_data.get_data(rgb=rgb, depth=depth)
                with torch.no_grad():
                    xc = x.to(device)
                    pred = net.predict(xc)
                    count += 1
                    if count == 1:
                        arm.set_position(x=403.8, y=5.2, z=650.2,
                                         roll=180, pitch=0, yaw=0, speed=200,
                                         is_radian=False)
                        while True:
                            ret, initPose = arm.get_position()
                            tolerance = 3
                            if abs(initPose[0] - 403.8) < tolerance and \
                                    abs(initPose[1] - 5.2) < tolerance and \
                                    abs(initPose[2] - 650.2) < tolerance:
                                break
                        time.sleep(2)
                        print("Entering the initial view, ready to grasping...")

            # MVA!!!
            elif move < 3:
                move += 1

                # Gobal MVA
                if move == 1:
                    center_x, center_y = depth.shape[1] // 2, depth.shape[0] // 2
                    start_x, start_y = center_x - 310, center_y - 160
                    end_x, end_y = center_x + 310, center_y + 160
                    center_region = depth[start_y:end_y, start_x:end_x]

                    filtered_depth_img = np.where(center_region < 0.2, np.inf, center_region)
                    min_depth_value = np.min(filtered_depth_img)
                    min_index = np.argmin(filtered_depth_img)
                    min_coords = np.unravel_index(min_index, filtered_depth_img.shape)

                    # Move camera
                    if min_depth_value < 0.662:  # Prevent collision with table
                        min_coords_squeezed = (min_coords[0] + start_y, min_coords[1] + start_x)

                        dis = depth[min_coords_squeezed[0], min_coords_squeezed[1]]
                        x, y, z = rs.rs2_deproject_pixel_to_point(intrin=color_intrinsics,
                                                                  pixel=[min_coords_squeezed[1], min_coords_squeezed[0]],
                                                                  depth=dis)
                        campos = [x, y, z]

                        robot_pos = (campos[1], -campos[0])
                        arm.set_tool_position(x=robot_pos[0] * 1000, y=robot_pos[1] * 1000, z=0,
                                              roll=0, pitch=0, yaw=0, speed=200,
                                              is_radian=False)

                        time.sleep(2)

                # Local MVA
                else:
                    # Dynamic Monozone
                    center_x, center_y = depth.shape[1] // 2, depth.shape[0] // 2
                    start_x, start_y = center_x - 112, center_y - 112
                    end_x, end_y = center_x + 112, center_y + 112
                    center_region = depth[start_y:end_y, start_x:end_x]

                    filtered_depth_img = np.where(center_region < 0.2, np.inf, center_region)
                    min_depth_value = np.min(filtered_depth_img)
                    min_index = np.argmin(filtered_depth_img)
                    min_coords = np.unravel_index(min_index, filtered_depth_img.shape)

                    if min_depth_value < 0.662:
                        min_coords_squeezed = (min_coords[0] + start_y, min_coords[1] + start_x)

                        dis = depth[min_coords_squeezed[0], min_coords_squeezed[1]]
                        x, y, z = rs.rs2_deproject_pixel_to_point(intrin=color_intrinsics,
                                                                  pixel=[min_coords_squeezed[1],
                                                                         min_coords_squeezed[0]],
                                                                  depth=dis)
                        campos = [x, y, z]

                        # move robot
                        robot_pos = (campos[1], -campos[0])
                        arm.set_tool_position(x=robot_pos[0] * 1000, y=robot_pos[1] * 1000, z=0,
                                              roll=0, pitch=0, yaw=0, speed=200,
                                              is_radian=False)

                        time.sleep(2)

# ----------------------------------------------------------------------------------------------------------------------------#
                                                       # ISGD part
# ----------------------------------------------------------------------------------------------------------------------------#

            # Conduct CPS in the aligned view
            else:

                if min_depth_value < 0.1:
                    print('Dangerous grasp, jump to next frame:', min_depth_value)
                    count = 0
                    move = 0
                    continue

                elif min_depth_value > 0.662:
                    print('Grasp finised')
                    count = 0
                    move = 0
                    time.sleep(2)
                    break

                else:
                    # initialize SAM
                    depth_squeezed = np.squeeze(depth, axis=2)
                    initial_point = (240, 320)
                    depth_value = min_depth_value
                    delta_d = 0.008
                    structure_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    d_mask = minkowski_sum(depth_squeezed, initial_point, depth_value, delta_d, structure_element)
                    min_coords_squeezed = (240, 320)

                    sam_checkpoint = "sam_vit_b_01ec64.pth"
                    model_type = "vit_b"
                    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
                    sam.to(device=device)
                    predictor = SamPredictor(sam)
                    predictor.set_image(rgb)

                    # Single-point segmentation
                    input_point = np.array([[min_coords_squeezed[1], min_coords_squeezed[0]]])
                    input_label = np.array([1])
                    masks, scores, logits = predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=False,
                    )

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()


                    mask = masks.squeeze(0)
                    edges = sobel_compute(mask)

                    # Find CPS points
                    point1, point2 = find_farthest_points(edges)
                    midpoint, slope, x_values, y_values = find_perpendicular(point1, point2, (480, 640))

                    # Find intersection points (cross points)
                    intersections = find_intersection_points(x_values, y_values, edges)

                    # CPS segmentation
                    if len(intersections) < 2:
                        input_point = np.array([input_point[0], (point1[1], point1[0]), (point2[1], point2[0])])
                        input_label = np.array([1, 1, 1])
                        masks, scores, logits = predictor.predict(
                            point_coords=input_point,
                            point_labels=input_label,
                            multimask_output=False,
                        )
                    else:
                        input_point = np.array(
                            [input_point[0], (point1[1], point1[0]), (point2[1], point2[0]),\
                             (intersections[0][0], intersections[0][1]),
                             (intersections[-1][0], intersections[-1][1])])
                        input_label = np.array([1, 1, 1, 1, 1])
                        masks, scores, logits = predictor.predict(
                            point_coords=input_point,
                            point_labels=input_label,
                            multimask_output=False,
                        )

                    second_largest_mask = masks.squeeze(0)
                    second_largest_mask_i = second_largest_mask.astype(int)
                    points = np.column_stack(np.nonzero(second_largest_mask_i))
                    print("mask points:", len(points))

                    if len(points) < 1500:
                        print("mask is too small:", len(points))
                        # Refine mask
                        second_largest_mask = second_largest_mask_i + d_mask
                        second_largest_mask = second_largest_mask.astype(bool)

                    rgb[~second_largest_mask] = [255, 255, 255]

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                        # Grasp detection
                        x, depth_img, rgb_img = cam_data.get_data(rgb=rgb, depth=depth)
                        with torch.no_grad():
                            xc = x.to(device)
                            pred = net.predict(xc)
                            q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'],
                                                                            pred['width'])

                        grasp_point1 = peak_local_max(q_img, min_distance=1, threshold_abs=0.3, num_peaks=100)
                        print(len(grasp_point1))

                        edges = sobel_compute(second_largest_mask)

                        # The first part of GCO
                        for grasp_point in grasp_point1:
                            length = width_img[grasp_point[0], grasp_point[1]]

                            # Calculate the optimal angle and minimum angle difference
                            best_angle, min_difference = calculate_total_difference(grasp_point, length,
                                                                                    edges)
                            # Update ang_img
                            if best_angle is not None:
                                ang_img[grasp_point[0], grasp_point[1]] = best_angle

                        graspable_points = grasp_point1

                        # The second part of GCO
                        graspable_points_adjacent = []
                        for grasp_point in graspable_points:
                            angle = ang_img[grasp_point[0], grasp_point[1]]
                            length = width_img[grasp_point[0], grasp_point[1]]
                            width = 43

                            # decode grasps
                            p1, p2, p3, p4 = decode_box(angle, grasp_point, length, width)

                            center, intersection, max_side_length = get_intersection_and_rectangle(p1, p2, p3, p4,
                                                                                                   second_largest_mask)
                            if center is not None:

                                all_relations_true = True

                                for i in range(13, 14):
                                    f_grasp_point = [int(center[1]) - 128, int(center[0]) - 208]
                                    f_length = min(max_side_length + i, 113)
                                    f_width = 43
                                    f_angle = angle

                                    p1, p2, p3, p4 = decode_box(f_angle, f_grasp_point, f_length, f_width)

                                    relation_one = compute_adjacent_relation(p1, p2, p3, p4, second_largest_mask,
                                                                             f_grasp_point, depth)

                                    if not relation_one:
                                        all_relations_true = False
                                        break

                                if all_relations_true:
                                    graspable_points_adjacent.append(grasp_point)

                        if len(graspable_points_adjacent) == 0:
                            print('no graspable_points，preparing to rotate image')

                            for i in np.linspace(30, 300, 5):
                                rotation = i
                                depth_r = np.mean(depth, axis=2)

                                # rotate rgb and depth
                                pil_rgb = Image.fromarray(rgb.astype('uint8'), 'RGB')
                                pil_depth = Image.fromarray(depth_r)
                                rotated_rgb = pil_rgb.rotate(rotation, resample=Image.BILINEAR)
                                rotated_depth = pil_depth.rotate(rotation, resample=Image.BILINEAR)
                                rotated_rgb = np.array(rotated_rgb)
                                rotated_depth_np = np.array(rotated_depth)
                                rotated_depth_np = np.stack([rotated_depth_np] * 1, axis=-1)

                                # rotate mask
                                r_second_largest_mask = second_largest_mask
                                pil_mask = Image.fromarray(r_second_largest_mask.astype(np.uint8))
                                r_second_largest_mask = pil_mask.rotate(rotation, resample=Image.BILINEAR)
                                r_second_largest_mask = np.array(r_second_largest_mask)

                                # Recalculate the edge of the new mask
                                r_edges = sobel_compute(r_second_largest_mask)

                                r_x, r_depth_img, r_rgb_img = cam_data.get_data(rgb=rotated_rgb, depth=rotated_depth_np)
                                with torch.no_grad():
                                    r_xc = r_x.to(device)
                                    r_pred = net.predict(r_xc)
                                    r_q_img, r_ang_img, r_width_img = post_process_output(r_pred['pos'], r_pred['cos'], r_pred['sin'],
                                                                                    r_pred['width'])

                                    grasp_point1 = peak_local_max(r_q_img, min_distance=1, threshold_abs=0.1, num_peaks=100)

                                    for grasp_point in grasp_point1:
                                        length = r_width_img[grasp_point[0], grasp_point[1]]

                                        best_angle, min_difference = calculate_total_difference(grasp_point, length,
                                                                                                r_edges)
                                        if best_angle is not None:
                                            r_ang_img[grasp_point[0], grasp_point[1]] = best_angle

                                graspable_points = grasp_point1

                                # Filter grasps that collide with adjacent objects
                                graspable_points_adjacent = []
                                for grasp_point in graspable_points:
                                    angle = ang_img[grasp_point[0], grasp_point[1]]
                                    length = width_img[grasp_point[0], grasp_point[1]]
                                    width = 43

                                    # decode grasps
                                    p1, p2, p3, p4 = decode_box(angle, grasp_point, length, width)

                                    center, intersection, max_side_length = get_intersection_and_rectangle(p1, p2, p3,
                                                                                                           p4,
                                                                                                           second_largest_mask)
                                    if center is not None:

                                        all_relations_true = True

                                        for i in range(13, 14):
                                            f_grasp_point = [int(center[1]) - 128, int(center[0]) - 208]
                                            f_length = min(max_side_length + i, 113)
                                            f_width = 43
                                            f_angle = angle

                                            p1, p2, p3, p4 = decode_box(f_angle, f_grasp_point, f_length, f_width)

                                            relation_one = compute_adjacent_relation(p1, p2, p3, p4,
                                                                                     second_largest_mask,
                                                                                     f_grasp_point, depth)

                                            if not relation_one:
                                                all_relations_true = False
                                                break

                                        if all_relations_true:
                                            graspable_points_adjacent.append(grasp_point)

                                if len(graspable_points_adjacent) != 0:
                                    print("can grasp now, rotated角度为:", rotation)
                                    break

                            if len(graspable_points_adjacent) == 0:
                                print('still can not grasp this object after rotated, jump to next frame')
                                plot_results(fig=fig,
                                             rgb_img=cam_data.get_rgb(rotated_rgb, False),
                                             depth_img=np.squeeze(cam_data.get_depth(rotated_depth_np)),
                                             grasp_q_img=r_q_img,
                                             grasp_angle_img=r_ang_img,
                                             no_grasps=args.n_grasps,
                                             grasp_width_img=r_width_img,
                                             point=None,
                                             fine_point=None,
                                             fine_width=None)
                                count = 0
                                move = 0
                                rotation = 0
                                continue

                        # Update variables
                        if rotation != 0:
                            rgb = rotated_rgb
                            depth = rotated_depth_np
                            q_img = r_q_img
                            width_img = r_width_img
                            ang_img = r_ang_img
                            second_largest_mask = r_second_largest_mask
                            edges = r_edges
                            depth_img = r_depth_img

                        # Select the best grasp based on depth
                        graspable_points_depth = []

                        for grasp_point in graspable_points_adjacent:
                            x, y = grasp_point[0], grasp_point[1]
                            depth_img_squeezed = np.squeeze(depth_img, axis=0)
                            depth_value = depth_img_squeezed[x, y]
                            graspable_points_depth.append((grasp_point, depth_value))

                        if graspable_points_depth:
                            graspable_points_depth.sort(key=lambda x: x[1])
                            best_grasp_point = graspable_points_depth[0][0]
                        else:
                            print("No graspable points found.")

                        # Refine optimal grasp
                        F_best_grasp_point = None
                        for width in range(10, 2, -1):
                            angle = ang_img[best_grasp_point[0], best_grasp_point[1]]
                            length = width_img[best_grasp_point[0], best_grasp_point[1]]

                            p1, p2, p3, p4 = decode_box(angle, best_grasp_point, length, width)

                            center, intersection, max_side_length = get_intersection_and_rectangle(p1, p2, p3, p4,
                                                                                                   second_largest_mask)
                            if center is not None:
                                F_best_grasp_point = [center[1], center[0]]
                                F_length = min(max_side_length+13, 113)
                                F_angle = ang_img[best_grasp_point[0], best_grasp_point[1]]
                                break

                            if F_best_grasp_point is None:
                                print('No intersection, jump to next frame:')
                                count = 0
                                move = 0
                                rotation = 0

                                plot_results(fig=fig,
                                             rgb_img=cam_data.get_rgb(rgb, False),
                                             depth_img=np.squeeze(cam_data.get_depth(depth)),
                                             grasp_q_img=q_img,
                                             grasp_angle_img=ang_img,
                                             no_grasps=args.n_grasps,
                                             grasp_width_img=width_img,
                                             point=best_grasp_point,
                                             fine_point=None,
                                             fine_width=None)
                                continue

                        C_F_best_grasp_point = [F_best_grasp_point[1] - 128, F_best_grasp_point[0] - 208]
                        print("Finetuned Pose (center, length, angle):", F_best_grasp_point, F_length, F_angle)

                        center = [F_best_grasp_point[0], F_best_grasp_point[1]]
                        data_array = [int(center[0]), int(center[1])]  # Project 224*224 back to 640*480
                        print("rotated_data_array:", data_array)

                        # Project angle back to aligned view
                        if rotation != 0:
                            rotated_x = data_array[0]
                            rotated_y = data_array[1]
                            original_width = 640
                            original_height = 480

                            center_x = original_width / 2
                            center_y = original_height / 2

                            # Calculate the rotation matrix
                            theta = np.radians(-rotation)
                            cos_theta = np.cos(theta)
                            sin_theta = np.sin(theta)
                            rotation_matrix = np.array([[cos_theta, -sin_theta],
                                                        [sin_theta, cos_theta]])

                            # Calculate the offset of the rotated pixel coordinates
                            rotated_offset_x = rotated_x - center_x
                            rotated_offset_y = rotated_y - center_y

                            # Calculate the pixel coordinates (x, y) before rotation
                            original_offset_x, original_offset_y = np.dot(rotation_matrix.T,
                                                                          [rotated_offset_x, rotated_offset_y])

                            # Convert the offset back to the coordinates (x, y) of the original image
                            original_x = original_offset_x + center_x
                            original_y = original_offset_y + center_y

                            original_x = int(round(original_x))
                            original_y = int(round(original_y))

                            data_array[0] = original_x
                            data_array[1] = original_y

                        dis = depth[data_array[1], data_array[0]]
                        print("dis:", dis)
                        x, y, z = rs.rs2_deproject_pixel_to_point(intrin=color_intrinsics,
                                                                  pixel=[data_array[0], data_array[1]],
                                                                  depth=dis)
                        campos = [x, y, z]
                        if dis < 0.1 or dis > 0.662:
                            print('Dangerous grasp, jump to next frame:', dis)
                            count = 0
                            move = 0
                            rotation = 0

                            plot_results(fig=fig,
                                         rgb_img=cam_data.get_rgb(rgb, False),
                                         depth_img=np.squeeze(cam_data.get_depth(depth)),
                                         grasp_q_img=q_img,
                                         grasp_angle_img=ang_img,
                                         no_grasps=args.n_grasps,
                                         grasp_width_img=width_img,
                                         point=best_grasp_point,
                                         fine_point=C_F_best_grasp_point,
                                         fine_width=F_length)
                            continue

                        plot_results(fig=fig,
                                     rgb_img=cam_data.get_rgb(rgb, False),
                                     depth_img=np.squeeze(cam_data.get_depth(depth)),
                                     grasp_q_img=q_img,
                                     grasp_angle_img=ang_img,
                                     no_grasps=args.n_grasps,
                                     grasp_width_img=width_img,
                                     point=best_grasp_point,
                                     fine_point=C_F_best_grasp_point,
                                     fine_width=F_length)

# ----------------------------------------------------------------------------------------------------------------------------#
                                                    # Grasping part
# ---------------------------------------------------------------------------------------------------------------------------- #

                        if is_grasping:

                            save_results_f(
                                         rgb_img=cam_data.get_rgb(rgb, False),
                                         depth_img=np.squeeze(cam_data.get_depth(depth)),
                                         grasp_q_img=q_img,
                                         grasp_angle_img=ang_img,
                                         no_grasps=0,
                                         grasp_width_img=width_img,
                                         point=best_grasp_point,
                                         fine_point=C_F_best_grasp_point,
                                         fine_width=F_length)

                            if campos != None:

                                # Add rotation offset
                                if rotation != 0:
                                    F_angle = F_angle * (180 / np.pi)
                                    F_angle = F_angle - rotation
                                    F_angle = map_to_minus_90_to_90 (F_angle)
                                    F_angle = F_angle * (np.pi / 180)
                                angle = F_angle * (180 / np.pi)

                                # Note: please use your own hand-eye relation, here we only use translation.
                                calibration_result = [0.0852, - 0.026, 0.18]
                                robot_pos = (campos[1] + 0.0682, 0.025 - campos[0])

                                arm.set_tool_position(x=robot_pos[0] * 1000, y=robot_pos[1] * 1000, z=0,
                                                      roll=0, pitch=0, yaw=0, speed=200,
                                                      is_radian=False)
                                code = arm.set_gripper_position(F_length * 7.5, wait=True)
                                arm.set_tool_position(x=0, y=0,
                                                      z=dis * 1000 - 0.08 * 1000,
                                                      roll=0, pitch=0, yaw=-angle, speed=50,
                                                      is_radian=False)
                                code = arm.set_gripper_mode(0)
                                code = arm.set_gripper_position(20, wait=True)
                                arm.set_tool_position(x=0, y=0,
                                                      z=-200,
                                                      roll=0, pitch=0, yaw=0, speed=150,
                                                      is_radian=False)
                                arm.set_position(x=413, y=-536, z=220,
                                                 roll=180, pitch=0, yaw=0, speed=200,
                                                 is_radian=False)
                                arm.set_position(x=413, y=-536, z=55,
                                                 roll=180, pitch=0, yaw=0, speed=200,
                                                 is_radian=False)
                                code = arm.set_gripper_position(850, wait=True)
                                arm.set_position(x=403.8, y=5.2, z=650.2,
                                                 roll=180, pitch=0, yaw=0, speed=200,
                                                 is_radian=False)
                                while True:
                                    ret, initPose = arm.get_position()
                                    tolerance = 3
                                    if abs(initPose[0] - 403.8) < tolerance and \
                                            abs(initPose[1] - 5.2) < tolerance and \
                                            abs(initPose[2] - 650.2) < tolerance:
                                        break
                                count = 0
                                move = 0
                                rotation = 0

                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            else:
                                print("No optimal grasping point detected, move robot again")
                                aligned_depth_frame = image_bundle['aligned_depth_frame']
                                dis = depth[min_coords_squeezed[0], min_coords_squeezed[1]]
                                x, y, z = rs.rs2_deproject_pixel_to_point(intrin=color_intrinsics,
                                                                          pixel=[min_coords_squeezed[1],
                                                                                 min_coords_squeezed[0]],
                                                                          depth=dis)
                                campos = [x, y, z]

                                robot_pos = (campos[1], -campos[0])
                                arm.set_tool_position(x=robot_pos[0] * 1000, y=robot_pos[1] * 1000, z=0,
                                                      roll=0, pitch=0, yaw=0, speed=200,
                                                      is_radian=False)
                                time.sleep(2)

    finally:
        None