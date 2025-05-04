import logging
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
import matplotlib.pyplot as plt
import numpy as np
import cv2
logging.basicConfig(level=logging.INFO)
import sys
sys.path.append("..")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from bresenham import bresenham
from scipy.ndimage import sobel
import itertools


# Grasp decode
def decode_box(angle, grasp_point, length, width):
    xo = np.cos(angle)
    yo = np.sin(angle)

    y1 = grasp_point[0] + length / 2 * yo
    x1 = grasp_point[1] - length / 2 * xo
    y2 = grasp_point[0] - length / 2 * yo
    x2 = grasp_point[1] + length / 2 * xo

    p1 = (int(y1 - width / 2 * xo), int(x1 - width / 2 * yo))
    p2 = (int(y2 - width / 2 * xo), int(x2 - width / 2 * yo))
    p3 = (int(y2 + width / 2 * xo), int(x2 + width / 2 * yo))
    p4 = (int(y1 + width / 2 * xo), int(x1 + width / 2 * yo))

    return p1, p2, p3, p4


# Sobel operator
def sobel_compute(mask):
    edges_x = sobel(mask, axis=0)
    edges_y = sobel(mask, axis=1)
    edges = np.hypot(edges_x, edges_y)
    edges = (edges > 0).astype(np.uint8)
    return edges


# find intersection points
def find_intersection_points(x_values, y_values, mask):
    intersections = []
    x_values = np.array(x_values, dtype=int)
    y_values = np.array(y_values, dtype=int)

    # Ensure line covers the image dimensions
    if x_values[0] == x_values[1]:  # Vertical line
        for y in range(min(y_values), max(y_values) + 1):
            if mask[y, x_values[0]] == 1:
                intersections.append((x_values[0], y))
    else:  # Non-vertical line
        slope = (y_values[1] - y_values[0]) / (x_values[1] - x_values[0])
        for x in range(min(x_values), max(x_values) + 1):
            y = int(y_values[0] + slope * (x - x_values[0]))
            if mask[y, x] == 1:
                intersections.append((x, y))

    return intersections


#  find perpendicular
def find_perpendicular(point1, point2, image_shape):
    # Calculate midpoint of the line segment
    midpoint = ((point1[1] + point2[1]) / 2, (point1[0] + point2[0]) / 2)

    # Calculate slope of the line segment
    if point2[1] - point1[1] == 0:
        slope = None
    else:
        slope = (point2[0] - point1[0]) / (point2[1] - point1[1])

    # Calculate slope of the perpendicular line
    if slope is None:
        perpendicular_slope = float('inf')
    elif slope == 0:
        perpendicular_slope = 0
    else:
        perpendicular_slope = -1 / slope

    # Determine the perpendicular line equation
    if perpendicular_slope != 0 and perpendicular_slope != float('inf'):
        x_values = [midpoint[0] - 100, midpoint[0] + 100]
        y_values = [perpendicular_slope * (x - midpoint[0]) + midpoint[1] for x in x_values]
    else:
        x_values = [midpoint[0], midpoint[0]]
        y_values = [midpoint[1] - 100, midpoint[1] + 100]

    # Ensure the coordinates fit within image shape
    x_values = [max(0, min(image_shape[1] - 1, x)) for x in x_values]
    y_values = [max(0, min(image_shape[0] - 1, y)) for y in y_values]

    return midpoint, perpendicular_slope, x_values, y_values


def find_farthest_points(mask):
    points = np.column_stack(np.nonzero(mask))
    max_distance = 0
    point1, point2 = None, None

    for pt1, pt2 in itertools.combinations(points, 2):
        distance = np.linalg.norm(pt1 - pt2)
        if distance > max_distance:
            max_distance = distance
            point1, point2 = pt1, pt2

    return point1, point2


# Mask dilate
def minkowski_sum(depth_image, initial_point, depth_value, delta_d, structure_element):
    mask = np.zeros_like(depth_image, dtype=np.uint8)
    mask[initial_point[0], initial_point[1]] = 1

    while True:
        new_mask = cv2.dilate(mask, structure_element)
        depth_mask = (depth_image >= depth_value - delta_d) & (depth_image <= depth_value + delta_d)
        new_mask = new_mask & depth_mask

        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    return mask


# Angle projection
def map_to_minus_90_to_90(angle):
    angle = angle % 360
    if angle < 0:
        angle += 360
    if angle > 180:
        angle -= 360
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180
    return angle


# Vector Angle
def angle_between(v1, v2):
    dot_prod = np.dot(v1, v2)
    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v2)
    cos_angle = dot_prod / (mag_v1 * mag_v2)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)


def calculate_total_difference(grasp_point, length, edges):
    min_difference = float('inf')
    best_angle = None

    # Traverse the angles in the range [-pi/2, pi/2]
    for angle in np.linspace(-np.pi, np.pi, 180):
        for width in range(43, 2, -1):
            # Calculate the rotated point
            p1, p2, p3, p4 = decode_box(angle, grasp_point, length, width)
            line_pixels_bottom = list(bresenham(p4[0], p4[1], p3[0], p3[1]))
            line_pixels_top = list(bresenham(p1[0], p1[1], p2[0], p2[1]))
            bottom_transformed_pixels = [(x + 128, y + 208) for (x, y) in line_pixels_bottom]
            top_transformed_pixels = [(x + 128, y + 208) for (x, y) in line_pixels_top]

            line_pixels_bottom_r = list(bresenham(p3[0], p3[1], p4[0], p4[1]))
            line_pixels_top_r = list(bresenham(p2[0], p2[1], p1[0], p1[1]))
            bottom_transformed_pixels_r = [(x + 128, y + 208) for (x, y) in line_pixels_bottom_r]
            top_transformed_pixels_r = [(x + 128, y + 208) for (x, y) in line_pixels_top_r]

            top_intersection_pixels_r = [pixel for pixel in top_transformed_pixels_r  if edges[pixel[0], pixel[1]]]
            bottom_intersection_pixels_r = [pixel for pixel in bottom_transformed_pixels_r if edges[pixel[0], pixel[1]]]

            top_intersection_pixels = [pixel for pixel in top_transformed_pixels  if edges[pixel[0], pixel[1]]]
            bottom_intersection_pixels = [pixel for pixel in bottom_transformed_pixels if edges[pixel[0], pixel[1]]]

            if len(top_intersection_pixels) > 0 and len(bottom_intersection_pixels) > 0 \
                    and len(bottom_intersection_pixels_r) > 0 and len(top_intersection_pixels_r) > 0:
                #The angle between the two intersection points on the left and the lower long side
                l1 = top_intersection_pixels[0]
                l2 = bottom_intersection_pixels[0]
                left_vector = np.array([l1[0] - l2[0], l1[1] - l2[1]])
                line_vector = np.array([p4[0] - p3[0], p4[1] - p3[1]])
                dot_product = np.dot(line_vector, left_vector)
                line_vector_length = np.linalg.norm(line_vector)
                left_vector_length = np.linalg.norm(left_vector)
                angle_radians_1 = np.arccos(dot_product / (line_vector_length * left_vector_length))
                angle_diff_1 = np.degrees(angle_radians_1)
                angle_differences_1 = abs(angle_diff_1 - 90)

                # The angle between the two intersection points on the right side and the lower long side
                r1 = top_intersection_pixels_r[0]
                r2 = bottom_intersection_pixels_r[0]
                right_vector = np.array([r1[0] - r2[0], r1[1] - r2[1]])
                dot_product = np.dot(line_vector, right_vector)
                line_vector_length = np.linalg.norm(line_vector)
                right_vector_length = np.linalg.norm(right_vector)
                angle_radians_2 = np.arccos(dot_product / (line_vector_length * right_vector_length))
                angle_diff_2 = np.degrees(angle_radians_2)
                angle_differences_2 = abs(angle_diff_2 - 90)

                angle_differences = angle_differences_1 + angle_differences_2
                if angle_differences < min_difference:
                    min_difference = angle_differences
                    best_angle = angle
                break

            else:
                continue

    return best_angle, min_difference


def is_point_in_polygon(x, y, poly_x, poly_y):
    n = len(poly_x)
    inside = False
    xinters = 0
    p1x, p1y = poly_x[0], poly_y[0]
    for i in range(n + 1):
        p2x, p2y = poly_x[i % n], poly_y[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


# Finding minimum rectangle in the grasp
def get_intersection_and_rectangle(p1, p2, p3, p4, mask):

    rect_x = [p1[0] + 128, p2[0] + 128, p3[0] + 128, p4[0] + 128, p1[0] + 128]
    rect_y = [p1[1] + 208, p2[1] + 208, p3[1] + 208, p4[1] + 208, p1[1] + 208]

    min_x = min(rect_x)
    max_x = max(rect_x)
    min_y = min(rect_y)
    max_y = max(rect_y)

    pixels_inside_rect = set()

    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            if is_point_in_polygon(x, y, rect_x, rect_y):
                pixels_inside_rect.add((x, y))

    pixels_inside_rect = list(pixels_inside_rect)
    transformed_pixels = pixels_inside_rect

    intersection_pixels = [pixel for pixel in transformed_pixels if mask[pixel[0], pixel[1]]]

    if not intersection_pixels:
        return None, None, None

    # Find the minimum bounding box rectangle
    min_rect_min_x = min(pixel[0] for pixel in intersection_pixels)
    min_rect_max_x = max(pixel[0] for pixel in intersection_pixels)
    min_rect_min_y = min(pixel[1] for pixel in intersection_pixels)
    min_rect_max_y = max(pixel[1] for pixel in intersection_pixels)

    # Calculate the center point of the rectangle
    center_x = (min_rect_min_x + min_rect_max_x) / 2
    center_y = (min_rect_min_y + min_rect_max_y) / 2
    center_point = (center_x, center_y)

    # Calculate the longest side length of the minimum enclosing rectangle
    min_rect_width = min_rect_max_x - min_rect_min_x
    min_rect_height = min_rect_max_y - min_rect_min_y
    max_length = max(min_rect_width, min_rect_height)

    return center_point, intersection_pixels, max_length


# Finding the point with minimum depth in the grasp
def get_min_depth_rectangle(p1, p2, p3, p4, depth):

    # Define the boundary points of the original grasp
    rect_x = [p1[0] + 128, p2[0] + 128, p3[0] + 128, p4[0] + 128, p1[0] + 128]
    rect_y = [p1[1] + 208, p2[1] + 208, p3[1] + 208, p4[1] + 208, p1[1] + 208]

    min_x = min(rect_x)
    max_x = max(rect_x)
    min_y = min(rect_y)
    max_y = max(rect_y)

    pixels_inside_rect = set()

    # Iterate over each pixel in the grasp
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            if is_point_in_polygon(x, y, rect_x, rect_y):
                pixels_inside_rect.add((x, y))

    pixels_inside_rect = list(pixels_inside_rect)
    transformed_pixels = pixels_inside_rect

    box_min_depth = 1
    for point in transformed_pixels:
        if depth[int(point[0]), int(point[1])] < box_min_depth:
            box_min_depth = depth[int(point[0]), int(point[1])]

    return box_min_depth


def compute_adjacent_relation(p1, p2, p3, p4, mask, grasp_point, depth):

    line_pixels_left = list(bresenham(p1[0], p1[1], p4[0], p4[1]))
    line_pixels_right = list(bresenham(p2[0], p2[1], p3[0], p3[1]))

    combined_pixels = line_pixels_left+ line_pixels_right

    transformed_pixels = [(x + 128, y + 208) for (x, y) in combined_pixels]
    dis_box_center_depth = depth[int(grasp_point[0] + 128), int(grasp_point[1] + 208)]

    for pixel in transformed_pixels:
        if abs(depth[int(pixel[0]), int(pixel[1])] - dis_box_center_depth) < 0.009:
            return False

    return True


def get_distance_points(input_point, mask):
    max_distance_point = [0, 0]
    max_distance = 0
    indices = np.argwhere(mask ==True)
    for x,y in indices:
        distance = np.sqrt((y- input_point[0][0])**2 + (x- input_point[0][1]) ** 2)
        if max_distance < distance:
            max_distance_point = [x,y]
            max_distance = distance
    return [max_distance_point[1],max_distance_point[0]]


# SAM vis mask
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


# SAM vis points
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


# SAM vis bounding box
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))