import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from utils.dataset_processing.grasp import detect_grasps

warnings.filterwarnings("ignore")


def plot_results(
        fig,
        rgb_img,
        rgb_img_new,
        rgb_img_og,
        grasp_q_img,
        grasp_angle_img,
        depth_img=None,
        no_grasps=1,
        grasp_width_img= None,
        point = None,
        fine_point = None,
        fine_width = None
):
    """
    Plot the output of a network
    :param fig: Figure to plot the output
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of network
    :param grasp_angle_img: Angle output of network
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of network
    :return:
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps, point=point, fine_point = fine_point, fine_width=fine_width)

    plt.ion()
    plt.clf()
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(rgb_img_new)
    ax.set_title('RGB')
    ax.axis('off')

    if depth_img is not None:
        ax = fig.add_subplot(2, 2, 2)
        ax.imshow(depth_img, cmap='gray')
        for g in gs:
            g.plot(ax)
            ax.plot(g.center[1], g.center[0], 'o', color='orange', markersize=3.5)
        ax.set_title('Depth')
        ax.axis('off')

    ax = fig.add_subplot(2, 2, 3)
    ax.imshow(rgb_img_og)
    for g in gs:
        g.plot(ax)
        ax.plot(g.center[1], g.center[0], 'o', color='orange', markersize=3)
    ax.set_title('Grasp')
    ax.axis('off')

    ax = fig.add_subplot(2, 2, 4)
    plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Q')
    ax.axis('off')

    plt.pause(0.1)
    fig.canvas.draw()


def plot_grasp(
        fig,
        grasps=None,
        save=False,
        rgb_img=None,
        grasp_q_img=None,
        grasp_angle_img=None,
        no_grasps=1,
        grasp_width_img=None
):
    """
    Plot the output grasp of a network
    :param fig: Figure to plot the output
    :param grasps: grasp pose(s)
    :param save: Bool for saving the plot
    :param rgb_img: RGB Image
    :param grasp_q_img: Q output of network
    :param grasp_angle_img: Angle output of network
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of network
    :return:
    """
    if grasps is None:
        grasps = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)

    plt.ion()
    plt.clf()

    ax = plt.subplot(111)
    ax.imshow(rgb_img)
    for g in grasps:
        g.plot(ax)
    ax.set_title('Grasp')
    ax.axis('off')

    plt.pause(0.1)
    fig.canvas.draw()

    if save:
        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.savefig('results/{}.png'.format(time))


def save_results(
            rgb_img,
            grasp_q_img,
            grasp_angle_img,
            depth_img=None,
            no_grasps=1,
            grasp_width_img=None,
            point=None,
            fine_point=None,
            fine_width=None
    ):
    """
    Plot the output of a network
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of network
    :param grasp_angle_img: Angle output of network
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of network
    :return:
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps, point=point,
                       fine_point=fine_point, fine_width=fine_width)

    fig = plt.figure(figsize=(10, 10),dpi=600)
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    ax.imshow(rgb_img)

    ax.axis('off')
    fig.savefig('results/rgb.png',bbox_inches='tight', pad_inches=0)

    fig = plt.figure(figsize=(10, 10),dpi=600)
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    ax.imshow(rgb_img)
    for g in gs:
        g.plot(ax)
        ax.plot(g.center[1], g.center[0], 'o', color='orange', markersize=1)

    ax.axis('off')
    fig.savefig('results/grasp.png',bbox_inches='tight', pad_inches=0)

    fig = plt.figure(figsize=(10, 10),dpi=300)
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
    for g in gs:
        # g.plot(ax)
        ax.plot(g.center[1], g.center[0], 'o', color='orange', markersize=1)

    ax.axis('off')
    # plt.colorbar(plot, ax=ax, shrink=0.796)
    fig.savefig('results/quality.png',bbox_inches='tight', pad_inches=0)

    fig = plt.figure(figsize=(10, 10),dpi=300)
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    plot = ax.imshow(grasp_angle_img, cmap='hot', vmin=-np.pi / 2, vmax=np.pi / 2)

    ax.axis('off')
    # plt.colorbar(plot, ax=ax, shrink=0.796)
    fig.savefig('results/angle.png',bbox_inches='tight', pad_inches=0)

    fig = plt.figure(figsize=(10, 10), dpi=300)
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    plot = ax.imshow(grasp_width_img, cmap='hot', vmin=0, vmax=100)
    ax.axis('off')
    # plt.colorbar(plot, ax=ax, shrink=0.796)
    fig.savefig('results/width.png', bbox_inches='tight', pad_inches=0)

    fig.canvas.draw()
    plt.close(fig)


def save_results_s(
            rgb_img,
            grasp_q_img,
            grasp_angle_img,
            depth_img=None,
            no_grasps=1,
            grasp_width_img=None,
            point=None,
            fine_point=None,
            fine_width=None
    ):
    """
    Plot the output of a network
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of network
    :param grasp_angle_img: Angle output of network
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of network
    :return:
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps, point=point,
                       fine_point=fine_point, fine_width=fine_width)

    fig = plt.figure(figsize=(10, 10),dpi=600)
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    ax.imshow(rgb_img)
    for g in gs:
        g.plot(ax)
        ax.plot(g.center[1], g.center[0], 'o', color='orange', markersize=15)

    ax.axis('off')
    fig.savefig('results/grasp_s.png',bbox_inches='tight', pad_inches=0)
    fig.canvas.draw()
    plt.close(fig)

def save_results_o(
            rgb_img,

            grasp_q_img,
            grasp_angle_img,
            depth_img=None,
            no_grasps=1,
            grasp_width_img=None,
            point=None,
            fine_point=None,
            fine_width=None
    ):
    """
    Plot the output of a network
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of network
    :param grasp_angle_img: Angle output of network
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of network
    :return:
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps, point=point,
                       fine_point=fine_point, fine_width=fine_width)

    fig = plt.figure(figsize=(10, 10),dpi=600)
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    ax.imshow(rgb_img)
    for g in gs:
        g.plot(ax)
        ax.plot(g.center[1], g.center[0], 'o', color='orange', markersize=15)

    ax.axis('off')
    fig.savefig('results/grasp_o.png',bbox_inches='tight', pad_inches=0)


    fig.canvas.draw()
    plt.close(fig)

def save_results_f(
            rgb_img,
            rgb_img_new,
            rgb_img_og,
            grasp_q_img,
            grasp_angle_img,
            depth_img=None,
            no_grasps=1,
            grasp_width_img=None,
            point=None,
            fine_point=None,
            fine_width=None
    ):
    """
    Plot the output of a network
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of network
    :param grasp_angle_img: Angle output of network
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of network
    :return:
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps, point=point,
                       fine_point=fine_point, fine_width=fine_width)

    fig = plt.figure(figsize=(10, 10),dpi=600)
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    ax.imshow(rgb_img_new)

    ax.axis('off')
    fig.savefig('results/rgb.png',bbox_inches='tight', pad_inches=0)

    fig = plt.figure(figsize=(10, 10),dpi=600)
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    ax.imshow(rgb_img_og)

    ax.axis('off')
    fig.savefig('results/rgb_og.png',bbox_inches='tight', pad_inches=0)

    fig = plt.figure(figsize=(10, 10), dpi=600)
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    ax.imshow(rgb_img_og)
    for g in gs:
        g.plot(ax)
        ax.plot(g.center[1], g.center[0], 'o', color='orange', markersize=10)

    ax.axis('off')
    fig.savefig('results/grasp_f.png',bbox_inches='tight', pad_inches=0)
    fig.canvas.draw()
    plt.close(fig)
