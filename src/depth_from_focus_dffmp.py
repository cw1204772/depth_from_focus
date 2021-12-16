from gco import cut_grid_graph_simple as cut_simple
from utils import *

import argparse
import cv2
import numpy as np

def focus_stack(aligned_img, gaussian_size, laplacian_size):
    imGray = convert_to_grayscale(aligned_img)
    gaussian_img = cv2.GaussianBlur(imGray, (gaussian_size, gaussian_size), 0)
    laplacian_img = cv2.Laplacian(gaussian_img, cv2.CV_64F, ksize=laplacian_size)
    laplacian_img = np.abs(laplacian_img)
    return laplacian_img

def focus_to_cost(focus_img, gaussian_sigma=13):
    ksize = int(gaussian_sigma * 2 * 3 + 1)
    cost_img = cv2.GaussianBlur(focus_img, (ksize, ksize), gaussian_sigma)
    cost_img = -1 * cost_img
    return cost_img

def focus_measure_cal(cost_volume, sigma=7):
    focus_measure = np.zeros_like(cost_volume)
    ksize = int(sigma * 2 * 3 + 1)

    for i in range(len(cost_volume)):
        focus_img = cost_volume[i]
        focus_measure[i] = focus_img
        if sigma > 0:
            focus_measure[i] = cv2.GaussianBlur(focus_measure[i], (ksize, ksize), sigma)

    return focus_measure

def all_in_focus(img_list, cost_volume, sharpness_sigma, gaussian_size):
    bgr_imgs = np.asarray(img_list)
    
    all_in_focus_img = np.zeros_like(bgr_imgs[0])
    height, width, channels = all_in_focus_img.shape
    
    focus_measure = focus_measure_cal(cost_volume, sharpness_sigma)
    argmax = np.argmax(focus_measure, axis=0)
    
    depth_map = (normalize(argmax) * 255)
    if gaussian_size > 0:
        depth_map = cv2.GaussianBlur(depth_map, (gaussian_size, gaussian_size), 0)
    
    for i in range(height):
        for j in range(width):
            idx = argmax[i, j]
            all_in_focus_img[i, j, :] = bgr_imgs[idx, i, j, :]
    
    return depth_map, all_in_focus_img


def graph_cut(cost_volume, unary_scale, pair_scale, n_iter):
    n = len(cost_volume)
    ii, jj = np.meshgrid(range(n), range(n))
    
    unary_cost = normalize(np.stack(cost_volume, axis=-1)) * unary_scale
    pairwise_cost = np.abs(ii - jj) * pair_scale

    graph_img = cut_simple(unary_cost.astype(np.int32), pairwise_cost.astype(np.int32), n_iter)
    
    height, width = unary_cost.shape[0], unary_cost.shape[1]
    graph_img = graph_img.reshape(height, width)
    graph_img = graph_img / float(n) * 255

    return graph_img

def weighted_median_filter(img):
    wmf_img = np.zeros_like(img)
    
    height, width = img.shape
    kernel_size = len(WEIGHTS)
    MARGIN = int(kernel_size/2)
    medIdx = int((np.sum(WEIGHTS) - 1) / 2)
    
    for i in range(MARGIN, height-MARGIN):
        for j in range(MARGIN, width-MARGIN):
            neighbors = []
            for k in range(-MARGIN, MARGIN+1):
                for l in range(-MARGIN, MARGIN+1):
                    a = img.item(i+k, j+l)
                    w = WEIGHTS[k+MARGIN, l+MARGIN]
                    for _ in range(w):
                        neighbors.append(a)
            neighbors.sort()
            median = neighbors[medIdx]
            wmf_img.itemset((i,j), median)
    
    return wmf_img

def main(args):
    img_path = args.base_path + "align/"
    focus_save_path  = args.base_path + "focus_stack/"
    depth_save_path = args.base_path + "depth_map/"
    all_focus_save_path = args.base_path + "all_in_focus/"
    graph_save_path = args.base_path + "graph_cut_step2/"
    wmf_save_path = args.base_path + "wmf_step2/"
    save_as = "output.jpg"
    save_heatmap_as = 'output_heatmap.jpg'
    focus_save_as = "focus_%05d.jpg"
    focus_save_heatmap_as = "focus_heatmap_%05d.jpg"
    
    img_list = read_images_from_path(img_path)
    if args.reverse_input_order:
        img_list.reverse()
    stacked_focus_imgs = []
    
    print("Stacking focus using LoG ... ")
    for i, aligned_img in enumerate(img_list):        
        laplacian_img = focus_stack(aligned_img, gaussian_size=args.LoG_gaussian_ksize, laplacian_size=args.LoG_laplacian_ksize)
        stacked_focus_imgs.append(laplacian_img)
        
        print("... Saving images ...")
        save_image(focus_save_path, focus_save_as % (i), laplacian_img / 1000. * 255)
        print("... Saving images ...")
        save_heatmap(focus_save_path, focus_save_heatmap_as % (i), laplacian_img / 1000. * 255)

    cost_volume = np.asarray(stacked_focus_imgs)

    print("Extracting focus from each images ...")
    depth_map, all_in_focus_img = all_in_focus(img_list, cost_volume, sharpness_sigma=args.AIF_sharpness_sigma, gaussian_size=args.AIF_depth_ksize)
    print("Saving depth-from-focus image : ", depth_save_path)
    save_image(depth_save_path, save_as, depth_map)
    save_heatmap(depth_save_path, save_heatmap_as, depth_map)
    print("Saving all-in-focus image : ", all_focus_save_path)
    save_image(all_focus_save_path, save_as, all_in_focus_img)

    print("Computing cost from stacked focus...")
    cost_volume = []
    for focus_img in stacked_focus_imgs:
        cost = focus_to_cost(focus_img, gaussian_sigma=args.graphcut_sharpness_sigma)
        cost_volume.append(cost)
    cost_volume = np.asarray(cost_volume)

    print("Obtaining graph-cut depth-map ... It'll take a while ...")
    graph_img = graph_cut(cost_volume, unary_scale=args.graphcut_unary_scale, pair_scale=args.graphcut_pair_scale, n_iter=-1)
    print("Saving graph-cut depth-map : ", graph_save_path)
    save_image(graph_save_path, save_as, graph_img)
    save_heatmap(graph_save_path, save_heatmap_as, graph_img)
    
    print("Obtaining weighted median filtered depth-map ...")
    wmf_img = weighted_median_filter(graph_img)
    print("Saving weighted-median-filtered depth map : ", wmf_save_path)
    save_image(wmf_save_path, save_as, wmf_img)
    save_heatmap(wmf_save_path, save_heatmap_as, wmf_img)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Depth from Focus")
    parser.add_argument('base_path', help="Root dir to result from alignment step")
    parser.add_argument('--LoG_gaussian_ksize', type=int, default=5, help='LoG gaussian kernel size')
    parser.add_argument('--LoG_laplacian_ksize', type=int, default=5, help='LoG laplacian kernel size')
    parser.add_argument('--AIF_sharpness_sigma', type=float, default=7, help='Sharpness gaussian sigma for all-in-focus image')
    parser.add_argument('--AIF_depth_ksize', type=int, default=5, help='Depth gaussian sigma for all-in-focus image')
    parser.add_argument('--graphcut_sharpness_sigma', type=float, default=2, help='Sharpness gaussian sigma for graphcut')
    parser.add_argument('--graphcut_unary_scale', type=float, default=100, help='Unary term weight for graphcut')
    parser.add_argument('--graphcut_pair_scale', type=float, default=1, help='Pairwise term weight for graphcut')
    parser.add_argument('--reverse_input_order', action='store_true', help='Reverse aligned image reading order')
    args = parser.parse_args()
    
    main(args)