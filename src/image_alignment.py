from __future__ import print_function
from utils import *

import argparse
import cv2
import numpy as np
from scipy.interpolate import interpn

def compute_descriptors(imGray):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(imGray, None)
    print("keypoints: {}, descriptors: {}".format(len(keypoints), descriptors.shape))
    return keypoints, descriptors

def create_matcher(trees, checks):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=trees)
    search_params = dict(checks=checks)
    
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    return matcher

def find_good_matches_loc(matcher, keypoints1, descriptors1, keypoints2, descriptors2, factor):
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    
    for m, n in matches:
        if m.distance < factor * n.distance:
            good_matches.append(m)
            
    points1 = np.float32([keypoints1[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)
    return good_matches, points1, points2

def apply_homography(homography_curr2ref, img1, img2, points1, points2):
    # warp img2 to img1
    height, width, channels = img1.shape
    homography, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
    homography_curr2ref = np.dot(homography_curr2ref, homography)
    aligned_img2 = cv2.warpPerspective(img2, homography_curr2ref, (width, height))
    return aligned_img2, homography_curr2ref

def align_im2_to_im1_homography(homography_curr2ref, img1, img2):
    img1Gray = convert_to_grayscale(img1)
    img2Gray = convert_to_grayscale(img2)

    keypoints1, descriptors1 = compute_descriptors(img1Gray)
    keypoints2, descriptors2 = compute_descriptors(img2Gray)

    matcher = create_matcher(trees=5, checks=50)
    good_matches, points1, points2 = find_good_matches_loc(matcher, keypoints1, descriptors1, keypoints2, descriptors2, factor=0.7)
    
    imMatches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=2)
    aligned_img2, homography_curr2ref = apply_homography(homography_curr2ref, img1, img2, points1, points2)

    return imMatches, aligned_img2, homography_curr2ref

def apply_flow(flow_curr2ref, flow, img1, img2):
    h, w = flow.shape[0], flow.shape[1]
    x, y = np.arange(w), np.arange(h)
    X, Y = np.meshgrid(x, y)
    im1_points = flow_curr2ref.copy()    # flow-warped coord from ref to curr (im1)
    im1_points[:, :, 0] += X
    im1_points[:, :, 1] += Y
    flow_u = interpn((y, x), flow[:, :, 0], im1_points[:, :, ::-1], bounds_error=False)
    flow_v = interpn((y, x), flow[:, :, 1], im1_points[:, :, ::-1], bounds_error=False)

    flow_curr2ref[:, :, 0] += flow_u
    flow_curr2ref[:, :, 1] += flow_v

    im2_points = flow_curr2ref.copy()    # flow-warped coord from ref to curr (im2)
    im2_points[:, :, 0] += X
    im2_points[:, :, 1] += Y
    img2_warped = cv2.remap(img2, im2_points, None, cv2.INTER_LINEAR)

    return img2_warped, flow_curr2ref

def display_flow(flow):
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[...,1] = 255
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang * 180 / np.pi / 2
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def align_im2_to_im1_flow(flow_curr2ref, img1, img2, flow=None):
    img1Gray = convert_to_grayscale(img1)
    img2Gray = convert_to_grayscale(img2)

    if flow is None:
        flow = cv2.calcOpticalFlowFarneback(
            img1Gray, img2Gray, None, 
            pyr_scale=0.5, levels=3, winsize=15, iterations=3, 
            poly_n=5, poly_sigma=1.2, flags=0)
    else:
        h, w = img1.shape[0], img1.shape[1]
        # flow = flow[:h, :w]
        if flow.shape[0] != h or flow.shape[1] != w:
            flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)

    aligned_img2, flow_curr2ref = apply_flow(flow_curr2ref, flow, img1, img2)
    flowVisual = display_flow(flow)

    return flowVisual, aligned_img2, flow_curr2ref

def main(img_path, flow_path, save_path, match_path, method):
    all_files = find_all_files(img_path)
    if len(flow_path) != 0:
        all_flow_files = find_all_files(flow_path, ext='.npy')
        print(flow_path)
    
    img1 = read_image(img_path + all_files[0])
    save_image(save_path, "align_%05d.jpg" % (0), img1)

    h, w, c = img1.shape
    homography_curr2ref = np.eye(3)
    flow_curr2ref = np.zeros((h, w, 2), dtype=np.float32)
    for i in range(len(all_files)-1):
        img1_path = img_path + all_files[i]
        img2_path = img_path + all_files[i+1]
        if len(flow_path) != 0:
            flow1_path = flow_path + all_flow_files[i]
        match_save_as = "matches_%05d.jpg" % (i+1)
        flow_save_as  = "flow_%05d.jpg" % (i+1)
        align_save_as = "align_%05d.jpg" % (i+1)
        
        print("Reading a source image : ", img1_path)
        img1 = read_image(img1_path)
    
        print("Reading a target image : ", img2_path)
        img2 = read_image(img2_path)

        if len(flow_path) != 0:
            flow1 = read_flow(flow1_path)
        else:
            flow1 = None
    
        print("Aligning images ...")
        if method == 'homography':
            imMatches, aligned_img2, homography_curr2ref = align_im2_to_im1_homography(homography_curr2ref, img1, img2)
        elif method == 'flow' or method == 'RAFT':
            flowVisual, aligned_img2, flow_curr2ref = align_im2_to_im1_flow(flow_curr2ref, img1, img2, flow1)

        if method == 'homography':
            print("Saving an feature matching image : ", save_path)
            save_image(match_path, match_save_as, imMatches)
        elif method == 'flow' or method == 'RAFT':
            print("Saving flow map : ", save_path)
            save_image(match_path, flow_save_as, flowVisual)

        print("Saving an aligned image : ", save_path)
        save_image(save_path, align_save_as, aligned_img2)
        
        print("\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image Alignment")
    parser.add_argument('img_path', help="Dir to input focal stack")
    parser.add_argument('save_path', help="Root dir to result")
    parser.add_argument('match_save_path', help="Dir to frame-by-frame alignment visualization")
    parser.add_argument('--method', required=True, choices=['flow', 'homography', 'RAFT'], help='Alignment method')
    parser.add_argument('--flow_path', default='', help='Dir to optical flow .npy files')
    args = parser.parse_args()
    
    img_path = args.img_path
    flow_path = args.flow_path
    save_path = args.save_path
    match_save_path = args.match_save_path
    method = args.method
    
    main(img_path, flow_path, save_path, match_save_path, method)