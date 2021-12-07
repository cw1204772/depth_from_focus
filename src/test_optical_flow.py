import os
import numpy as np
import cv2


data_dir = '../depth_from_focus_data2/Figure1/keyboard/'
fpath1 = os.path.join(data_dir, '00.jpg')
fpath2 = os.path.join(data_dir, '01.jpg')


def display_flow(flow):
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[...,1] = 255
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang * 180 / np.pi / 2
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('flow', bgr)
    k = cv2.waitKey(0)


# def warp(img, flow):




if __name__ == '__main__':

    prev = cv2.imread(fpath1)
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    next = cv2.imread(fpath2)
    next = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame1', prev)
    k = cv2.waitKey(0)
    cv2.imshow('frame2', next)
    k = cv2.waitKey(0)

    flow = cv2.calcOpticalFlowFarneback(
        prev, next, None, 
        pyr_scale=0.5, 
        levels=3, 
        winsize=15, 
        iterations=3, 
        poly_n=5, 
        poly_sigma=1.2, 
        flags=0)

    display_flow(flow)

    # prev_warped = warp(prev, flow)
    h, w = flow.shape[:2]
    # flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    prev_warped = cv2.remap(next, flow, None, cv2.INTER_LINEAR)
    cv2.imwrite('prev_warped.jpg', prev_warped)
    cv2.imwrite('prev.jpg', prev)
