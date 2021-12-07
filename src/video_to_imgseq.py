import argparse
import cv2
from pathlib import Path
import os

parser = argparse.ArgumentParser()
parser.add_argument('video_path', type=str)
parser.add_argument('--w', type=int, default=640)
parser.add_argument('--h', type=int, default=360)
args = parser.parse_args()

# create output folder
p = Path(args.video_path)
fname = str(p.name).split('.')[0]
save_dir = os.path.join(str(p.parent), fname)
os.makedirs(save_dir, exist_ok=True)

# read video
video = cv2.VideoCapture(args.video_path)
frame_num = 0
while True:
    ret, frame = video.read()
    if not ret:
        break
    frame = cv2.resize(frame, (args.w, args.h))
    save_path = os.path.join(save_dir, '%05d.jpg' % (frame_num))
    cv2.imwrite(save_path, frame)
    frame_num += 1