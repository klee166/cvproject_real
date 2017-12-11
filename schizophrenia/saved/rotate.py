from PIL import Image
import logging
import argparse
import cv2
import numpy as np
import sys
# add argument
# logging.basicConfig(
#     format='%(asctime)s %(levelname)s: %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

# parser = argparse.ArgumentParser(description="CV assignment")
# parser.add_argument("--input", default=0, type=str, required=True,
#                     help="Set if it is cuda or not (0 = not cuda)(1 = cuda)")
# parser.add_argument("--rotate", default=0, type=int, required=True,
#                     help="Set if it is cuda or not (0 = not cuda)(1 = cuda)")
# parser.add_argument("--save", default=0, type=int, required=True,
#                     help="Set if it is cuda or not (0 = not cuda)(1 = cuda)")


# args = parser.parse_args()


# im = cv2.imread(args.input,0)
# rows, cols = im.shape
# print im

# M = cv2.getRotationMatrix2D((cols/2, rows/2),args.rotate,1)
# dst = cv2.warpAffine(im, M,(cols,rows))
# if args.save == 1:
# 	cv2.imwrite(args.input, dst)

path = sys.argv[1]
im = cv2.imread(path,0)
print im.shape
cropped = im[34:210, 0:200]
print cropped.shape
cv2.imwrite(path, cropped)
