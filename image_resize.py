import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to input image that we'll align to template")
ap.add_argument("-t", "--scale", type=int, required=False, default=[2],
	help="how much the image is to be scaled by")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
y, x, _ = image.shape
print('Original Size: {0}x{1}x3'.format(y,x))
scale = args['scale'][0]
print('Scale factor: {0}'.format(scale))

new_image = cv2.resize(image, (scale*x, scale*y))
new_y, new_x, _ = new_image.shape
print('New Size: {0}x{1}x3'.format(new_y, new_x))

cv2.imwrite(args["image"][:-4]+'_LARGE'+'.png',new_image)
