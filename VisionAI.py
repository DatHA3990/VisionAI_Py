import numpy as np
import cv2 as cv
import os

SCALAR_SIZE = 3

COLOR_NAME = 0
COLOR_BGR = 1
COLOR_DIFFERENCE = 2
COLOR_ACCURACY = 3

DIR_TRAIN_DATA = "train_data"
DIR_TEST_DATA = "test_data"

DIR_NAME = "name"
FILE_NAME = "name.txt"

DIR_IMAGE = "images"

def new_color(name="", bgr=[0] * SCALAR_SIZE, difference=[0] * SCALAR_SIZE, accuracy=1):
	return [name, bgr, difference, accuracy]

def get_bgr_difference(bgr):
	return [bgr[0] - bgr[1], bgr[1] - bgr[2], bgr[2] - bgr[0]]

def get_color_accuracy(color1, color2):
	return 1 - (np.average(abs(np.subtract(color1, color2)))) / 255

def get_color(image, colors):
	difference = get_bgr_difference(np.average(np.average(image, axis=0), axis=0))
	accuracy = []
	for color in colors:
		accuracy.append(get_color_accuracy(color[COLOR_DIFFERENCE], difference))
	color_accuracy = max(accuracy)
	color_match = colors[accuracy.index(color_accuracy)]
	color_match[COLOR_ACCURACY] = color_accuracy
	return color_match

def get_trained_colors():
	color = []
	for train_data_folder in os.walk(DIR_TRAIN_DATA):
		for color_name in train_data_folder[1]:
			if color_name != DIR_NAME and color_name != DIR_IMAGE:
				location = DIR_TRAIN_DATA + '/' + color_name + '/'
				bgr = []
				for color_images in os.walk(location + '/' + DIR_IMAGE):
					for image_file in color_images[2]:
						bgr.append(
							np.average(np.average(cv.imread(location + '/' + DIR_IMAGE + '/' + image_file, cv.IMREAD_COLOR), axis=0),
							           axis=0))
				bgr = np.average(bgr, axis=0)
				color.append(new_color(open(location + DIR_NAME + '/' + FILE_NAME).read(), bgr, get_bgr_difference(bgr)))
	return color

def get_position_in_list(myList, v):
	for i, x in enumerate(myList):
		if v in x:
			return i, x.index(v)

def get_target_coordinates(color, image, target_color_name, tolerance):
	color = colors[get_position_in_list(colors, target_color_name)[0]][COLOR_BGR]
	cnt = cv.findContours(cv.inRange(cv.blur(image, (15, 15)), np.subtract(color, tolerance), np.add(color, tolerance)), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[1][0]
	approx = cv.approxPolyDP(cnt, 0.1 * cv.arcLength(cnt, True), True)
	return np.average(np.average(approx, axis=0), axis=0)

def draw_target(image, coordinates):
	cv.circle(image, (int(coordinates[0]), int(coordinates[1])), 5, (255, 0, 255), -1)
	cv.imshow("image", image)
	cv.waitKey(0)

cv.namedWindow("image", cv.WINDOW_AUTOSIZE)
colors = get_trained_colors()
image = cv.imread(DIR_TEST_DATA + '/' + "boiler3.jpg", cv.IMREAD_COLOR)
coordinates = get_target_coordinates(colors, image, "target", 40)
draw_target(image, coordinates)
input()
cv.destroyAllWindows()