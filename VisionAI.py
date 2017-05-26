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
DIR_SAVED_DATA = "saved_data"

DIR_NAME = "name"
FILE_NAME = "name.txt"

DIR_IMAGE = "images"

FILE_SAVED_HSV = "hsv_values.txt"

def nothing(something):
	pass

def get_bgr_difference(bgr):
	return [bgr[0] - bgr[1], bgr[1] - bgr[2], bgr[2] - bgr[0]]

def get_color(image, colors):
	difference = get_bgr_difference(np.average(np.average(image, axis=0), axis=0))
	accuracy = []
	for color in colors:
		accuracy.append(1 - (np.average(abs(np.subtract((color[COLOR_DIFFERENCE], difference))))/ 255))
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
						bgr.append(np.average(np.average(cv.imread(location + '/' + DIR_IMAGE + '/' + image_file, cv.IMREAD_COLOR), axis=0), axis=0))
				bgr = np.average(bgr, axis=0)
				color.append([open(location + DIR_NAME + '/' + FILE_NAME).read(), bgr, get_bgr_difference(bgr), 1])
	return color

def get_position_in_list(myList, v):
	for i, x in enumerate(myList):
		if v in x:
			return i, x.index(v)

def get_target_coordinates_bgr(colors, image, target_color_name, tolerance):
	color = colors[get_position_in_list(colors, target_color_name)[0]][COLOR_BGR]
	cnt = cv.findContours(cv.inRange(cv.blur(image, (15, 15)), np.subtract(color, tolerance), np.add(color, tolerance)), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[1][0]
	approx = cv.approxPolyDP(cnt, 0.1 * cv.arcLength(cnt, True), True)
	return np.average(np.average(approx, axis=0), axis=0)

def get_target_coordinates_hsv(image, tolerance):
	cv.threshold(image,0,255,cv.THRESH_BINARY_INV)
	cv.cvtColor(image, cv.COLOR_BGR2HSV)
	image = cv.inRange(cv.cvtColor(image, cv.COLOR_BGR2HSV), tolerance[0], tolerance[1])
	return image

def draw_target(image, coordinates):
	cv.circle(image, (int(coordinates[0]), int(coordinates[1])), 5, (255, 0, 255), -1)
	cv.imshow("image", image)
	cv.waitKey(1)

def nothing(something):
	pass

def draw_trackbar_hsv():
	window_name = 'tracker'
	cv.namedWindow(window_name)
	for i in ['h','s','v']:
		for j in range(2):
			cv.createTrackbar(i+str(j), 'tracker', 0, 255, nothing)

def get_trackbar():
	hsv = np.array([[0] * 3] * 2)
	hsv[0][0] = cv.getTrackbarPos('h0','tracker')
	hsv[1][0] = cv.getTrackbarPos('h1', 'tracker')
	hsv[0][1] = cv.getTrackbarPos('s0', 'tracker')
	hsv[1][1] = cv.getTrackbarPos('s1', 'tracker')
	hsv[0][2] = cv.getTrackbarPos('v0', 'tracker')
	hsv[1][2] = cv.getTrackbarPos('v1', 'tracker')
	return hsv

def set_trackbar():
	text_file = open(DIR_SAVED_DATA + '/' + FILE_SAVED_HSV, "r")
	hsv = text_file.read().split(',')
	text_file.close()
	hsv.remove('')
	hsv = [int(i) for i in hsv]
	cv.setTrackbarPos('h0', 'tracker', hsv[0])
	cv.setTrackbarPos('h1', 'tracker', hsv[3])
	cv.setTrackbarPos('s0', 'tracker', hsv[1])
	cv.setTrackbarPos('s1', 'tracker', hsv[4])
	cv.setTrackbarPos('v0', 'tracker', hsv[2])
	cv.setTrackbarPos('v1', 'tracker', hsv[5])


def save_trackbar_hsv(hsv):
	name = DIR_SAVED_DATA + '/' + FILE_SAVED_HSV
	open(name, "w").close()
	text_file = open(name, "w")
	for i in hsv:
		for j in i:
			text_file.write(str(j))
			text_file.write(',')
	text_file.close()

#colors = get_trained_colors()
image = cv.imread(DIR_TEST_DATA + '/' + "boiler3.jpg", cv.IMREAD_COLOR)
#coordinates = get_target_coordinates_bgr(colors, image, "target", 40)
cv.namedWindow("hsv")
draw_trackbar_hsv()
set_trackbar()
while True:
	hsv_val = get_trackbar()
	hsv = get_target_coordinates_hsv(image, hsv_val)
	save_trackbar_hsv(hsv_val)
	cv.imshow("hsv", hsv)
	cv.waitKey(1)
#draw_target(image, coordinates)
input()
cv.destroyAllWindows()