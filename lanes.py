# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 03:28:07 2019

@author: TP
"""

import cv2
import numpy as np
 
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    print(image.shape) 
    y1 = int(image.shape[0])
    """height, width, no. of channels = shape"""
    y2 = int(y1*3/5)
    """422"""
    x1 = int((y1 - intercept)/slope)
    
    x2 = int((y2 - intercept)/slope)
    return[[x1, y1, x2, y2]]
    
def average_slope_intercept(image, lines):
    left_fit =[]
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        for x1,y1,x2,y2 in line: 
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope <0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis =0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])
    
def canny(image):
   gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
   blur = cv2.GaussianBlur(gray, (5,5), 0)
   canny = cv2.Canny(blur, 50, 150)
   return canny 
"""Canny Edge Detection: An image is represented in 2D coordinate based: X & Y. X axis goes along with the image as width (no. of columns) 
whereas Y axis goes along in the image as height (no. of rows). Product of both total height and width gives the no. of pixels in the image.
Thus, the f(x,y): function of pixel intensities is used to represent the rapid change of X & Y. Canny edge performs a derivative of X & Y.
It measures change in intensity in all directions (Gradient) of X and Y.
If the gradient is greater than the upper threshold set in canny edge detection, then it is considered as the edge. Alternatively, if the gradient
is lower than than lower threshold then it's rejected. If the gradient is in between the threshold then only it is accepted if it is connected to strong edge.
"""
"""Here 50 is represented as lower threshold whereas 150 is represented as high threshold"""
  
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line: 
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image
    
def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550,250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

""" The mask will have zero intensity image. If either X or Y has a zero value, then the overall block will have zero value as binary.
    Thus the image will be completely blocked."""
"""The mask: completely blocked image is filled with triangle"""
    
cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_canny = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, np.array([]), minLineLength=40,maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("result", combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()