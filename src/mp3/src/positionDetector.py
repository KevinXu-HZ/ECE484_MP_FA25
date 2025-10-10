#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from cv_bridge import CvBridge
from controller import bicycleModel
from sensor_msgs.msg import Image

class PositionDetector(Node):
    def __init__(self, resolution):
        super().__init__('position_detector')
        self.resolution = resolution

        self.model = bicycleModel()

        self.birdsEyeSub = self.create_subscription(
            Image, "/mp0/BirdsEye", self.detectPosition, 1)
        self.debugPub = self.create_publisher(Image, "/mp0/Debug", 1)
        self.finalPub = self.create_publisher(Image, "/mp0/PedestrianAnnotate", 1)

        self.position = 0

        self.cvBridge = CvBridge()
        self.finalIm = None

    def detectPosition(self, data):
        """
            Uses Hough circles to detect the location of the pedestrian in the
            birds eye view image and saves it.

            Input: data - the birds eye view Lidar image

            Output: None

            Side Effects: Changes the detected position of the pedestrian
        """
        # Use Hough Circle detector to detect pedestrian
        # Reference: https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/hough_circle/hough_circle.html

        img = self.cvBridge.imgmsg_to_cv2(data, 'mono8').astype(np.uint8)
        x_max = img.shape[1]
        y_max = img.shape[0]

        processedImg = self.preprocessImg(img)

        self.debugPub.publish(self.cvBridge.cv2_to_imgmsg(processedImg, 'mono8'))

        # set center distance at only one major circle is detected
        # detects center and radius of the circle
        # each element will be (x_pos, y_pos, radius)
        circles = cv2.HoughCircles(processedImg, method=cv2.HOUGH_GRADIENT, dp=1,
                                    minDist=processedImg.shape[0]*0.95, param1=200,
                                    param2=7, minRadius=2, maxRadius=8)

        finalIm = processedImg.copy()
        finalIm[:, :] = 255 - finalIm[:, :]
        finalIm = cv2.cvtColor(finalIm, cv2.COLOR_GRAY2BGR)

        # drawing the road
        cv2.rectangle(finalIm, (int(x_max/2 - 40), 0), (x_max, y_max), (128, 128, 128), -1)
        # drawing lane divider
        cv2.rectangle(finalIm, (int(x_max/2 + 50), 0), (int(x_max/2 + 52), int(y_max/9)), (0, 255, 255), -1)
        cv2.rectangle(finalIm, (int(x_max/2 + 50), 3*int(y_max/9)), (int(x_max/2 + 52), 6*int(y_max/9)), (0, 255, 255), -1)
        cv2.rectangle(finalIm, (int(x_max/2 + 50), 8*int(y_max/9)), (int(x_max/2 + 52), y_max), (0, 255, 255), -1)

        # draw the car for reference
        cv2.rectangle(finalIm, (int(x_max/2 - 15), int(y_max - y_max/60)),
                                (int(x_max/2 + 35), y_max - 1), (0, 0, 255), -1)

        # if a circle is detected save the position
        if circles is not None and circles[0] is not None:
            circle = circles[0][0]

            center = (int(round(circle[0])), int(round(circle[1])))

            # drawing the pedestrian position on the image
            cv2.circle(finalIm, center, int(round(circle[2])), (0, 0, 255), -1, 8, 0)

            # drawing shortest distance to pedestrian from car and surrounding region
            cv2.arrowedLine(finalIm, (x_max/2, y_max), center, (0, 0, 0))
            radius = np.linalg.norm(np.array(center) - np.array((x_max/2, y_max)))
            cv2.circle(finalIm, (x_max/2, y_max), int(round(radius)), (255, 0, 0))


            # the position of the car is always at position (x_max/2, y_max) in the image
            # the image cooradinate axis us placed there - hence we need to align
            # the position correctly for the transformation to work smoothly
            ped_img_x = center[0] - x_max/2
            ped_img_y = y_max - center[1]

            self.position = (ped_img_x, ped_img_y)

        else:
            # no pedestrian was detected
            self.position = 0

        # display annotated pedestrian image
        self.finalIm = finalIm

    def preprocessImg(self, img):
        """
            Preprocesses the birds eye view lidar image so that we have a better
            image for circle detection.

            Input: img - birds eye view lidar image

            Output: processed image
        """
        kernel = np.array([[0, 1, 0],
                           [0, 1, 0],
                           [0, 1, 0]], np.uint8)

        # dilate to thicken lines
        dilated_img = cv2.dilate(img, kernel, iterations=8)

        # erode to smoothen the lines
        eroded_img = cv2.erode(dilated_img, kernel, iterations=8)

        # blur to enchance regions
        blurred_img = cv2.GaussianBlur(eroded_img, (5, 5), 9)

        # threshold so that any regions that matter are accounted for
        _, threshold_img = cv2.threshold(blurred_img, 1, 255, cv2.THRESH_BINARY)

        return threshold_img

    def getPosition(self):
        if self.finalIm is not None:
            self.finalPub.publish(self.cvBridge.cv2_to_imgmsg(self.finalIm, 'bgr8'))
        return self.position
