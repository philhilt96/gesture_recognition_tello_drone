#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2 as cv

# from gestures.tello_gesture_controller import TelloGestureController
from tello_control import GestureControl, KeyboardControl
from calculations import CalculateFPS

from djitellopy import Tello
# from gestures import *
from tello_control import GestureControl, KeyboardControl
from gesture_recognition import *

import threading

import args

# openCV keyboard mappings
ESC_KEY = 27
SPACE_KEY = 32

# def select_mode(key, mode):
#     number = -1
#     if 48 <= key <= 57:  # 0 ~ 9
#         number = key - 48
#     if key == 110:  # n
#         mode = 0
#     if key == 107:  # k
#         mode = 1
#     if key == 104:  # h
#         mode = 2
#     return number, mode


def main():
    # init global vars
    global gesture_buffer
    global gesture_id
    global args
    # global battery_status

    # get config args and set defaults
    args = args.get_args()
    KEYBOARD_NAV = args.is_keyboard
    GESTURE_NAV = False
    is_drone_up = False

    # Get Input from Tello Camera
    tello = Tello()
    tello.connect()
    tello.streamon()
    camera = tello.get_frame_read()

    # objects for controlling navigation of drone
    gesture_navigation = GestureControl(tello)
    keyboard_controller = KeyboardControl(tello)

    # objects for recognition network model
    gesture_detector = GestureRecognition(args.use_static_image_mode, args.min_detection_confidence,
                                          args.min_tracking_confidence)
    gesture_buffer = GestureBuffer(buffer_len=args.buffer_len)

    # function to handle if drone nave comes from keyboard or gesture recognition
    def tello_control(key, keyboard_controller, gesture_navigation):
        global gesture_buffer

        if KEYBOARD_NAV:
            keyboard_controller.control(key)
        else:
            gesture_navigation.gesture_control(gesture_buffer)


    # set fps with opencv
    cv_fps_calc = CalculateFPS(buffer_len=10)

    # set defaults
    mode = 0
    number = -1

    # tello.move_down(20)

    while True:
        fps = cv_fps_calc.get()

        # Process Key (ESC: end)
        key = cv.waitKey(1) & 0xff
        # exit with ESC keypress
        if key == ESC_KEY:
            break
        # Take off/land with space keypress
        elif key == SPACE_KEY:
            if not is_drone_up:
                # Take-off drone
                tello.takeoff()
                is_drone_up = True

            elif is_drone_up:
                # Land tello
                tello.land()
                is_drone_up = False
        # switch to keyboard controller for up and down
        elif key == ord('k'):
            mode = 0
            KEYBOARD_NAV = True
            GESTURE_NAV = False
            tello.send_rc_control(0, 0, 0, 0)  # Stop moving
        # switch to geture controller
        elif key == ord('g'):
            KEYBOARD_NAV = False
        elif key == ord('n'):
            mode = 1
            GESTURE_NAV = True
            KEYBOARD_NAV = True

        if GESTURE_NAV:
            number = -1
            if 48 <= key <= 57:  # 0 ~ 9
                number = key - 48

        # Image from tello camera
        image = camera.frame

        debug_image, gesture_id = gesture_detector.recognize(image, number, mode)
        gesture_buffer.add_gesture(gesture_id)

        # Start control thread
        threading.Thread(target=tello_control, args=(key, keyboard_controller, gesture_navigation,)).start()
        # threading.Thread(target=tello_battery, args=(tello,)).start()

        debug_image = gesture_detector.draw_info(debug_image, fps, mode, number)
        # battery_status = 1

        # Battery status and image rendering
        # cv.putText(debug_image, "Battery: {}".format(battery_status), (5, 720 - 5),
        #            cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv.imshow('Gesture Recognition', debug_image)

    # land drone and end program
    tello.land()
    tello.end()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
