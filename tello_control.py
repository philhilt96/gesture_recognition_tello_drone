# Classes to handle keyboard control and gesture nav

from djitellopy import Tello
# import tello_test as nav
# import sys

# Class to handle naviation from gesture
class GestureControl:
    def __init__(self, tello: Tello):
        self.tello = tello
        # check if landing from space keypress
        self._is_landing = False

    def set_rc(self, for_back = 0, up_down = 0, left_right = 0, yaw = 0) :
        self.tello.send_rc_control(for_back, up_down, left_right, yaw)


    def gesture_control(self, buffer):

        # commands = "command"

        # for command in commands:
        # self.tello.send_command("command")
        gesture_id = buffer.get_gesture()
        # print("RECOGNIZED GESTURE:", gesture_id)


        # get input from gesture_id's here and control navigation
        if not self._is_landing:
            # handle forward
            if gesture_id == 0:
                print("RECOGNIZED GESTURE: Forward ", gesture_id)
                self.forw_back_velocity = 20
            # handle stop
            elif gesture_id == 1:
                print("RECOGNIZED GESTURE: Stop ", gesture_id)
                self.forw_back_velocity = 0
                self.up_down_velocity = 0
                self.left_right_velocity = 0
                self.yaw_velocity = 0
            # next check for back and up/down and left/right
            # handle back
            if gesture_id == 5:
                print("RECOGNIZED GESTURE: Back ", gesture_id)
                self.forw_back_velocity = -20
            # handle up
            elif gesture_id == 2:
                print("RECOGNIZED GESTURE: UP ", gesture_id)
                self.up_down_velocity = 25
            # handle down
            elif gesture_id == 4:
                print("RECOGNIZED GESTURE: DOWN ", gesture_id)
                self.up_down_velocity = -20
            # handle left
            elif gesture_id == 6:
                print("RECOGNIZED GESTURE: LEFT ", gesture_id)
                self.left_right_velocity = 25
            # handle right
            elif gesture_id == 7:
                print("RECOGNIZED GESTURE: RIGHT ", gesture_id)
                self.left_right_velocity = -25
            # idle when gesture is not recognized
            elif gesture_id == -1:
                print(gesture_id)
                # self.forw_back_velocity = 0
                # self.up_down_velocity = 0
                # self.left_right_velocity = 0
                # self.yaw_velocity = 0
            self.set_rc(self.left_right_velocity, self.forw_back_velocity, self.up_down_velocity, self.yaw_velocity)

# handle control from keyboard input
class KeyboardControl:
    def __init__(self, tello: Tello):
        self.tello = tello

    def control(self, key):
        if key == ord('a'):
            # TODO: make sure these values for move_up/move_down are adequate
            self.tello.move_up(20)
        elif key == ord('s'):
            self.tello.move_down(20)