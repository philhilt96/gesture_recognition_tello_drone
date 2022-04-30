import getch
from djitellopy import Tello

# Class to handle naviation from gesture
class GestureControl:
    def __init__(self, tello: Tello):
        self.tello = tello
        # check if landing from space keypress
        self._is_landing = False

    def gesture_control(self, gesture_id):
        # print("RECOGNIZED GESTURE:", gesture_id)
        gesture_id = int(gesture_id)


        # get input from gesture_id's here and control navigation
        if not self._is_landing:

            # Isaac's code here - set each rc control value with tello api when recieving gesture input
            # self.forw_back_velocity = 0
            # self.up_down_velocity = 0
            # self.left_right_velocity = 0
            # self.yaw_velocity = 0
            # check forward and stop first to avoid crash

            # handle forward
            if gesture_id == 0:
                print("RECOGNIZED GESTURE: Forward ", gesture_id)
            # handle stop
            elif gesture_id == 1:
                print("RECOGNIZED GESTURE: Stop ", gesture_id)

            # next check for back and up/down and left/right
            # handle back
            if gesture_id == 2:
                print("RECOGNIZED GESTURE: Back ", gesture_id)
            # handle up
            elif gesture_id == 3:
                print("RECOGNIZED GESTURE: UP ", gesture_id)
            # handle down
            elif gesture_id == 4:
                print("RECOGNIZED GESTURE: DOWN ", gesture_id)
            # elif gesture_id == 3:  # LAND
            #     self._is_landing = True
            #     self.forw_back_velocity = self.up_down_velocity = \
            #         self.left_right_velocity = self.yaw_velocity = 0
            #     self.tello.land()
            # handle left
            elif gesture_id == 5:
                print("RECOGNIZED GESTURE: LEFT ", gesture_id)
            # handle right
            elif gesture_id == 6:
                print("RECOGNIZED GESTURE: RIGHT ", gesture_id)
            # idle when gesture is not recognized
            elif gesture_id == -1:
                print(gesture_id)

            # uncomment this once velocities are set
            # self.tello.send_rc_control(self.left_right_velocity, self.forw_back_velocity,
            #                            self.up_down_velocity, self.yaw_velocity)

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


tello = Tello()
tello.connect()
tello.streamon()
gesture_navigation = GestureControl(tello)
print('Here goes...')

while True:
    # add code to handle tello.takeoff()
    key = getch.getch()
    gesture_navigation.gesture_control(key)
    print(key)
    # add code to handle tello.land()