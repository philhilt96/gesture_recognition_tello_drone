# test Tello connection

from tello import Tello
from datetime import datetime

start_time = str(datetime.now())

#takeoff, land, cw, ccw, forwards, backwards, up, down
commands = "command", "takeoff", "flip f", "forward 40", "land"

tello = Tello()
for command in commands:
    tello.send_command(command)