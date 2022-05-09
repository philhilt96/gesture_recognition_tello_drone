# file for custom connection to Tello to send navigaion comments

import socket
import threading
import time
from stats import Stats

class Tello:
    def __init__(me):
        me.local_ip = ''
        me.local_port = 8889
        me.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # socket for sending cmd
        me.socket.bind((me.local_ip, me.local_port))

        # thread for receiving cmd ack
        me.receive_thread = threading.Thread(target=me._receive_thread)
        me.receive_thread.daemon = True
        me.receive_thread.start()

        me.tello_ip = '192.168.10.1'
        me.tello_port = 8889
        me.tello_address = (me.tello_ip, me.tello_port)
        me.log = []

        me.MAX_TIME_OUT = 15.0

    def send_command(me, command):
        me.log.append(Stats(command, len(me.log)))

        me.socket.sendto(command.encode('utf-8'), me.tello_address)
        print('sending command: %s to %s' % (command, me.tello_ip))

        start = time.time()
        while not me.log[-1].got_response():
            now = time.time()
            diff = now - start

        print('Done!!! sent command: %s to %s' % (command, me.tello_ip))

    def _receive_thread(me):

        while True:
            me.response, ip = me.socket.recvfrom(1024)
            print('from %s: %s' % (ip, me.response))

            me.log[-1].add_response(me.response)

    def get_log(me):
        return me.log