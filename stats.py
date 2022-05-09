# class to get test stats from tello connection

from datetime import datetime

class Stats:
    def __init__(me, command, id):
        me.command = command
        me.response = None
        me.id = id

        me.start_time = datetime.now()
        me.end_time = None
        me.duration = None

    def add_response(me, response):
        me.response = response
        me.end_time = datetime.now()
        me.duration = me.get_duration()
        # me.print_stats()

    def get_duration(me):
        diff = me.end_time - me.start_time
        return diff.total_seconds()

    def print_stats(me):
        print('\nid: %s' % me.id)
        print('command: %s' % me.command)
        print('response: %s' % me.response)
        print('start time: %s' % me.start_time)
        print('end_time: %s' % me.end_time)
        print('duration: %s\n' % me.duration)

    def got_response(me):
        if me.response is None:
            return False
        else:
            return True

    def return_stats(me):
        str = ''
        str +=  '\nid: %s\n' % me.id
        str += 'command: %s\n' % me.command
        str += 'response: %s\n' % me.response
        str += 'start time: %s\n' % me.start_time
        str += 'end_time: %s\n' % me.end_time
        str += 'duration: %s\n' % me.duration
        return str