import os
import time

import utility_functions
import subprocess

class RosCoreManager(object):
    def __init__(self, mock):
        self.mock = mock
        self.kill_at_end = False

    def start_roscore(self):
        if self.mock:
            return
        subprocess.Popen('roscore')
        time.sleep(1)

    def stop_roscore(self):
        if self.mock:
            return
        if self.kill_at_end:
            os.system("killall roscore")
