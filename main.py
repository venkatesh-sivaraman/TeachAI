from leaputil import *
import Leap

import time

controller = Leap.Controller()

while True:
    frame = controller.frame()
    loc = leap_one_palm(frame)
    print(loc)
    time.sleep(0.1)
