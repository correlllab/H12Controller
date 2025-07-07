import time
import numpy as np
import tkinter as tk

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from channel_interface import HandSubscriber, HandPublisher

class HandController:
    def __init__(self):
        # initialize channel
        ChannelFactoryInitialize()

        # channel interface
        self.hand_subscriber = HandSubscriber()
        self.hand_publisher = HandPublisher()
        self.dt =  self.hand_publisher.dt

    @property
    def q_right(self):
        '''
        Get the right hand angles.
        0 for close, 1 for open
        #index 0:pinky
        #index 1:ring
        #index 2:middle
        #index 3:index
        #index 4:thumb
        #index 5:thumb angle
        '''
        return self.hand_subscriber.q_right

    @property
    def q_left(self):
        '''
        Get the left hand angles.
        0 for close, 1 for open
        #index 0:pinky
        #index 1:ring
        #index 2:middle
        #index 3:index
        #index 4:thumb
        #index 5:thumb angle
        '''
        return self.hand_subscriber.q_left

    def ctrl_right(self, right_arr):
        '''
        Control the right hand with right angles.
        0 for close, 1 for open
        #index 0:pinky
        #index 1:ring
        #index 2:middle
        #index 3:index
        #index 4:thumb
        #index 5:thumb angle
        '''
        assert(len(right_arr) == 6), 'Right angles must be of length 6.'
        self.hand_publisher.q[:6] = right_arr

    def ctrl_left(self, left_arr):
        '''
        Control the left hand with left angles.
        0 for close, 1 for open
        #index 0:pinky
        #index 1:ring
        #index 2:middle
        #index 3:index
        #index 4:thumb
        #index 5:thumb angle
        '''
        assert(len(left_arr) == 6), 'Left angles must be of length 6.'
        self.hand_publisher.q[6:] = left_arr

    def ctrl(self, right_arr, left_arr):
        '''
        Control the hand with right and left angles.
        0 for close, 1 for open
        #index 0:pinky
        #index 1:ring
        #index 2:middle
        #index 3:index
        #index 4:thumb
        #index 5:thumb angle
        '''
        assert(len(right_arr) == 6 and len(left_arr) == 6), 'Right and left angles must be of length 6.'
        self.ctrl_right(right_arr)
        self.ctrl_left(left_arr)

if __name__ == '__main__':
    hand_controller = HandController()

    root = tk.Tk()
    root.title('Hand Controller')
    root.geometry('600x400')

    # pack sliders side by side
    left_frame = tk.Frame(root)
    right_frame = tk.Frame(root)  # Commented out for now
    left_frame.pack(side=tk.LEFT, padx=10, pady=10)
    right_frame.pack(side=tk.RIGHT, padx=10, pady=10)  # Commented out for now

    # left hand sliders
    slider_l0 = tk.Scale(left_frame, label="Left Pinky",
                         from_=0.0, to=1.0, resolution=0.01,
                         orient=tk.HORIZONTAL, length=250)
    slider_l1 = tk.Scale(left_frame, label="Left Ring",
                         from_=0.0, to=1.0, resolution=0.01,
                         orient=tk.HORIZONTAL, length=250)
    slider_l2 = tk.Scale(left_frame, label="Left Middle",
                         from_=0.0, to=1.0, resolution=0.01,
                         orient=tk.HORIZONTAL, length=250)
    slider_l3 = tk.Scale(left_frame, label="Left Index",
                         from_=0.0, to=1.0, resolution=0.01,
                         orient=tk.HORIZONTAL, length=250)
    slider_l4 = tk.Scale(left_frame, label="Left Thumb",
                         from_=0.0, to=1.0, resolution=0.01,
                         orient=tk.HORIZONTAL, length=250)
    slider_l5 = tk.Scale(left_frame, label="Left Angle",
                         from_=0.0, to=1.0, resolution=0.01,
                         orient=tk.HORIZONTAL, length=250)
    slider_l0.pack(in_=left_frame, pady=5)
    slider_l1.pack(in_=left_frame, pady=5)
    slider_l2.pack(in_=left_frame, pady=5)
    slider_l3.pack(in_=left_frame, pady=5)
    slider_l4.pack(in_=left_frame, pady=5)
    slider_l5.pack(in_=left_frame, pady=5)

    # right hand sliders
    slider_r0 = tk.Scale(right_frame, label="Right Pinky",
                         from_=0.0, to=1.0, resolution=0.01,
                         orient=tk.HORIZONTAL, length=250)
    slider_r1 = tk.Scale(right_frame, label="Right Ring",
                         from_=0.0, to=1.0, resolution=0.01,
                         orient=tk.HORIZONTAL, length=250)
    slider_r2 = tk.Scale(right_frame, label="Right Middle",
                         from_=0.0, to=1.0, resolution=0.01,
                         orient=tk.HORIZONTAL, length=250)
    slider_r3 = tk.Scale(right_frame, label="Right Index",
                         from_=0.0, to=1.0, resolution=0.01,
                         orient=tk.HORIZONTAL, length=250)
    slider_r4 = tk.Scale(right_frame, label="Right Thumb",
                         from_=0.0, to=1.0, resolution=0.01,
                         orient=tk.HORIZONTAL, length=250)
    slider_r5 = tk.Scale(right_frame, label="Right Angle",
                         from_=0.0, to=1.0, resolution=0.01,
                         orient=tk.HORIZONTAL, length=250)
    slider_r0.pack(in_=right_frame, pady=5)
    slider_r1.pack(in_=right_frame, pady=5)
    slider_r2.pack(in_=right_frame, pady=5)
    slider_r3.pack(in_=right_frame, pady=5)
    slider_r4.pack(in_=right_frame, pady=5)
    slider_r5.pack(in_=right_frame, pady=5)

    root.update()

    while True:
        start_time = time.time()
        root.update()
        # get right and left states
        left_arr = np.array([slider_l0.get(), slider_l1.get(), slider_l2.get(),
                             slider_l3.get(), slider_l4.get(), slider_l5.get()])
        right_arr = np.array([slider_r0.get(), slider_r1.get(), slider_r2.get(),
                              slider_r3.get(), slider_r4.get(), slider_r5.get()])
        # control the hand
        hand_controller.ctrl(right_arr, left_arr)

        time.sleep(max(0, hand_controller.dt - (time.time() - start_time)))
