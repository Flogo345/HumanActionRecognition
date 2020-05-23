import os
import asyncio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import threading
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
from os import listdir
import sys

from argparse import ArgumentParser
import json
import numpy as np
import cv2

from MSG3D.msg3dRunningProcessor import MSG3DRunningProcessor
import MSG3D.data_gen.preprocess as preprocess
from lhpes3d.lhpes3dRunningProcessor import LPES3DRunningProcessor, rotate_poses
from lhpes3d.modules.input_reader import VideoReader
from lhpes3d.modules.draw import Plotter3d, draw_poses
from lhpes3d.modules.parse_poses import parse_poses



class RunningProcessor():
    def __init__(self, video, lpes3d_model_path, msg3d_model_path, height_size=256, fx=-1):
        #video
        self.frame_provider = VideoReader(video)
        self.height_size = height_size
        self.fx = fx

        #models
        self.lpes3d_model_path = lpes3d_model_path
        self.msg3d_model_path = msg3d_model_path
        self.lpes3d_model = LPES3DRunningProcessor(lpes3d_model_path)
        self.msg3d_model = MSG3DRunningProcessor(msg3d_model_path)
        self.action_result = "Loading"

        #Plotting
        self.canvas_3d = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.plotter = Plotter3d(self.canvas_3d.shape[:2])
        self.canvas_3d_window_name = 'Canvas 3D'
        cv2.namedWindow(self.canvas_3d_window_name)
        cv2.setMouseCallback(self.canvas_3d_window_name, Plotter3d.mouse_callback)

        file_path = os.path.join('lhpes3d\data', 'extrinsics.json')
        with open(file_path, 'r') as f:
            extrinsics = json.load(f)
        self.R = np.array(extrinsics['R'], dtype=np.float32)
        self.t = np.array(extrinsics['t'], dtype=np.float32)

        self.buffer = []
        self.joint_map_msg3d_lhpes3d = [2, -1, 0, 1, 3, 4, 5, 5, 9, 10, 11, 11, 6, 7, 8, 8, 12, 13, 14, 14, -2, 5, 5, 11, 11]

    def humanActionRecognition(self):
        self.mean_time = 0

        is_video = True
        delay = 1
        esc_code = 27
        p_code = 112
        space_code = 32

        buffer_init_time = 2
        buffer_time = cv2.getTickCount() 
        init_buffer_over = False
         
        msg3d_calculate_frame = 10
        msg3d_count = 0
        
        #Main running loop
        for frame in self.frame_provider:
            msg3d_count = (msg3d_count + 1) % msg3d_calculate_frame
            self.current_time = cv2.getTickCount()

            if frame is None:
                break

            #Preprocess video_stream
            scaled_img, input_scale = self.preprocessFrame(frame)

            #LHPES3D
            self.runLhpes3d(scaled_img, input_scale)


            #MSG3D
            if (len(self.buffer) >= 45 and msg3d_count == 0 ):
                init_buffer_over = True
                self.runMsg3d()
                

            #Display results
            self.displayFrame(frame)
            

            #Handle UserInput
            key = cv2.waitKey(delay)
            if key == esc_code:
                break
            if key == p_code:
                if delay == 1:
                    delay = 0
                else:
                    delay = 1
            if delay == 0 or not is_video:  # allow to rotate 3D canvas while on pause
                key = 0
                while (key != p_code
                       and key != esc_code
                       and key != space_code):
                    self.plotter.plot(self.canvas_3d, self.poses_3d, self.edges)
                    cv2.imshow(self.canvas_3d_window_name, self.canvas_3d)
                    key = cv2.waitKey(33)
                if key == esc_code:
                    break
                else:
                    delay = 1


    def runLhpes3d(self, scaled_img, input_scale):
        #Determin Value with model 
        inference_result = self.lpes3d_model(scaled_img)
        self.poses_3d, self.poses_2d = parse_poses(inference_result, input_scale, 8, self.fx, True)
        self.edges = []
        if len(self.poses_3d):
            self.poses_3d = rotate_poses(self.poses_3d, self.R, self.t)
            poses_3d_copy = self.poses_3d.copy()
            x = poses_3d_copy[:, 0::4]
            y = poses_3d_copy[:, 1::4]
            z = poses_3d_copy[:, 2::4]
            self.poses_3d[:, 0::4], self.poses_3d[:, 1::4], self.poses_3d[:, 2::4] = -z, x, -y

            self.poses_3d = self.poses_3d.reshape(self.poses_3d.shape[0], 19, -1)[:, :, 0:3]
            self.edges = (Plotter3d.SKELETON_EDGES + 19 * np.arange(self.poses_3d.shape[0]).reshape((-1, 1, 1))).reshape((-1, 2))

        self.buffer.append(self.poses_3d)
        if (len(self.buffer) > 45):
            self.buffer.pop(0)

    def runMsg3d(self):
        msg3d_input = np.zeros(shape=(1, 3, len(self.buffer), 25, 2), dtype= np.float)
        for chanels in range(len(msg3d_input[0])):
            for frame in range(len(msg3d_input[0][chanels])):
                for joint in range(len(msg3d_input[0][chanels][frame])):
                    if (len(self.buffer[frame]) > 0):
                        persons_arr = np.zeros(shape=(2))
                        msg3d_input[0][chanels][frame][joint] = persons_arr
                        for person in range(min(len(self.buffer[frame]), 2)):
                            index_to_read = self.joint_map_msg3d_lhpes3d[joint]
                            if (index_to_read == -1):
                                msg3d_input[0][chanels][frame][joint][person] = (self.buffer[frame][person][2][chanels] + 0.5 * ((self.buffer[frame][person][9][chanels] + 0.5 * (self.buffer[frame][person][3][chanels] - self.buffer[frame][person][9][chanels])) - self.buffer[frame][person][2][chanels])) / 1
                            elif (index_to_read == -2):
                                msg3d_input[0][chanels][frame][joint][person] = (self.buffer[frame][person][9][chanels] + 0.5 * (self.buffer[frame][person][3][chanels] - self.buffer[frame][person][9][chanels])) / 1
                            else:
                                msg3d_input[0][chanels][frame][joint][person] = self.buffer[frame][person][index_to_read][chanels] / 1
        

        np.set_printoptions(threshold=sys.maxsize)
        #msg3d_input = preprocess.pre_normalization(msg3d_input)
        msg3d_input = torch.from_numpy(msg3d_input)
        msg3d_input = msg3d_input.float().cuda()
        self.action_result = self.msg3d_model(msg3d_input)
        print(self.action_result)

    def displayFrame(self, frame):
        self.plotter.plot(self.canvas_3d, self.poses_3d, self.edges) #canvas_3d = img, poses_3d = vertices, edges = edges
        cv2.imshow(self.canvas_3d_window_name, self.canvas_3d)

        draw_poses(frame, self.poses_2d)
        self.current_time = (cv2.getTickCount() - self.current_time) / cv2.getTickFrequency()
        if self.mean_time == 0:
            self.mean_time = self.current_time
        else:
            self.mean_time = self.mean_time * 0.95 + self.current_time * 0.05
        cv2.putText(frame, 'FPS: {}'.format(int(1 / self.mean_time * 10) / 10),
                    (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        cv2.putText(frame, 'Action: {}'.format(self.action_result),
                    (40, 45), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        cv2.imshow('ICV 3D Human Pose Estimation', frame)


    def preprocessFrame(self, frame):
        input_scale = self.height_size / frame.shape[0]
        scaled_img = cv2.resize(frame, dsize=None, fx=input_scale, fy=input_scale)
        scaled_img = scaled_img[:, 0:scaled_img.shape[1] - (scaled_img.shape[1] % 8)]  #better to pad, but cut out for demo
        if self.fx < 0:  # Focal length is unknown
            self.fx = np.float32(0.8 * frame.shape[1])

        return scaled_img, input_scale

    def writeSkeletonFile(self):
        skeleton_file = ""
        skeleton_file += str(len(self.buffer)) + "\n"
        for temp_frame in range(len(self.buffer)):
            skeleton_file += str(len(self.buffer[temp_frame])) + "\n"
            for person in range(len(self.buffer[temp_frame])):
                skeleton_file += "0 0 0 0 0 0 0 0 0 0" + "\n"
                skeleton_file += str(len(self.buffer[temp_frame][person])) + "\n"
                for joint in range(len(msg3d_input[0][0][temp_frame])):
                    for channel in msg3d_input[0]:
                        skeleton_file += str(channel[temp_frame][joint][person]) + " "
                    skeleton_file += "0 0 0 0 0 0 0 0 0\n"
        self.write_file("skeleton.skeleton", skeleton_file)

    def write_file(self, name, text):
        f = open(name, "a")
        f.write(text)
        f.close()

    def msg3dTest(self):
        input =  torch.from_numpy(np.load(r'./MSG3D/data/Out/xset/val_data_joint.npy'))
        if torch.cuda.is_available():
            input = input.cuda()

        out = self.msg3d_model(input)
        print(out)


def main():
    processor = RunningProcessor(video=r'D:\Repos\HumanActionRecognition\ballthrow.mp4', lpes3d_model_path=r'..\..\..\human-pose-estimation-3d.pth', msg3d_model_path=r'..\..\..\ntu120-xset-joint.pt')
    processor.humanActionRecognition()

if __name__ == '__main__':
    main()