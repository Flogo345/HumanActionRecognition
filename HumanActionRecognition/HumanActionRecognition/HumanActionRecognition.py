import os
import asyncio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import threading
from torch.autograd import Variable
from os import listdir
import sys

from argparse import ArgumentParser
import json
import numpy as np
import cv2
import math

from MSG3D.msg3dRunningProcessor import MSG3DRunningProcessor
import MSG3D.data_gen.preprocess as preprocess
from lhpes3d.lhpes3dRunningProcessor import LPES3DRunningProcessor, rotate_poses
from lhpes3d.modules.input_reader import VideoReader
from lhpes3d.modules.draw import Plotter3d, draw_poses
from lhpes3d.modules.parse_poses import parse_poses

kinect_depth_h_fov = 70.6
kinect_depth_v_fov = 60.0
average_z_ntu = 3.0
average_z_hlpes3d = 100.0
lhpes3d_x_res = 1280
lhpes3d_y_res = 720

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

        file_path = os.path.join('lhpes3d/data', 'extrinsics.json')
        with open(file_path, 'r') as f:
            extrinsics = json.load(f)
        self.R = np.array(extrinsics['R'], dtype=np.float32)
        self.t = np.array(extrinsics['t'], dtype=np.float32)

        self.buffer = []
        self.joint_map_msg3d_lhpes3d = [2, -1, 0, 1, 3, 4, 5, 5, 9, 10, 11, 11, 6, 7, 8, 8, 12, 13, 14, 14, -2, 5, 5, 11, 11]

    async def humanActionRecognition(self):
        self.mean_time = 0

        #UI
        delay = 1 #wait for key (1ms)
        esc_code = 27
        p_code = 112
        space_code = 32
         
        #msg3d local data
        msg3d_calculate_frame = 30 #with 60fps = 6 actions/s
        msg3d_count = 0
        msg3d_task = None

        #lhpes3d local data
        lhpes3d_calculate_frame = 1 #with 60fps = 20 sceletons/s
        lhpes3d_count = 0
        
        #Main running loop
        for frame in self.frame_provider:
            msg3d_count = (msg3d_count + 1) % msg3d_calculate_frame
            lhpes3d_count = (lhpes3d_count + 1) % lhpes3d_calculate_frame
            self.current_time = cv2.getTickCount()

            if frame is None:
                break

            if (lhpes3d_count == 0): # Todo: change to 1
                #Preprocess video_stream
                scaled_img, input_scale = self.preprocessFrame(frame)

                #LHPES3D
                self.runLhpes3d(scaled_img, input_scale)


            #MSG3D
            if (len(self.buffer) >= 20 and msg3d_count == 0 ):
                if (msg3d_task != None):
                    await msg3d_task
                    print("Await msg3d_task")
                print("Start msg3d_task")
                loop = asyncio.get_event_loop()
                msg3d_task = loop.create_task(self.runMsg3d())
                
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
            if delay == 0:  # allow to rotate 3D canvas while on pause
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
        if (len(self.buffer) > 60):
            self.buffer.pop(0)

    async def runMsg3d(self):
        msg3d_input = np.zeros(shape=(1, 3, len(self.buffer), 25, 2), dtype= np.float)
        for chanels in range(len(msg3d_input[0])-1, -1, -1):
            for frame in range(len(msg3d_input[0][chanels])):
                for joint in range(len(msg3d_input[0][chanels][frame])):
                    persons_arr = np.zeros(shape=(2))
                    msg3d_input[0][chanels][frame][joint] = persons_arr
                    if (len(self.buffer[frame]) > 0):
                        for person in range(min(len(self.buffer[frame]), 2)):
                            index_to_read = self.joint_map_msg3d_lhpes3d[joint]
                            
                            if (index_to_read == -1):
                                temp = (self.buffer[frame][person][2][chanels] + 0.5 * ((self.buffer[frame][person][9][chanels] + 0.5 * (self.buffer[frame][person][3][chanels] - self.buffer[frame][person][9][chanels])) - self.buffer[frame][person][2][chanels])) / 1
                                temp = temp / 100.0
                            elif (index_to_read == -2):
                                temp = (self.buffer[frame][person][9][chanels] + 0.5 * (self.buffer[frame][person][3][chanels] - self.buffer[frame][person][9][chanels])) / 1
                                temp = temp / 100.0
                            else:
                                temp = self.buffer[frame][person][index_to_read][chanels] / 1
                                temp = temp / 100.0

                            if chanels == 2:
                                #z = self.convert_z(temp)
                                msg3d_input[0][chanels][frame][joint][person] = temp #z 
                            elif chanels == 1:
                                msg3d_input[0][chanels][frame][joint][person] = -temp #self.convert_y(temp, z)
                            elif chanels == 0:
                                msg3d_input[0][chanels][frame][joint][person] = temp #self.convert_x(temp, z)

        self.writeSkeletonFile(msg3d_input);
        msg3d_input = preprocess.pre_normalization(msg3d_input)
        msg3d_input = torch.from_numpy(msg3d_input)
        msg3d_input = msg3d_input.float().cuda()
        self.action_result = self.msg3d_model(msg3d_input)
        print(self.action_result)

    def displayFrame(self, frame):
        np.set_printoptions(threshold=sys.maxsize)
        #self.plotter.plot(self.canvas_3d, self.poses_3d, self.edges) #canvas_3d = img, poses_3d = vertices, edges = edges
        #cv2.imshow(self.canvas_3d_window_name, self.canvas_3d)

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

    def writeSkeletonFile(self, msg3d_input):
        skeleton_file = ""
        skeleton_file += str(len(self.buffer)) + "\n"
        for temp_frame in range(len(self.buffer)):
            skeleton_file += str(len(self.buffer[temp_frame])) + "\n"
            for person in range(min(len(self.buffer[temp_frame]), 2)):
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

    def convert_z(self, z):
        return z * average_z_ntu / average_z_hlpes3d

    def convert_x(self, x, z_meter):
        return (lhpes3d_x_res / 2.0 - x) * (2 * z_meter * math.tan(math.radians(kinect_depth_h_fov / 2.0)) / lhpes3d_x_res)

    def convert_y(self, y, z_meter):
        return (lhpes3d_y_res / 2.0 - y) * (2 * z_meter * math.tan(math.radians(kinect_depth_v_fov / 2.0)) / lhpes3d_y_res)


async def main():
    parser = ArgumentParser(description='3D Human Action Recognition. '
                                        'Press esc to exit, "p" to (un)pause video or process next image.')
    parser.add_argument('-msg3d', '--msg3dmodel',
                        help='Required. Path to trained MS-G3D Model',
                        type=str, required=True)
    parser.add_argument('-lhpe3d', '--lhpe3dmodel',
                        help='Required. Path to trained Lightweight Human Pose Estimation 3D Model',
                        type=str, required=True)
    parser.add_argument('--video', help='Optional. Path to video file or camera id.', type=str, default=0)
    args = parser.parse_args()

    #processor = RunningProcessor(video=r'D:\Repos\HumanActionRecognition\ballthrow.mp4', lpes3d_model_path=r'..\..\..\human-pose-estimation-3d.pth', msg3d_model_path=r'..\..\..\ntu120-xset-joint.pt')
    #processor = RunningProcessor(video=r'/home/hskass2020p7/dev/walking3.mp4', lpes3d_model_path=r'../../../human-pose-estimation-3d.pth', msg3d_model_path=r'../../../ntu120-xset-joint.pt')
    #processor = RunningProcessor(video=3, lpes3d_model_path=r'../../../human-pose-estimation-3d.pth', msg3d_model_path=r'../../../ntu120-xset-joint.pt')
    processor = RunningProcessor(args.video, lpes3d_model_path=args.lhpe3dmodel, msg3d_model_path=args.msg3dmodel)
    await processor.humanActionRecognition()

if __name__ == '__main__':
   # asyncio.run(main())
   loop = asyncio.get_event_loop()
   loop.run_until_complete(main())
