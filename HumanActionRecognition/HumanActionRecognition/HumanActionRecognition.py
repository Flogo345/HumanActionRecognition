import os
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




def msg3dTest():
    #Initialization
    msg3d_model_path = r'D:\Repos\HumanActionRecognition\ntu120-xset-joint.pt'
    msg3d_model = MSG3DRunningProcessor(msg3d_model_path)

    #Pipeline
    input =  torch.from_numpy(np.load(r'./MSG3D/data/Out/xset/val_data_joint.npy'))
    if torch.cuda.is_available():
        input = input.cuda()

    out = msg3d_model(input)
    print(out)


def humanActionRecognition():
    #Initialization
    video = r'D:\Repos\HumanActionRecognition\ballthrow.mp4' #video = 0; #
    height_size = 256
    fx = -1
    lpes3d_model_path = r'..\..\..\human-pose-estimation-3d.pth'
    msg3d_model_path = r'..\..\..\ntu120-xset-joint.pt'

    lpes3d_model = LPES3DRunningProcessor(lpes3d_model_path)
    msg3d_model = MSG3DRunningProcessor(msg3d_model_path)


    canvas_3d = np.zeros((720, 1280, 3), dtype=np.uint8)
    plotter = Plotter3d(canvas_3d.shape[:2])
    canvas_3d_window_name = 'Canvas 3D'
    cv2.namedWindow(canvas_3d_window_name)
    cv2.setMouseCallback(canvas_3d_window_name, Plotter3d.mouse_callback)

    
    file_path = os.path.join('lhpes3d\data', 'extrinsics.json')
    with open(file_path, 'r') as f:
        extrinsics = json.load(f)
    R = np.array(extrinsics['R'], dtype=np.float32)
    t = np.array(extrinsics['t'], dtype=np.float32)

    #Frameprovider (for stream)
    #frame_provider = VideoReader(video)
    cam = cv2.VideoCapture(video)
    #cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    #cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    is_video = True
    base_height = height_size

    delay = 1
    esc_code = 27
    p_code = 112
    space_code = 32
    mean_time = 0

    buffer_init_time = 2
    buffer_time = cv2.getTickCount() 
    init_buffer_over = False
    buffer = []
    joint_map_msg3d_lhpes3d = [2, -1, 0, 1, 3, 4, 5, 5, 9, 10, 11, 11, 6, 7, 8, 8, 12, 13, 14, 14, -2, 5, 5, 11, 11] 
    msg3d_calculate_frame = 10

   
    msg3d_count = 0
    #Main running loop
    #for frame in frame_provider:
    while True:
        msg3d_count = (msg3d_count + 1) % msg3d_calculate_frame
        ret_val, frame = cam.read()
        current_time = cv2.getTickCount()
        if frame is None:
            break
        input_scale = base_height / frame.shape[0]
        scaled_img = cv2.resize(frame, dsize=None, fx=input_scale, fy=input_scale)
        scaled_img = scaled_img[:, 0:scaled_img.shape[1] - (scaled_img.shape[1] % 8)]  # better to pad, but cut out for demo
        if fx < 0:  # Focal length is unknown
            fx = np.float32(0.8 * frame.shape[1])

        #Determin Value with model 
        inference_result = lpes3d_model(scaled_img)
        poses_3d, poses_2d = parse_poses(inference_result, input_scale, 8, fx, is_video)
        edges = []
        if len(poses_3d):
            poses_3d = rotate_poses(poses_3d, R, t)
            poses_3d_copy = poses_3d.copy()
            x = poses_3d_copy[:, 0::4]
            y = poses_3d_copy[:, 1::4]
            z = poses_3d_copy[:, 2::4]
            poses_3d[:, 0::4], poses_3d[:, 1::4], poses_3d[:, 2::4] = -z, x, -y

            poses_3d = poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]
            edges = (Plotter3d.SKELETON_EDGES + 19 * np.arange(poses_3d.shape[0]).reshape((-1, 1, 1))).reshape((-1, 2))
        plotter.plot(canvas_3d, poses_3d, edges)
        cv2.imshow(canvas_3d_window_name, canvas_3d)



        #Place for MSG3D
        buffer.append(poses_3d)
        if (len(buffer) > 45 and msg3d_count == 0 ):
            buffer.pop(0)
            init_buffer_over = True
            msg3d_input = np.zeros(shape=(1, 3, len(buffer), 25, 2), dtype= np.float)
            for chanels in range(len(msg3d_input[0])):
                for frame in range(len(msg3d_input[0][chanels])):
                    for joint in range(len(msg3d_input[0][chanels][frame])):
                        if (len(buffer[frame]) > 0):
                            persons_arr = np.zeros(shape=(2))
                            msg3d_input[0][chanels][frame][joint] = persons_arr
                            for person in range(min(len(buffer[frame]), 2)):
                                index_to_read = joint_map_msg3d_lhpes3d[joint]
                                if (index_to_read == -1):
                                    msg3d_input[0][chanels][frame][joint][person] = (buffer[frame][person][2][chanels] + 0.5 * ((buffer[frame][person][9][chanels] + 0.5 * (buffer[frame][person][3][chanels] - buffer[frame][person][9][chanels])) - buffer[frame][person][2][chanels])) / 1
                                elif (index_to_read == -2):
                                    msg3d_input[0][chanels][frame][joint][person] = (buffer[frame][person][9][chanels] + 0.5 * (buffer[frame][person][3][chanels] - buffer[frame][person][9][chanels])) / 1
                                else:
                                    msg3d_input[0][chanels][frame][joint][person] = buffer[frame][person][index_to_read][chanels] / 1
            skeleton_file = ""
            skeleton_file += str(len(buffer)) + "\n"
            for frame in buffer:
                skeleton_file += str(len(frame)) + "\n"
                for person in frame:
                    skeleton_file += "0 0 0 0 0 0 0 0 0 0" + "\n"
                    skeleton_file += str(len(person)) + "\n"
                    for joint in person:
                        skeleton_file += str(joint[0]) + " " + str(joint[1]) + " " + str(joint[2]) + " 0 0 0 0 0 0 0 0 0\n"
            write_file(skeleton.skeleton, skeleton_file)

            np.set_printoptions(threshold=sys.maxsize)
            #msg3d_input = preprocess.pre_normalization(msg3d_input)
            msg3d_input = torch.from_numpy(msg3d_input)
            msg3d_input = msg3d_input.float().cuda()
            out = msg3d_model(msg3d_input)
            print(out)
            

        draw_poses(frame, poses_2d)
        current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
        if mean_time == 0:
            mean_time = current_time
        else:
            mean_time = mean_time * 0.95 + current_time * 0.05
        cv2.putText(frame, 'FPS: {}'.format(int(1 / mean_time * 10) / 10),
                    (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        cv2.imshow('ICV 3D Human Pose Estimation', frame)
            

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
                plotter.plot(canvas_3d, poses_3d, edges)
                cv2.imshow(canvas_3d_window_name, canvas_3d)
                key = cv2.waitKey(33)
            if key == esc_code:
                break
            else:
                delay = 1

def write_file(name, text):
    f = open(name, "a")
    f.write(text)
    f.close()



def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 1)
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()


def main():
    humanActionRecognition()

if __name__ == '__main__':
    main()