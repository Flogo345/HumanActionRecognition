import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
from os import listdir
from MSG3D.runningProcessor import RunningProcessor
import numpy as np
import cv2

def humanActionRecognition():
    #Initialization
    msg3d = RunningProcessor(r'D:\Repos\HumanActionRecognition\ntu120-xset-joint.pt')


    #Pipeline
    input =  torch.from_numpy(np.load(r'./MSG3D/data/Out/xset/val_data_joint.npy'))
    if torch.cuda.is_available():
        input = input.cuda()

    out = msg3d(input)
    print(out)



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