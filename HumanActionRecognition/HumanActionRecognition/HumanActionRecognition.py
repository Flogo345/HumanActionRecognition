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

if os.path.isfile('./Openpose/body_pose_model.pth'):
    model = torch.load ('./Openpose/body_pose_model.pth')
    util


model.eval()
torch.set_grad_enabled(False)
img = Image.open('random.jpg')
out = model(img)

print (out)
#input = get_input()
#input = preprocess_input(input)
#model.eval()
#out = model(input)
