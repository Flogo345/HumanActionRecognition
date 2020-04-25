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

if os.path.isfile('openpose.pt'):
    model = torch.load('openpose.pt')

input = get_input()
input = preprocess_input(input)
model.eval()
out = model(input)
