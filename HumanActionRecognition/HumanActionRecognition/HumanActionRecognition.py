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


msg3d = RunningProcessor(r'D:\Repos\ntu120-xset-joint.pt')

input =  torch.from_numpy(np.load(r'./MSG3D/data/Out/xset/val_data_joint.npy'))
if torch.cuda.is_available():
    input = input.cuda()

out = msg3d(input)
print(out)



if os.path.isfile(r'D:\Repos\ntu120-xset-joint.pt'):
    model = torch.load (r'D:\Repos\ntu120-xset-joint.pt')


model.eval()
torch.set_grad_enabled(False)
img = Image.open('random.jpg')
out = model(img)

print (out)
#input = get_input()
#input = preprocess_input(input)
#model.eval()
#out = model(input)
