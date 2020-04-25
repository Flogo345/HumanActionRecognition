import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import caffemodel2pytorch
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
from os import listdir

#if os.path.isfile('openpose.pt'):
#    model = torch.load('openpose.pt')

model = caffemodel2pytorch.Net(
	prototxt = 'VGG_ILSVRC_16_layers_deploy.prototxt',
	weights = 'VGG_ILSVRC_16_layers.caffemodel')

model.eval()
torch.set_grad_enabled(False)
img = Image.open('random.jpg')
out = model(img)

print (out)
#input = get_input()
#input = preprocess_input(input)
#model.eval()
#out = model(input)
