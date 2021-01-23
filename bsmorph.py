import torch
import torch.nn as nn
import torch.optim as optim
import sys
import argparse
import torchvision.models as models
from torchvision import transforms as trn
from PIL import Image
import numpy as np
from big_sleep.biggan import BigGAN
from torchvision.utils import save_image
#from pytorch_pretrained_biggan import (BigGAN, truncated_noise_sample, save_as_images)

# use together with https://github.com/huggingface/pytorch-pretrained-BigGAN


parser = argparse.ArgumentParser()
parser.add_argument('--imageSize', type=int, default=512, help='the height / width of the input image to network')
parser.add_argument('--trunc', type=float, default=0.7, help='truncation, between 0.4 and 1')
parser.add_argument('--lat1', required=True, help='path to startpoint latents')
parser.add_argument('--lat2', required=True, help='path to endpoint latents')
parser.add_argument('--savePath', default="out", help='path to store output')
parser.add_argument('--name', default="farme", help='filename to store output')
parser.add_argument('--steps', type=int, default=200, help='number of intermediate steps')
opt = parser.parse_args()

assert(opt.imageSize in [256,512])
imgSize = opt.imageSize

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ladataan latenssitiedot
lat1 = torch.load(opt.lat1)
lat2 = torch.load(opt.lat2)

best1 = lat1.best
best2 = lat2.best

noise1 = lat1.normu.to(device) 
noise2 = lat2.normu.to(device) 
class1 = lat1.cls.to(device) 
class2 = lat2.cls.to(device)


# ladataan biggan
# load biggan
model = BigGAN.from_pretrained(f'biggan-deep-{imgSize}')
model.eval()

truncation = opt.trunc
model.to(device)

n_delta = (noise2 - noise1) / opt.steps
c_delta = (class2 - class1) / opt.steps

noise_vector = noise1
class_vector = class1

with torch.no_grad():
  for i in  range(0, opt.steps):
    # Generate an image
    output = model(noise_vector, torch.sigmoid(class_vector), truncation)

    # save it
    output = output.to('cpu')
    output = (output + 1)/2
    save_image(output, opt.savePath+"/morf"+str(i)+".png")

    noise_vector += n_delta
    class_vector += c_delta

    
