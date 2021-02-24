#!/usr/bin/python3

import argparse
import urllib

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torchvision.utils import save_image

# fix for python 3.6
try:
    import cyclegan
except:
    import sys

    sys.path.insert(0, './')

from cyclegan import Generator

parser = argparse.ArgumentParser()

parser.add_argument('--image-path', type=str, default='image.jpg', help='image')
parser.add_argument('--generator_A2B', type=str, default='output/netG_A2B.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='output/netG_B2A.pth', help='B2A generator checkpoint file')
opt = parser.parse_args()
print(opt)

# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)

cuda_available = torch.cuda.is_available()
if cuda_available:
    netG_A2B.cuda()
    netG_B2A.cuda()

# Load state dicts
netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
netG_B2A.load_state_dict(torch.load(opt.generator_B2A))

# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

# Dataset loader
transform_test = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

url = 'https://raw.githubusercontent.com/nicolalandro/ntsnet-cub200/master/images/nts-net.png'
img = Image.open(urllib.request.urlopen(url))
scaled_img = transform_test(img)

torch_image_A = scaled_img.unsqueeze(0)

torch_image_B = scaled_img.unsqueeze(0)

with torch.no_grad():
    if cuda_available:
        torch_image_A = torch_image_A.cuda()
        torch_image_B = torch_image_B.cuda()
    # Set model input
    real_A = Variable(input_A.copy_(torch_image_A))
    real_B = Variable(input_B.copy_(torch_image_B))

    # Generate output
    fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)
    fake_A = 0.5 * (netG_B2A(real_B).data + 1.0)

    # Save image files
    save_image(fake_A, 'output/A.png')
    save_image(fake_B, 'output/B.png')
