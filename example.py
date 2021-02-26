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
parser.add_argument('--generator_B2A', type=str,
                    default='/media/mint/Barracuda/Models/cyclegan/monet/downloaded/netG_B2A.pth',
                    help='B2A generator checkpoint file')
opt = parser.parse_args()
print(opt)

# Networks
netG_B2A = Generator(3, 3)

cuda_available = torch.cuda.is_available()
if cuda_available:
    netG_B2A.cuda()

# Load state dicts
netG_B2A.load_state_dict(torch.load(opt.generator_B2A))

# Set model's test mode
netG_B2A.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if cuda_available else torch.Tensor
input_B = Tensor(1, 3, 256, 256)

# Dataset loader
transform_test = transforms.Compose([
    transforms.Resize(size=(256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# url = 'https://raw.githubusercontent.com/nicolalandro/ntsnet-cub200/master/images/nts-net.png'
# img = Image.open(urllib.request.urlopen(url))
img = Image.open('./images/photo.jpg')
scaled_img = transform_test(img)

torch_image_B = scaled_img.unsqueeze(0)

with torch.no_grad():
    if cuda_available:
        torch_image_B = torch_image_B.cuda()
    # Set model input
    real_B = Variable(input_B.copy_(torch_image_B))

    # Generate output
    fake_A = 0.5 * (netG_B2A(real_B).data + 1.0)

    # Save image files
    save_image(fake_A, 'images/A.png')
