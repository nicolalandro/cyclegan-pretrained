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

img = Image.open('./images/scala_madonnina_del_mare.jpeg')
print(img.size)
scale_factor = 0.8
shape = [int(x * scale_factor) for x in img.size]
print('original_shape:', img.size, 'scaled shape:', shape, f'(Scaled Factor: {scale_factor})')
shape = [shape[1], shape[0]]
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
input_B = Tensor(1, 3, shape[0], shape[1])

# Dataset loader
transform_test = transforms.Compose([
    transforms.Resize(size=(shape[0], shape[1])),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# url = 'https://raw.githubusercontent.com/nicolalandro/ntsnet-cub200/master/images/nts-net.png'
# img = Image.open(urllib.request.urlopen(url))
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
    save_image(fake_A, 'images/A2.png')
