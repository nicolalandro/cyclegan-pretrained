import argparse

import torch
import torchvision.transforms as transforms
from PIL import Image
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
                    default='/media/mint/Barracuda/Models/cyclegan/vangog/downloaded/netG_B2A.pth',
                    help='B2A generator checkpoint file')

img = Image.open('./images/scala_madonnina_del_mare.jpeg')
print(img.size)
scale_factor = 0.25
shape = [int(x * scale_factor) for x in img.size]
print('original_shape:', img.size, 'scaled shape:', shape, f'(Scaled Factor: {scale_factor})')
shape = [shape[1], shape[0]]
opt = parser.parse_args()
print(opt)

# Networks
netG_B2A = Generator(3, 3)

cuda_available = torch.cuda.is_available()
# cuda_available = False

device = torch.device('cpu')
if cuda_available:
    netG_B2A.cuda()
    device = torch.device('cuda')

# Load state dicts
netG_B2A.load_state_dict(torch.load(opt.generator_B2A, map_location=device))
# netG_B2A.load_state_dict(
#             torch.hub.load_state_dict_from_url(
#                 'https://github.com/nicolalandro/cyclegan_pretrained/releases/download/0.1/netG_B2A_cezanne.pth',
#                 progress=True,
#                 map_location=torch.device(device)
#             )
#         )
print('loaded...')

# Set model's test mode
netG_B2A.eval()

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

    # Generate output
    fake_A = 0.5 * (netG_B2A(torch_image_B).data + 1.0)

    # Save image files
    save_image(fake_A, 'images/A2.png')
