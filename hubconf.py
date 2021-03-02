dependencies = ['torch']

import torch
from cyclegan import Generator

model_urls = {
    'img2cezanne': 'https://github.com/nicolalandro/cyclegan_pretrained/releases/download/0.1/netG_B2A_cezanne.pth',
    'img2vangogh': 'https://github.com/nicolalandro/cyclegan_pretrained/releases/download/0.1/netG_B2A_vangogh.pth',
    'img2monet': 'https://github.com/nicolalandro/cyclegan_pretrained/releases/download/0.1/netG_B2A_monet.pth',
}


def cyclegan(pretrained=None, device='cpu'):
    """ # This docstring shows up in hub.help()
    CycleGan model
    pretrained (string): kwargs, load pretrained weights into the model (img2vangogh)
    **kwargs
        device (str): 'cuda' or 'cpu'
    """
    net = Generator(3, 3)
    if device == 'cuda':
        net.cuda()
    net.trained_models_list = [k for k in model_urls.keys()]

    if pretrained is not None:
        net.load_state_dict(
            torch.hub.load_state_dict_from_url(
                model_urls[pretrained],
                progress=True,
                map_location=torch.device(device)
            )
        )

    return net
