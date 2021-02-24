dependencies = ['torch']

import torch

model_url = 'xxxx'


def cyclegan(pretrained=None, **kwargs):
    """ # This docstring shows up in hub.help()
    CycleGan model
    pretrained (string): kwargs, load pretrained weights into the model (vangog2photo)
    **kwargs
        device (str): 'cuda' or 'cpu'
    """
    net = cyclegan(**kwargs)
    if pretrained:
        net.load_state_dict(torch.hub.load_state_dict_from_url(model_url, progress=True))
    return net
