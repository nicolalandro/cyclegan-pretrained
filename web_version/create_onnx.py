import torch
import onnx

# fix for python 3.6
try:
    import cyclegan
except:
    import sys

    sys.path.insert(0, './')

from cyclegan import Generator

model_path = '/media/mint/Barracuda/Models/cyclegan/cezanne/downloaded/netG_B2A.pth'
netG_B2A = Generator(3, 3)
netG_B2A.load_state_dict(torch.load(model_path, map_location=(torch.device('cpu'))))
print('loaded...')

dummy_input = torch.randn(1, 3, 224, 224, device='cpu')
input_names = ["input"]
output_names = ["output"]

torch.onnx.export(netG_B2A,
                  dummy_input,
                  "static/cezanne.onnx",
                  verbose=True,
                  input_names=input_names,
                  output_names=output_names,
                  opset_version=10)

# Test
print('Test')
# Load the ONNX model
onnx_model = onnx.load("static/cezanne.onnx")
# Check that the IR is well formed
onnx.checker.check_model(onnx_model)
# Print a human readable representation of the graph
print(onnx.helper.printable_graph(onnx_model.graph))
