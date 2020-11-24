import torch
from utils import google_utils

weights = "yolov5.pt"
google_utils.attempt_download(weights)
model = torch.load(weights, map_location=torch.device('cpu'))['model'].float()

model.eval()
# model.model[-1].export = True
input_data = torch.randn(1, 3, 320, 416)#, device=device

y = model(input_data)
print("model = ", model)
torch.onnx.export(model, input_data, "yolov5s.onnx", verbose=False, opset_version=11, input_names=['images'], output_names=['classes', 'boxes'] if y is None else ['output'])
print("Done!")