import os
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')
from models.utils import *

IMAGE_SIZE = (416, 416)
CONF_TH = 0.3
NMS_TH = 0.45
CLASSES = 80
weights = "yolov5s.pt"

if 1:
    device = torch.device("cuda")#使用GPU
else:
    device = torch.device("cpu")#使用CPU

class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        y = torch.stack(y).mean(0)  # mean ensemble
        return y, None  # inference, train output


model = Ensemble()
model.append(torch.load(weights, map_location=device)['model'].float().fuse().eval())  # load FP32 model
model = model[-1]
half = device.type != 'cpu'
if half:
    model.half()

def detect(img):
    image = cv2.resize(img, IMAGE_SIZE) #(宽，高)
    image = image.transpose(2, 0, 1)
    dataset = (image, img)

    img = torch.from_numpy(dataset[0]).to(device)#现将numpy转成tensor
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0) #维度扩充

    pred = model(img, augment=False)[0]
    pred = nms(pred, CONF_TH, NMS_TH, agnostic=False)
    for det in pred:  # detections per image
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], dataset[1].shape).round()  # 输入resize的图像尺寸， box坐标，原始图像尺寸
    if det == None:
        return np.array([])
    return det

def draw(path, boxinfo):
    img = cv2.imread(path)
    for *xyxy, conf, cls in boxinfo:
        label = '{}|{}'.format(int(cls), '%.2f' % conf)
        plot_one_box(xyxy, img, label=label, color=[0, 0, 255])
    cv2.imencode('.jpg', img)[1].tofile('dst1.jpg')
    return 0


if __name__ == '__main__':
    import time
    src = 'bus.jpg'

    t1 = time.time()
    img = cv2.imdecode(np.fromfile(src, dtype=np.uint8), -1)
    results = detect(img)
    t2 = time.time()

    print(results)
    print(device, ": time = ", t2 - t1)

    if results is not None and len(results):
        draw(src, results)
    print('Down!')


