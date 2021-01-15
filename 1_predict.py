import os
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')
from models.utils import *

IMAGE_SIZE = 416
CONF_TH = 0.7
NMS_TH = 0.45
CLASSES = 1

def select_device(device='', apex=False, batch_size=None):
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity
    cuda = False if cpu_request else torch.cuda.is_available()
    return torch.device('cuda:0' if cuda else 'cpu')

if torch.cuda.is_available():
    device = select_device('0')
else:
    device = 'cpu'

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

weights = "yolov5.pt"
model = Ensemble()
model.append(torch.load(weights, map_location=device)['model'].float().fuse().eval())  # load FP32 model
model = model[-1]

half = device.type != 'cpu'
if half:
    model.half()

def detect(img):
    image = cv2.resize(img, (416, 320))
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

names = model.module.names if hasattr(model, 'module') else model.names
colors = [[0, 0, 255]]
def draw(path, boxinfo):
    img = cv2.imread(path)
    for *xyxy, conf, cls in boxinfo:
        label = '%s %.2f' % (names[int(cls)], conf)
        print('xyxy: ', xyxy)
        plot_one_box(xyxy, img, label=label, color=colors[int(cls)], line_thickness=3)
    # cv2.imshow("dst", img)
    # cv2.waitKey(1000)
    cv2.imencode('.jpg', img)[1].tofile("dst.jpg")
    return 0


if __name__ == '__main__':
    import glob
    src = '/home/lzm/Downloads/img'
    for i in glob.glob(os.path.join(src, "*")):
        img = cv2.imdecode(np.fromfile(i, dtype=np.uint8),-1)
        results = detect(img)
        if results is not None and len(results):
            draw(i, results)
    print('Down!')


