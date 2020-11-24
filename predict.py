from models.experimental import *
from utils.utils import *

IMAGE_SIZE = 416
CONF_TH = 0.55
NMS_TH = 0.45
CLASSES = 1

if torch.cuda.is_available():
    device = torch_utils.select_device('0')
else:
    device = 'cpu'

model = attempt_load("yolov5.pt", map_location=device)
half = device.type != 'cpu'
if half:
    model.half()

names = model.module.names if hasattr(model, 'module') else model.names
colors = [[0, 0, 255]]

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True): #保证图像的长宽均为32的倍数
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape) #(416,416)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])#最小缩放比例
    if not scaleup:  # only scale down, do not scale up (for better test mAP), 都是将原始图像尺寸进行缩小
        r = min(r, 1.0)

    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))#使用同一缩放比进行缩放

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR) #将图像resize到新的尺寸

    # Compute padding，计算扩充尺寸(已经确保长边能够被32整除，即，以下都是在计算在短边上的扩充尺寸)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding，
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding，取余，计算最小的扩充尺寸
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    #计算扩充到两端的尺寸
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 扩充边界,(top,bottom,left,right,表示在img的指定端扩充的像素数)
    #cv2.imwrite("board32.jpg", img)
    return img, ratio, (dw, dh)

def Resize(path, img_size=640):
    img0 = cv2.imread(path)  # BGR
    img = letterbox(img0, new_shape=img_size)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416，颜色空间转化，通道转化
    img = np.ascontiguousarray(img)#连续的内存空间
    return path, img, img0

def nms(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, agnostic=False):
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32
    xc = prediction[..., 4] > conf_thres  # candidates
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    redundant = True  # require redundant detections
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence
        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        box = xywh2xyxy(x[:, :4])

        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        n = x.shape[0]  # number of boxes
        if not n:
            continue
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass
        output[xi] = x[i]

    return output

def detect(img):
    imgsz = check_img_size(IMAGE_SIZE, s=model.stride.max())
    dataset = Resize(img, img_size=imgsz)
    img = torch.from_numpy(dataset[1]).to(device)#现将numpy转成tensor
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0) #维度扩充

    print("image = ", img, img.shape, img.dtype)
    pred = model(img, augment=False)[0]
    print("pred = ", pred)
    pred = nms(pred, CONF_TH, NMS_TH, agnostic=False)
    for det in pred:  # detections per image
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], dataset[2].shape).round()  # 输入resize的图像尺寸， box坐标，原始图像尺寸

    return det

def draw(path, boxinfo):
    img = cv2.imread(path)
    for *xyxy, conf, cls in boxinfo:
        label = '%s %.2f' % (names[int(cls)], conf)
        plot_one_box(xyxy, img, label=label, color=colors[int(cls)], line_thickness=3)
    cv2.imshow("dst", img)
    cv2.waitKey(1000)
    # cv2.imencode('.jpg', img)[1].tofile("/home/lzm/Downloads/img/dst.jpg")
    return 0

if __name__ == '__main__':
    import glob
    src = "/home/lzm/Downloads/img"
    for img in glob.glob(os.path.join(src, "*")):
        results = detect(img)
        if results is not None and len(results):
            draw(img, results)
    print("hello")


