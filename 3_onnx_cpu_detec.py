import onnxruntime
from models.utils import *
import time

IMAGE_SIZE = (416, 416)
CONF_TH = 0.3
NMS_TH = 0.45
CLASSES = 80

model = onnxruntime.InferenceSession("yolov5s.onnx")
anchor_list = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
stride = [8, 16, 32]

def draw(img, boxinfo, dst, id):
    for *xyxy, conf, cls in boxinfo:
        label = '{}|{}'.format(int(cls), '%.2f' % conf)
        plot_one_box(xyxy, img, label=label, color=[0, 0, 255])
    cv2.imencode('.jpg', img)[1].tofile(dst)

def detect(image):

    img = cv2.resize(image, IMAGE_SIZE)
    img = img.transpose(2, 0, 1)
    dataset = (img, image)

    img = dataset[0].astype('float32')
    img_size = [dataset[0].shape[1], dataset[0].shape[2]]
    img /= 255.0
    img = img.reshape(1, 3, img_size[0], img_size[1])

    inputs = {model.get_inputs()[0].name: img}
    pred = torch.tensor(model.run(None, inputs)[0])

    anchor = torch.tensor(anchor_list).float().view(3, -1, 2)
    area = img_size[0]*img_size[1]
    size = [int(area/stride[0]**2), int(area/stride[1]**2), int(area/stride[2]**2)]
    feature = [[int(j/stride[i]) for j in img_size] for i in range(3)]

    y = []
    y.append(pred[:, :size[0]*3, :])
    y.append(pred[:, size[0]*3:size[0]*3+size[1]*3, :])
    y.append(pred[:, size[0]*3+size[1]*3:, :])

    grid = []
    for k, f in enumerate(feature):
        grid.append([[i, j] for j in range(f[0]) for i in range(f[1])])

    z = []
    for i in range(3):
        src = y[i]
        xy = src[..., 0:2] * 2. - 0.5
        wh = (src[..., 2:4] * 2) ** 2
        dst_xy = []
        dst_wh = []
        for j in range(3):
            dst_xy.append((xy[:, j*size[i]:(j+1)*size[i], :] + torch.tensor(grid[i])) * stride[i])
            dst_wh.append(wh[:, j*size[i]:(j+1)*size[i], :] * anchor[i][j])
        src[..., 0:2] = torch.from_numpy(np.concatenate((dst_xy[0], dst_xy[1], dst_xy[2]), axis=1))
        src[..., 2:4] = torch.from_numpy(np.concatenate((dst_wh[0], dst_wh[1], dst_wh[2]), axis=1))
        z.append(src.view(1, -1, CLASSES+5)) #85

    pred = torch.cat(z, 1)
    pred = nms(pred, CONF_TH, NMS_TH)

    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], dataset[1].shape).round()
    if det == None:
        return np.array([])
    return det



if __name__ == '__main__':
    import time
    src = 'bus.jpg'

    t1 = time.time()
    img = cv2.imdecode(np.fromfile(src, dtype=np.uint8), -1)
    results = detect(img)
    t2 = time.time()

    print(results)
    print("onnxruntime time = ", t2 - t1)

    if results is not None and len(results):
        draw(img, results, 'dst3.jpg', str(id))
    print('Down!')