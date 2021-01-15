import onnxruntime
from models.utils import *
import os

colors = [[0, 0, 255]]
def draw(img, boxinfo, dst, id):
    for *xyxy, conf, cls in boxinfo:
        label = '%s %.2f' % ('pest', conf)
        print('xyxy: ', xyxy)
        plot_one_box(xyxy, img, label=label, color=colors[int(cls)], line_thickness=3)
    # cv2.imshow("dst", img)
    # cv2.waitKey(3000)
    cv2.imencode('.jpg', img)[1].tofile(os.path.join(dst, id+".jpg"))
    return 0

if __name__ == '__main__':

    session = onnxruntime.InferenceSession("yolov5s.onnx")
    input = "/home/lzm/Downloads/img"
    output = "/home/lzm/Downloads/img_dst"
    if not os.path.exists(output):
        os.makedirs(output)

    import glob
    import time
    for id, image_name in enumerate(glob.glob(os.path.join(input, "*"))):
        t1 = time.time()
        image = cv2.imread(image_name)
        img = cv2.resize(image, (416, 320))
        img = img.transpose(2, 0, 1)
        dataset = (image_name, img, image)
        # dataset1 = Resize(image_name, 416)

        img = dataset[1].astype('float32')
        img_size = [dataset[1].shape[1], dataset[1].shape[2]]
        img /= 255.0
        img = img.reshape(1, 3, img_size[0], img_size[1])

        inputs = {session.get_inputs()[0].name: img}

        pred = torch.tensor(session.run(None, inputs)[0])

        stride = [8, 16, 32]
        anchor_list = [[28,33, 40,56, 85,61], [61,120, 114,126, 102,219], [201,149, 183,289, 297,297]]
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
            z.append(src.view(1, -1, 6))

        results = torch.cat(z, 1)
        results = nms(results, 0.3, 0.45)
        t2 = time.time()

        print("time = ", t2-t1)
        for det in results:  # detections per image
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], dataset[2].shape).round()

        if det is not None and len(det):
            draw(image, det, output, str(id))

        # print(id, image_name.rsplit("/")[-1])
    print('Down!')