# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os


import cv2
from keras.preprocessing.image import img_to_array

# gpu显存设置
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)   #设置每个进程使用GPU内存所占的比例

PATH_TO_CKPT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/ssd_mobilenet_v2/frozen_inference_graph.pb"))
if not os.path.exists(PATH_TO_CKPT):
    print("There is no predict files.")
else:
    print(PATH_TO_CKPT)


class CarDetectot_SSD(object):

    def __init__(self):
        self.car_boxes = []         # 检测到的车辆位置box列表
        self.detection_graph = tf.Graph()

        # 加载计算图
        with self.detection_graph.as_default():
            with tf.gfile.GFile(PATH_TO_CKPT, "rb") as f:
                inference_graph_def = tf.GraphDef()
                inference_graph_def.ParseFromString(f.read())
                tf.import_graph_def(inference_graph_def, name='')

        self.sess = tf.Session(graph=self.detection_graph, config=tf.ConfigProto(gpu_options=gpu_options))
        self.image_tensor = self.detection_graph.get_tensor_by_name("image_tensor:0")
        self.boxes = self.detection_graph.get_tensor_by_name("detection_boxes:0")

        self.scores = self.detection_graph.get_tensor_by_name("detection_scores:0")
        self.classes = self.detection_graph.get_tensor_by_name("detection_classes:0")
        self.num_detections = self.detection_graph.get_tensor_by_name("num_detections:0")

    def load_image_into_numpy_array(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = img_to_array(image)
        image_batch = np.expand_dims(image, axis=0).astype(np.uint8)
        return image_batch

    def box_normal_to_pixel(self, box, image_shape):
        height, width = image_shape[0:2]
        print(box)
        print(height, width)
        box_pixel = [int(box[0]*height), int(box[1]*width), int(box[2]*height), int(box[3]*width)]
        return np.array(box_pixel)

    def get_locations(self, image):
        (boxes, scores, classes, num_detections) = self.sess.run(
            [self.boxes, self.scores, self.classes, self.num_detections], feed_dict={self.image_tensor: image})
        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes)
        scores = np.squeeze(scores)
        cls = classes.tolist()

        # 在COCO数据集中,car对应的类别为3
        idx_vec = [i for i, v in enumerate(cls) if ((v == 3) and scores[i] > 0.3)]

        if len(idx_vec) == 0:
            print("No Detection.")
            self.car_boxes = []
        else:
            self.car_boxes = []
            for index in idx_vec:
                box = self.box_normal_to_pixel(boxes[index], image.shape[1:3])
                print(box)
                box_h = box[2] - box[0]
                box_w = box[3] - box[1]
                ratio = box_h / (box_w + 0.01)

                if ((ratio < 0.8) and (box_h>20) and (box_w>20)):
                    self.car_boxes.append(box)
                    print(box, ', confidence: ', scores[index], 'ratio:', ratio)
                else:
                    continue
        return self.car_boxes


if __name__ == "__main__":
    det = CarDetectot_SSD()
    image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../test_image/"))
    images_name = os.listdir(image_path)
    print(images_name)
    for image_name in images_name:
        image_single_path = os.path.abspath(os.path.join(image_path, image_name))
        print(image_single_path)

        img = cv2.imread(image_single_path, cv2.IMREAD_COLOR)
        img_show = np.copy(img)
        cv2.imshow("frame_src", img_show)
        img = det.load_image_into_numpy_array(img)
        box_list = det.get_locations(img)
        print(box_list)
        for box in box_list:
            cv2.rectangle(img_show, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)
        cv2.imshow("frame", img_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

