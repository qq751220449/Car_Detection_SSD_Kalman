import cv2
import numpy as np


def draw_box_label(img, box_list, label_str=None, box_color=(255, 0, 0)):
    img_show = np.copy(img)
    for box in box_list:
        cv2.rectangle(img_show, (box[1], box[0]), (box[3], box[2]), box_color, 2)
    if label_str is not None:
        box = box_list[0]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_show, str(label_str), ((box[3] - box[1]) // 2 + box[1], (box[2] - box[0]) // 2 + box[0]), font, 1, (200, 100, 255), 2, cv2.LINE_AA)
    # cv2.imshow("detector image", img_show)
    # cv2.waitKey(0)
    return img_show


def box_iou(a, b):
    """
    输入a格式[up, lefy, down, right]
    :param a: box1
    :param b: box2
    :return: iou
    """
    h_intsec = np.maximum(0, (np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])))
    w_intsec = np.maximum(0, (np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])))
    s_intsec = w_intsec * h_intsec
    s_a = (a[2] - a[0]) * (a[3] - a[1])
    s_b = (b[2] - b[0]) * (b[3] - b[1])

    return float(s_intsec) / (s_a + s_b - s_intsec)