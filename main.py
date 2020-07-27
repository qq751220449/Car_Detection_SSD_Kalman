import os
import cv2
import numpy as np
from collections import deque
from utils.ssd_detector import CarDetectot_SSD
from utils.tools import draw_box_label, box_iou
from sklearn.utils.linear_assignment_ import linear_assignment      # sklearn 0.19.1
from utils.kalman_tracker import Kalman_Tracker

frame_count = 0         # 帧计数器
max_age = 30            # 若一个跟踪目标长时间未被检测到,则删除这个跟踪目标,该参数设置多久未跟踪到目标时删除
min_hits = 30           # 检测到同一目标多少次后进行跟踪

tracker_list = []        # 跟踪目标列表
# 跟踪目标ID列表
track_id_list = deque(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'])

debug = False


def assign_detections_to_trackers(trackers, detections, iou_thrd=0.3):
    '''
    将已有的跟踪目标与新检测到的目标进行匹配
    '''
    IOU_mat = np.zeros((len(trackers), len(detections)), dtype=np.float32)  # IOU矩阵
    for t, trk in enumerate(trackers):
        for d, det in enumerate(detections):
            IOU_mat[t, d] = box_iou(trk, det)

    # 匹配算法-匈牙利算法

    matched_idx = linear_assignment(-IOU_mat)   # 原本的函数是找最小值,因此这里转换为负数
    # 该函数后续版本删除,替代方案如下:https://www.cnblogs.com/clemente/p/12321745.html

    """
    sklearn API result:
    [[0 1]
    [1 0]
    [2 2]]
    """

    unmatched_trackers, unmatched_detections = [], []
    for t, trk in enumerate(trackers):
        if (t not in matched_idx[:, 0]):
            unmatched_trackers.append(t)

    for d, det in enumerate(detections):
        if (d not in matched_idx[:, 1]):
            unmatched_detections.append(d)

    matches = []

    # 删除IOU小于阈值的匹配点,这些可能是噪声
    for m in matched_idx:
        if (IOU_mat[m[0], m[1]] < iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if (len(matches) == 0):         # 如果找不到匹配的
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def pipeline(image):

    global frame_count
    global tracker_list
    global track_id_list

    frame_count += 1

    if debug:
        print("frame_count:", frame_count)

    img = car_detector.load_image_into_numpy_array(image)
    detect_boxes = car_detector.get_locations(img)    # 检测到的Boxes框
    img_draw = draw_box_label(image, detect_boxes, box_color=(255, 0, 0))
    if debug:
        cv2.imshow("detector image", img_draw)
        cv2.waitKey(0)

    track_boxes = []
    if len(tracker_list) > 0:       # 已经有待跟踪的目标
        for trk in tracker_list:
            track_boxes.append(trk.box)

    # 已经提取到待追踪的目标和检测到新的目标,需进行目标匹配
    matched, unmatched_dets, unmatched_trks = assign_detections_to_trackers(track_boxes, detect_boxes, iou_thrd=0.01)

    if debug:
         print('Detection: ', detect_boxes)
         print('track_boxes: ', track_boxes)
         print('matched:', matched)
         print('unmatched_det:', unmatched_dets)
         print('unmatched_trks:', unmatched_trks)


    # 处理匹配成功的detections
    if matched.size > 0:
        for trk_idx, det_idx in matched:
            z = detect_boxes[det_idx]           # 取出检测Box(也即测量值)
            z = np.expand_dims(z, axis=0).T     # 维度变换,便于操作
            tmp_trk = tracker_list[trk_idx]     # 取出当前跟踪目标
            tmp_trk.predict_update(z)           # 卡尔曼更新
            new_box = tmp_trk.x_state.T[0].tolist()
            new_box = [new_box[0], new_box[2], new_box[4], new_box[6]]
            track_boxes[trk_idx] = new_box
            tmp_trk.box = new_box
            tmp_trk.hits += 1
            tmp_trk.no_losses = 0

    # 处理未匹配的检测到的目标
    if len(unmatched_dets) > 0:
        for idx in unmatched_dets:
            z = detect_boxes[idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk = Kalman_Tracker()              # Create a new tracker
            x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
            tmp_trk.x_state = x
            tmp_trk.predict_only()
            new_box = tmp_trk.x_state.T[0].tolist()
            new_box = [new_box[0], new_box[2], new_box[4], new_box[6]]
            tmp_trk.box = new_box
            tmp_trk.id = track_id_list.popleft()
            tracker_list.append(tmp_trk)
            track_boxes.append(new_box)

    # 处理未匹配的跟踪目标
    if len(unmatched_trks) > 0:
        for trk_idx in unmatched_trks:
            tmp_trk = tracker_list[trk_idx]
            tmp_trk.no_losses += 1
            tmp_trk.predict_only()
            new_box = tmp_trk.x_state.T[0].tolist()
            new_box = [new_box[0], new_box[2], new_box[4], new_box[6]]
            tmp_trk.box = new_box
            track_boxes[trk_idx] = new_box

    # The list of tracks to be annotated
    good_tracker_list = []
    for trk in tracker_list:
        if ((trk.hits >= min_hits) and (trk.no_losses <= max_age)):     # 显示更新后的boxes
            good_tracker_list.append(trk)
            img_draw = draw_box_label(img_draw, [trk.box], trk.id, (0, 0, 255))
    out_detect.write(img_draw)
    if debug:
        cv2.imshow("update frame", img_draw)
        cv2.waitKey(0)

    # Book keeping
    deleted_tracks = filter(lambda x: x.no_losses > max_age, tracker_list)
    for trk in deleted_tracks:
        track_id_list.append(trk.id)

    tracker_list = [x for x in tracker_list if x.no_losses <= max_age]

    if debug:
        print('Ending tracker_list: ', len(tracker_list))
        print('Ending good tracker_list: ', len(good_tracker_list))

    return image


if __name__ == "__main__":
    car_detector = CarDetectot_SSD()    # 生成基于SSD的车辆检测器

    if debug:           # 调试时,将max_age和min_hits参数进行调整
        images_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./test_image/"))
        images_name = os.listdir(images_path)
        for image_single_name in images_name:
            image_single_path = os.path.abspath(os.path.join(images_path, image_single_name))
            img = cv2.imread(image_single_path, cv2.IMREAD_COLOR)
            pipeline(img)
    else:
        video_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./video/project_video.mp4"))  # 视频文件存放地址
        video_capture = cv2.VideoCapture(video_path)
        size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc1 = cv2.VideoWriter_fourcc(*'XVID')
        out_detect = cv2.VideoWriter('output_detect.avi', fourcc1, 20.0, size)
        while True:
            success, frame = video_capture.read()  # 读取当前帧
            if success:  # 视频读取成功
                pipeline(frame)
            else:
                break
        video_capture.release()  # 释放候选框
        out_detect.release()


