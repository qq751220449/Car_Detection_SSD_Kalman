import cv2
import os

video_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../video/project_video.mp4"))     # 视频文件存放地址
print(video_path)

frame_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../test_image/"))
if not os.path.exists(frame_path):
    os.mkdir(frame_path)

start_time = 12000
end_time = 16000
step = 200

video_capture = cv2.VideoCapture(video_path)
count = 0
for i, time in enumerate(range(start_time, end_time, step)):
    video_capture.set(cv2.CAP_PROP_POS_MSEC, time)      # 设定当前视频帧的位置
    success, frame = video_capture.read()               # 读取当前帧
    if success:     # 视频读取成功
        filename = "frame{%.5d}.jpg" % count
        frame_single_path = os.path.abspath(os.path.join(frame_path, filename))
        cv2.imwrite(frame_single_path, frame)
        count += 1

