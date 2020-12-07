import sys
import pygame
import cv2

capture  = cv2.VideoCapture('test_sample.mov')
fps = capture.get(cv2.CAP_PROP_FPS)
size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# 将预测结果写成视频
video_writer = cv2.VideoWriter('result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

def generate_image():
    while True:
        # frame_rgb即视频的一帧数据
        ret, frame_rgb = capture.read() 
        # 按q键即可退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if frame_rgb is None:
            break
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        shape = frame_bgr.shape
        yield cv2.resize(frame_bgr, (round(shape[1] / 3), round(shape[0] / 3)))
    capture.release()
    video_writer.release()
    cv2.destroyAllWindows()

pygame.init()
size = 

for index, img in enumerate(generate_image(), start=1):
    cv2.imshow("1",img)
