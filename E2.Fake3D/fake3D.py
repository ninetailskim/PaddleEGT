import numpy as np
import os
import paddlehub as hub
import cv2
from moviepy.editor import VideoFileClip
from tqdm import tqdm 
import copy

os.environ["CUDA_VISIBLE_DEVICES"]="0"

module = hub.Module(name="deeplabv3p_xception65_humanseg")

def do_seg(frame):
    result = module.segmentation(images=[frame])
    return result[0]['data']

originname = "test.mp4"
resultname = "result.avi"
onlyV = True
addAudio = True

cap = cv2.VideoCapture(originname)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(resultname, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width,height))

for index in tqdm(range(framecount)):
    ret, frame = cap.read()
    if not ret:
        break
    seg_mask = np.around(do_seg(frame) / 255)
    seg_mask3 = np.repeat(seg_mask[:,:,np.newaxis], 3, axis=2)
    origin_size = frame.shape
    h1, h2 = int(origin_size[0]/4), int(origin_size[0]/4*3)
    w1, w2 = int(origin_size[1]/4), int(origin_size[1]/4*3)
    background = copy.deepcopy(frame)
    background[:,w1-6:w1+6,:] = 255
    background[:,w2-6:w2+6,:] = 255
    if not onlyV:
        background[h1-6:h1+6,:] = 255
        background[h2-6:h2+6,:] = 255
    result = seg_mask3 * frame + (1 - seg_mask3) * background
    out.write(result.astype(np.uint8))

cap.release()
out.release()

if addAudio:
    result_no_Audio = VideoFileClip(resultname)
    result = result_no_Audio.set_audio(VideoFileClip(originname).audio)
    result.write_videofile(resultname)