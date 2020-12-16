import numpy as np
import os
import paddlehub as hub
import cv2
from moviepy.editor import VideoFileClip
from tqdm import tqdm 
import copy

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def do_seg(module, frame):
    result = module.segmentation(images=[frame],use_gpu=True)
    return result[0]['data']

module = hub.Module(name="deeplabv3p_xception65_humanseg")

originname = "test.mp4"
resultname = "test_ngresult.avi"
shadowcount = 9

cap = cv2.VideoCapture(originname)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(resultname, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width, height))

tmpres = []

for index in tqdm(range(framecount)):
    ret, frame = cap.read()
    if not ret:
        break
    seg_mask = np.around(do_seg(module,frame) / 255)
    seg_mask3 = np.repeat(seg_mask[:,:,np.newaxis], 3, axis=2)
    background = copy.deepcopy(frame)
    stbackground = copy.deepcopy(frame)
    if len(tmpres) > shadowcount:
        tmpres = tmpres[1:]
    # tmpres.append([copy.deepcopy(seg_mask3), copy.deepcopy(cv2.GaussianBlur(seg_mask3 * background,(9,9),0))])
    tmpres.append([copy.deepcopy(seg_mask3), copy.deepcopy(seg_mask3 * background)])
    thuman = copy.deepcopy(seg_mask3 * background)
    if index > len(tmpres):
        for fi, [t_mask3, t_human] in enumerate(tmpres):
            background = t_human * (fi + 1) / len(tmpres) + t_mask3 * (len(tmpres) - 1 - fi) / len(tmpres) * stbackground + (1 - t_mask3) * background

    result = background.astype(np.uint8)
    out.write(result)
    
cap.release()
out.release()