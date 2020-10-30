# coding: utf8
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import os.path as osp
import cv2
import numpy as np

from utils.humanseg_postprocess import postprocess, threshold_mask
import models
import transforms


def parse_args():
    parser = argparse.ArgumentParser(description='HumanSeg inference for video')
    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        help='Model path for inference',
        type=str)
    parser.add_argument(
        '--video_path',
        dest='video_path',
        help=
        'Video path for inference, camera will be used if the path not existing',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the inference results',
        type=str,
        default='./output')
    parser.add_argument(
        "--image_shape",
        dest="image_shape",
        help="The image shape for net inputs.",
        nargs=2,
        default=[192, 192],
        type=int)
    parser.add_argument(
        "--distance",
        dest="distance",
        default=500,
        type=int)

    return parser.parse_args()


def predict(img, model, test_transforms):
    model.arrange_transform(transforms=test_transforms, mode='test')
    img, im_info = test_transforms(img)
    img = np.expand_dims(img, axis=0)
    result = model.exe.run(
        model.test_prog,
        feed={'image': img},
        fetch_list=list(model.test_outputs.values()))
    score_map = result[1]
    score_map = np.squeeze(score_map, axis=0)
    score_map = np.transpose(score_map, (1, 2, 0))
    return score_map, im_info


def recover(img, im_info):
    keys = list(im_info.keys())
    for k in keys[::-1]:
        if k == 'shape_before_resize':
            h, w = im_info[k][0], im_info[k][1]
            img = cv2.resize(img, (w, h), cv2.INTER_LINEAR)
        elif k == 'shape_before_padding':
            h, w = im_info[k][0], im_info[k][1]
            img = img[0:h, 0:w]
    return img


def video_infer(args):
    resize_h = args.image_shape[1]
    resize_w = args.image_shape[0]

    test_transforms = transforms.Compose(
        [transforms.Resize((resize_w, resize_h)),
         transforms.Normalize()])
    model = models.load_model(args.model_dir)
    if not args.video_path:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise IOError("Error opening video stream or file, "
                      "--video_path whether existing: {}"
                      " or camera whether working".format(args.video_path))
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    disflow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    prev_gray = np.zeros((resize_h, resize_w), np.uint8)
    prev_cfd = np.zeros((resize_h, resize_w), np.float32)
    is_init = True

    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.video_path:
        index = 100
        findex = 0
        print('Please wait. It is computing......')
        # 用于保存预测结果视频
        if not osp.exists(args.save_dir):
            os.makedirs(args.save_dir)
        out = cv2.VideoWriter(
            osp.join(args.save_dir, 'result3.avi'),
            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width, height))
        # 开始获取视频帧
        while cap.isOpened():
            print(findex)
            ret, frame = cap.read()
            if ret:
                score_map, im_info = predict(frame, model, test_transforms)

                cv2.imshow("sss",score_map[:,:,0])
                cv2.waitKey(0)
                cv2.imshow("sss",score_map[:,:,1])
                cv2.waitKey(0)
                tt = score_map[:,:,1]

                x = np.logical_or(tt, np.zeros_like(tt)).astype(np.float32)
                cv2.imshow("sss",x)
                cv2.waitKey(0)
                cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cur_gray = cv2.resize(cur_gray, (resize_w, resize_h))
                score_map = 255 * score_map[:, :, 1]
                optflow_map = postprocess(cur_gray, score_map, prev_gray, prev_cfd, \
                        disflow, is_init)
                prev_gray = cur_gray.copy()
                prev_cfd = optflow_map.copy()
                is_init = False
                optflow_map = cv2.GaussianBlur(optflow_map, (3, 3), 0)
                optflow_map = threshold_mask(
                    optflow_map, thresh_bg=0.2, thresh_fg=0.8)
                img_matting = np.repeat(
                    optflow_map[:, :, np.newaxis], 3, axis=2)
                img_matting = recover(img_matting, im_info)
                h, w, _ = frame.shape
                newMask = np.zeros([h, w * 3, 3])
                newRes = np.zeros([h, w * 3, 3])

                newMask[:, w - index : 2 * w - index, :] = np.logical_or(newMask[:, w - index : 2 * w - index, :],img_matting).astype(np.float32)
                newMask[:, w + index : 2 * w + index, :] = np.logical_or(newMask[:, w + index : 2 * w + index, :],img_matting).astype(np.float32)
                newMask[:, w: 2 * w, :] = np.logical_or(newMask[:, w: 2 * w, :],img_matting).astype(np.float32)

                newRes[:,w - index : 2 * w - index, :] = newRes[:,w - index : 2 * w - index, :] * (1 - img_matting) + img_matting * frame
                newRes[:,w + index : 2 * w + index, :] = newRes[:,w + index : 2 * w + index, :] * (1 - img_matting) + img_matting * frame
                newRes[:,w : 2 * w,:] = newRes[:,w : 2 * w,:] * (1 - img_matting) + img_matting * frame

                res = (newRes[:, w : w * 2, :] + (1 - newMask[:, w : w * 2, :]) * frame).astype(np.uint8)

                index += 5
                findex += 1
                if index > args.distance:
                    index = args.distance
                #cv2.imwrite("%04d"% findex +'.png' , res)
                out.write(res)
            else:
                break
        cap.release()
        out.release()

    else:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                score_map, im_info = predict(frame, model, test_transforms)
                cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cur_gray = cv2.resize(cur_gray, (resize_w, resize_h))
                score_map = 255 * score_map[:, :, 1]
                optflow_map = postprocess(cur_gray, score_map, prev_gray, prev_cfd, \
                                          disflow, is_init)
                prev_gray = cur_gray.copy()
                prev_cfd = optflow_map.copy()
                is_init = False
                optflow_map = cv2.GaussianBlur(optflow_map, (3, 3), 0)
                optflow_map = threshold_mask(
                    optflow_map, thresh_bg=0.2, thresh_fg=0.8)
                img_matting = np.repeat(
                    optflow_map[:, :, np.newaxis], 3, axis=2)
                img_matting = recover(img_matting, im_info)
                bg_im = np.ones_like(img_matting) * 255
                comb = (img_matting * frame + (1 - img_matting) * bg_im).astype(
                    np.uint8)
                cv2.imshow('HumanSegmentation', comb)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()


if __name__ == "__main__":
    args = parse_args()
    video_infer(args)
