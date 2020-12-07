import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import paddlehub as hub
from paddlehub.common.logger import logger
import numpy as np
import math

class FaceGameController(object):
    def __init__(self, debug, use_gpu):
        self.debug = debug
        self.module = hub.Module(name="face_landmark_localization")
        self.capture = cv2.VideoCapture(0) 
        self.actiondim = 8
        self.actionset = ["left","right","up","down","mopen","mheri","lefteye","righteye"]
        self.leftHist = None
        self.rightHist = None
        self.initeye(use_gpu = use_gpu)
        
        # self.leftSize = 0
        # self.rightSize = 0

    def initeye(self, use_gpu):
        success = False
        init = False
        while success is False or init is False:
            ret, frame_rgb = self.capture.read()
            frame_rgb = cv2.flip(frame_rgb,1)
            success, face_landmark = self.get_face_landmark(image=frame_rgb, use_gpu=use_gpu)
            tips = ''
            if success:
                tips = "Found face, please keep your eyes open. Open your mouth to confirm initialization."
            else:
                tips = "Not found face, please adjust your pose."
            tmp_img = frame_rgb.copy()
            cv2.putText(tmp_img, tips, (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            cv2.imshow("Init", tmp_img)
            cv2.waitKey(1)
            if success:
                if face_landmark[66][1] - face_landmark[62][1] > 1.2 * (face_landmark[62][1] - face_landmark[51][1]):
                    leftimg, rightimg = self.getEyes(frame_rgb, face_landmark)
                    self.leftHist = self.calHist(leftimg)
                    self.rightHist = self.calHist(rightimg)
                    # self.leftSize = leftimg.shape
                    # self.rightSize = rightimg.shape
                    init = True
            else:
                continue
        
        cv2.destroyAllWindows()
    
    def getEyes(self, img, face_landmark):
        result = np.array(face_landmark, dtype=np.int32)

        lefteye1 = [result[36][0], result[37][1]]
        lefteye2 = [result[39][0], result[41][1]]
        righteye1 = [result[42][0], result[43][1]]
        righteye2 = [result[45][0], result[47][1]]

        left_img = img[lefteye1[1]:lefteye2[1]+1,lefteye1[0]:lefteye2[0],:]
        right_img = img[righteye1[1]:righteye2[1]+1,righteye1[0]:righteye2[0],:]

        return left_img, right_img

    def calHist(self, img):
        hist = []
        for i in range(3):
            hist.append(cv2.calcHist([img], [i], None, [256],[0,256]))
        hist = np.vstack(np.array(hist))
        return hist

    def get_face_landmark(self, image, use_gpu=False):
        try:
            res = self.module.keypoint_detection(images=[image], use_gpu=use_gpu)
            return True, res[0]['data'][0]
        except Exception as e:
            logger.error("Get face landmark localization failed! Exception: %s " % e)
            return False, None

    def judgeaction(self, face_landmark, img):
        action = np.zeros([8], dtype=np.int32)
        left = face_landmark[33][0] * 3 - face_landmark[2][0] - face_landmark[3][0] - face_landmark[4][0]
        right = face_landmark[14][0] + face_landmark[13][0] + face_landmark[12][0] - face_landmark[33][0] * 3
        if left < right * 0.6:
            action[0] = 1
        if right < left * 0.6:
            action[1] = 2

        up = face_landmark[29][1] - face_landmark[27][1]
        baseline = (face_landmark[27][1] * 2 - face_landmark[21][1] - face_landmark[22][1])  / 2
        if up < 0.8 * baseline:
            action[2] = 4

        if face_landmark[4][1] < face_landmark[50][1] or face_landmark[12][1] < face_landmark[52][1]:
            action[3] = 8

        if face_landmark[66][1] - face_landmark[62][1] > 1.8 * (face_landmark[62][1] - face_landmark[51][1]):
            action[4] = 16

        if face_landmark[48][1] - face_landmark[62][1] < 0 or face_landmark[54][1] - face_landmark[62][1] < 0:
            action[5] = 0
        
        jer = self.judgeEye(face_landmark, img)
        action[6] = jer[0]
        action[7] = 0 # jer[1]


        return action
        # return np.sum(action)        

    def judgeEye(self, face_landmark, img):
        eyeres = [0, 0]
        left, right = self.getEyes(img, face_landmark)
        lefthist = self.calHist(left)
        righthist = self.calHist(right)
        if cv2.compareHist(self.leftHist, lefthist, cv2.HISTCMP_CORREL) < 0.2:
            eyeres[0] = 64
        if cv2.compareHist(self.rightHist, righthist, cv2.HISTCMP_CORREL) < 0.2:
            eyeres[1] = 128
        return eyeres

    def control(self, use_gpu=False):
        ret, frame_rgb = self.capture.read()
        frame_rgb = cv2.flip(frame_rgb,1)
        success, face_landmark = self.get_face_landmark(image=frame_rgb, use_gpu=use_gpu)
        # print(len(result))
        if not success:
            if self.debug:
                cv2.imshow("Debug", frame_rgb)
                cv2.waitKey(1)  
            return [0,0,0,0,0,0,0,0]
        else:
            if self.debug:
                tmp_img = frame_rgb.copy()
                for _, point in enumerate(face_landmark):
                    cv2.circle(tmp_img, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
                cv2.imshow("Debug", tmp_img)
                cv2.waitKey(1)

            action = self.judgeaction(face_landmark, frame_rgb)
            for ind, i in enumerate(action):
                if i != 0:
                    print(self.actionset[ind])        
            return action
                

        # cv2.destroyAllWindows()
