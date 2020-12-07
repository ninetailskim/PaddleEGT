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
        self.actionset = ["left","right","up","down","mopen","mheri","","","lefteye","righteye"]
        self.leftHist = None
        self.rightHist = None
        self.model_points = np.array([
            [6.825897, 6.760612, 4.402142],
            [1.330353, 7.122144, 6.903745],
            [-1.330353, 7.122144, 6.903745],
            [-6.825897, 6.760612, 4.402142],
            [5.311432, 5.485328, 3.987654],
            [1.789930, 5.393625, 4.413414],
            [-1.789930, 5.393625, 4.413414],
            [-5.311432, 5.485328, 3.987654],
            [2.005628, 1.409845, 6.165652],
            [-2.005628, 1.409845, 6.165652],
            [2.774015, -2.080775, 5.048531],
            [-2.774015, -2.080775, 5.048531],
            [0.000000, -3.116408, 6.097667],
            [0.000000, -7.415691, 4.070434],
            [-7.308957, 0.913869, 0.000000],
            [7.308957, 0.913869, 0.000000],
            [0.746313,0.348381,6.263227],
            [0.000000,0.000000,6.763430],
            [-0.746313,0.348381,6.263227],
            ], dtype='float')
        self.reprojectsrc = np.float32([
            [10.0, 10.0, 10.0],
            [10.0, 10.0, -10.0], 
            [10.0, -10.0, -10.0],
            [10.0, -10.0, 10.0], 
            [-10.0, 10.0, 10.0], 
            [-10.0, 10.0, -10.0], 
            [-10.0, -10.0, -10.0],
            [-10.0, -10.0, 10.0]])
        # 头部3D投影点连线
        self.line_pairs = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        # self.leftSize = 0
        # self.rightSize = 0

    def get_face_landmark(self, image, use_gpu):
        try:
            res = self.module.keypoint_detection(images=[image], use_gpu=use_gpu)
            return True, res[0]['data'][0]
        except Exception as e:
            logger.error("Get face landmark localization failed! Exception: %s " % e)
            return False, None
        
    def get_image_points_from_landmark(self, face_landmark):
        """
        从face_landmark_localization的检测结果抽取姿态估计需要的点坐标
        """
        image_points = np.array([
            face_landmark[17], face_landmark[21], 
            face_landmark[22], face_landmark[26], 
            face_landmark[36], face_landmark[39], 
            face_landmark[42], face_landmark[45], 
            face_landmark[31], face_landmark[35],
            face_landmark[48], face_landmark[54],
            face_landmark[57], face_landmark[8],
            face_landmark[14], face_landmark[2], 
            face_landmark[32], face_landmark[33],
            face_landmark[34], 
            ], dtype='float')
        return image_points
    
    def caculate_pose_vector(self, image_points):
        """
        获取旋转向量和平移向量
        """
        # 相机视角
        center = (self.img_size[1]/2, self.img_size[0]/2)
        focal_length = center[0] / np.tan(60/ 2 * np.pi / 180)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]],
            dtype = "float")
        # 假设没有畸变
        dist_coeffs = np.zeros((4,1))
        
        success, rotation_vector, translation_vector= cv2.solvePnP(self.model_points, 
                                                                   image_points,
                                                                   camera_matrix, 
                                                                   dist_coeffs)
                                                                   
        reprojectdst, _ = cv2.projectPoints(self.reprojectsrc, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        return success, rotation_vector, translation_vector, camera_matrix, dist_coeffs, reprojectdst

    def caculate_euler_angle(self, rotation_vector, translation_vector):
        """
        将旋转向量转换为欧拉角
        """
        rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
        proj_matrix = np.hstack((rvec_matrix, translation_vector))
        euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
        pitch, yaw, roll = [math.radians(_) for _ in euler_angles]
        return pitch, yaw, roll

    
    def classify_pose_in_euler_angles(self, img):
        self.img_size = img.shape

        success, face_landmark = self.get_face_landmark(img, False)

        if not success:
            logger.info("Get face landmark localization failed! Please check your image!")
            return None

        image_points = self.get_image_points_from_landmark(face_landmark)
        success, rotation_vector, translation_vector, camera_matrix, dist_coeffs, reprojectdst = self.caculate_pose_vector(image_points)
        
        if not success:
            logger.info("Get rotation and translation vectors failed!")
            return None

        # 画出投影正方体
        alpha=0.3
        if not hasattr(self, 'before'):
            self.before = reprojectdst
        else:
            reprojectdst = alpha * self.before + (1-alpha)* reprojectdst
        reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))
        for start, end in self.line_pairs:
            cv2.line(img, reprojectdst[start], reprojectdst[end], (0, 0, 255))

        # 计算头部欧拉角
        pitch, yaw, roll = self.caculate_euler_angle(rotation_vector, translation_vector)
        cv2.putText(img, "pitch: " + "{:7.2f}".format(pitch), (20, int(self.img_size[0]/2 -10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 255), thickness=2)
        cv2.putText(img, "yaw: " + "{:7.2f}".format(yaw), (20, int(self.img_size[0]/2 + 30) ), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 255), thickness=2)
        cv2.putText(img, "roll: " + "{:7.2f}".format(roll), (20, int(self.img_size[0]/2 +70)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 255), thickness=2)
        # for index, action in enumerate(index_action):
        #     cv2.putText(img, "{}".format(self._index_action[action]), index_action[action][1], 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 50, 50), thickness=2)
        # frames_euler.append([index, img, pitch, yaw, roll])

        return img

    def control(self, use_gpu=False):
        ret, frame_rgb = self.capture.read()
        frame_rgb = cv2.flip(frame_rgb,1)
        img = self.classify_pose_in_euler_angles(frame_rgb)
        if img is not None:
            cv2.imshow("Debug", img)    
        else:
            cv2.imshow("Debug", frame_rgb)
        cv2.waitKey(1)

        # cv2.destroyAllWindows()

'''
yaw
> 0.45 left
< -0.45 right

< -0.35 up
> 0.2 down
'''