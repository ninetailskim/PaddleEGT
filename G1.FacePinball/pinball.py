import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pygame as pg
from pygame.compat import geterror
import math
import cv2
import paddlehub as hub
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

img_dir = "tmpimg"
module = hub.Module(name="face_landmark_localization")

def load_image(name, colorkey=None):
    fullname = os.path.join(img_dir, name)
    try:
        image = pg.image.load(fullname)
    except pg.error:
        print("Cannot load image:", fullname)
        raise SystemExit(str(geterror()))
    
    if image.get_alpha() is None:
        image = image.convert()
    else:
        image = image.convert_alpha()

    if colorkey is not None:
        if colorkey == -1:
            colorkey = image.get_at((0, 0))
        image.set_colorkey(colorkey, pg.RLEACCEL)
    return image, image.get_rect()

def degreeToRad(degree):
    return degree * math.pi / 180

class leftBu(pg.sprite.Sprite):
    def __init__(self, name, x, y, len, speed):
        pg.sprite.Sprite.__init__(self)
        self.timage, self.trect = load_image(name)
        self.max_angle = 179
        self.min_angle = 60
        self.cur_angle = 60
        self.controlled = False
        self.timage = pg.transform.scale(self.timage, (self.trect.width,len))
        self.trect = self.timage.get_rect()
        self.trect.left = x
        self.trect.top = y
        self.image = pg.transform.rotate(self.timage, self.cur_angle)
        self.rect = self.trect.copy()
        self.speed = speed

    def up(self):
        self.cur_angle += self.speed
        if self.cur_angle > self.max_angle:
            self.cur_angle = self.max_angle
            self.controlled = False
        self.image = pg.transform.rotate(self.timage, self.cur_angle)
        if self.cur_angle > 90:
            return self.image, (0, - math.sin(degreeToRad(self.cur_angle - 90)) * self.trect.height)
        else:
            return self.image, (0, 0)

    def down(self):
        self.cur_angle -= self.speed
        if self.cur_angle < self.min_angle:
            self.cur_angle = self.min_angle
        self.image = pg.transform.rotate(self.timage, self.cur_angle)        
        if self.cur_angle > 90:
            return self.image, (0, - math.sin(degreeToRad(self.cur_angle - 90)) * self.trect.height)
        else:
            return self.image, (0, 0)

    def update(self):
        if self.controlled:
            _, (tx, ty) = self.up()
            self.rect.top = self.trect.top + ty
            self.rect.left = self.trect.left + tx
        else:
            _, (tx, ty) = self.down()
            self.rect.top = self.trect.top + ty
            self.rect.left = self.trect.left + tx

class rightBu(pg.sprite.Sprite):
    def __init__(self, name, x, y, len, speed):
        pg.sprite.Sprite.__init__(self)
        self.timage, self.trect = load_image(name)
        self.max_angle = 300
        self.min_angle = 181
        self.cur_angle = 300
        self.controlled = False
        self.timage = pg.transform.scale(self.timage, (self.trect.width,len))
        self.trect = self.timage.get_rect()
        self.trect.left = x
        self.trect.top = y
        self.image = pg.transform.rotate(self.timage, self.cur_angle)
        self.rect = self.trect.copy()
        self.speed = speed

    def up(self):
        self.cur_angle -= self.speed
        if self.cur_angle < self.min_angle:
            self.cur_angle = self.min_angle
            self.controlled = False
        self.image = pg.transform.rotate(self.timage, self.cur_angle)
        if self.cur_angle < 270:
            return self.image, ( - math.cos(degreeToRad(270 - self.cur_angle)) * self.trect.height, -math.sin(degreeToRad(270 - self.cur_angle)) * self.trect.height)
        else:
            return self.image, ( - math.cos(degreeToRad(self.cur_angle - 270)) * self.trect.height, 0)

    def down(self):
        self.cur_angle += self.speed
        if self.cur_angle > self.max_angle:
            self.cur_angle = self.max_angle
        self.image = pg.transform.rotate(self.timage, self.cur_angle)        
        if self.cur_angle < 270:
            return self.image, ( - math.cos(degreeToRad(270 - self.cur_angle)) * self.trect.height, -math.sin(degreeToRad(270 - self.cur_angle)) * self.trect.height)
        else:
            return self.image, ( - math.cos(degreeToRad(self.cur_angle - 270)) * self.trect.height, 0)

    def update(self):
        if self.controlled:
            _, (tx, ty) = self.up()
            self.rect.top = self.trect.top + ty
            self.rect.left = self.trect.left + tx
        else:
            _, (tx, ty) = self.down()
            self.rect.top = self.trect.top + ty
            self.rect.left = self.trect.left + tx

class Ball(pg.sprite.Sprite):
    def __init__(self, name, vector,img_mask, wlist, x, y):
        self.image, self.rect = load_image(name)   
        self.vector = vector
        self.img_mask = img_mask
        self.wlist = wlist
        self.image = pg.transform.scale(self.image, (7,7))
        self.rect = self.image.get_rect()
        self.rect.top = y
        self.rect.left = x
        #print(self.rect)

    def calcnewpos(self, rect, vector):
        (angle, z) = vector
        (dx, dy) = (z*math.cos(angle), z*math.sin(angle))
        return rect.move(dx, dy)

    def update(self):
        newpos = self.calcnewpos(self.rect, self.vector)
        self.rect = newpos
        (angle, z) = self.vector
        #print(newpos)
        #print(self.rect)
        tl = [newpos.left, newpos.top]
        tr = [newpos.left + newpos.width, newpos.top]
        bl = [newpos.left, newpos.top + newpos.height]
        br = [newpos.left + newpos.width, newpos.top + newpos.height]

        cotl = self.img_mask[tl[1],tl[0]]
        cotr = self.img_mask[tr[1],tr[0]]
        cobl = self.img_mask[bl[1],bl[0]]
        cobr = self.img_mask[br[1],br[0]]

        if cotl > 0:
            no_angle = - angle
            mi_angle = math.atan(self.wlist[cotl - 1])
            end_angle = no_angle + mi_angle * 2
            self.vector = (end_angle, z)
            return
        if cotr > 0:
            no_angle = - angle
            mi_angle = math.atan(self.wlist[cotr - 1])
            end_angle = no_angle + mi_angle * 2
            self.vector = (end_angle, z)
            return
        if cobl > 0:
            no_angle = - angle
            mi_angle = math.atan(self.wlist[cobl - 1])
            end_angle = no_angle + mi_angle * 2
            self.vector = (end_angle, z)
            return
        if cobr > 0:
            no_angle = - angle
            mi_angle = math.atan(self.wlist[cobr - 1])
            end_angle = no_angle + mi_angle * 2
            self.vector = (end_angle, z)
            return
    
    def updateBou(self, bou):
        (angle, z) = self.vector
        no_angle = -angle
        mi_angle = bou
        end_angle = no_angle + mi_angle * 2
        self.vector = (end_angle, z)
        return 

def preprocess(img):
    result = module.keypoint_detection(images=[img])[0]['data'][0]
    if len(result) == 0:
        return None, None, None
    result = np.array(result, dtype=np.int32)

    lefteye1 = [result[36][0], result[37][1]]
    lefteye2 = [result[39][0], result[41][1]]
    righteye1 = [result[42][0], result[43][1]]
    righteye2 = [result[45][0], result[47][1]]

    #print(lefteye2[1] - lefteye1[1])
    #print(righteye2[1] - righteye1[1])

    left_img = img[lefteye1[1]:lefteye2[1]+1,lefteye1[0]:lefteye2[0],:]
    right_img = img[righteye1[1]:righteye2[1]+1,righteye1[0]:righteye2[0],:]

    return result, left_img, right_img

def makeMask(ss, res):

    img_mask = np.zeros((ss[1],ss[0]), np.uint8)

    bounder_pairs =[
        [11,14],[14,17],[17,25],[25,20],
        [20,1],[1,4],[4, 7],[28,29],[30,31],
        [49,52],[52,55],[55,58],[58,49]]

    wlist = []
    windex = 1

    for start, end in bounder_pairs:
        cv2.line(img_mask, tuple(res[start - 1]), tuple(res[end - 1]), windex, 15)
        windex += 1
        wlist.append((res[start - 1][1] - res[end - 1][1]) / ((res[start - 1][0] - res[end - 1][0]) + 0.01))

    return img_mask, wlist

def calHist(img):
    hist = []
    for i in range(3):
        hist.append(cv2.calcHist([img], [i], None, [256],[0,256]))
    hist = np.vstack(np.array(hist))
    return hist

def cv2pygame(timg ,ss):
    #print(timg.shape)
    timg = cv2.resize(timg, (ss[1], ss[0]), interpolation=cv2.INTER_AREA)
    tpg_img = cv2.cvtColor(timg, cv2.COLOR_RGB2BGR)
    tpg_img = np.rot90(tpg_img,k=-1)
    tpg_img = pg.surfarray.make_surface(tpg_img)
    tpg_img = pg.transform.flip(tpg_img, False, True)
    return tpg_img

def main():
    index = 0
    capture  = cv2.VideoCapture(0) 
    #capture  = cv2.VideoCapture('./test_sample.mov')

    init = 'N'
    while init is 'N':
        ret, frame_rgb = capture.read()
        index += 1
        #print(index)
        result = module.keypoint_detection(images=[frame_rgb.copy()])[0]['data'][0]
        if len(result) == 0:
            continue
        else:
            cv2.imshow("Monitor", frame_rgb)
            cv2.waitKey(10)
            init = input("Detected Your face, Use this face? Y/N")
    cv2.destroyAllWindows()
    shape = frame_rgb.shape
    #print(shape)
    tt = math.floor(max(shape[0],shape[1]) / 1000)
    #print(tt)
    if tt == 0:
        tt = 1
    ss = (round(shape[1]/tt),round(shape[0]/tt))
    src_img = cv2.resize(frame_rgb, ss, interpolation=cv2.INTER_AREA)

    result, left_img, right_img = preprocess(src_img.copy())

    leftStandHist = calHist(left_img)
    rightStandHist = calHist(right_img)
    leftStandSize = left_img.shape
    rightStandSize = right_img.shape
    #leftStandHist = cv2.calcHist([left_img],[])

    bolength = round(math.sqrt((result[6][1] - result[9][1]) ** 2 + (result[6][0] - result[9][0]) ** 2) * 1/2)

    #######
    #制作histgram

    pg_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)

    img_mask, wlist = makeMask(ss, result)
    
    pg.init()
    size = ss
    screen = pg.display.set_mode(size)
    pg.display.set_caption('face ball')

    pg_img = np.rot90(pg_img,k=-1)

    pg_img = pg.surfarray.make_surface(pg_img)
    bg_img = pg.transform.flip(pg_img, False, True)

    bounder_pairs =[[11,14],[14,17],[17,25],[25,20],[20,1],[1,4],[4, 7]]
    for start, end in bounder_pairs:
        pg.draw.line(bg_img, (255, 0, 0), (result[start - 1]), (result[end - 1]), 5)

    screen.blit(bg_img,(0,0))

    ball = Ball("ball.gif",(0.47,5),img_mask,wlist, result[27][0], result[27][1])
    lbou = leftBu("123.png", result[6][0], result[6][1], bolength, 15)
    rbou = rightBu("123.png", result[10][0], result[10][1], bolength, 15)
    lbou.controlled = True
    rbou.controlled = True
    clock = pg.time.Clock()

    while True:
        ret, frame_rgb = capture.read() 

        cv2.imshow("face", frame_rgb)
        cv2.waitKey(10)
        index += 1
        #(index)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if frame_rgb is None:
            break
        frame_rgb = cv2.resize(frame_rgb, ss, interpolation=cv2.INTER_AREA)
        nr, nle, nre = preprocess(frame_rgb)
        ######
        #cale histgram, decide how to caozuo
        if len(nr) != 0:
            lTHist = calHist(nle)
            rTHist = calHist(nre)

            if cv2.compareHist(leftStandHist, lTHist, cv2.HISTCMP_CORREL) < 0.1:
                rbou.controlled = True
            if cv2.compareHist(rightStandHist, rTHist, cv2.HISTCMP_CORREL) < 0.1:
                lbou.controlled = True

            sur_limg = cv2pygame(nle, leftStandSize)
            sur_rimg = cv2pygame(nre, rightStandSize)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return 
        timg = bg_img.copy()
        if len(nr) != 0:
            timg.blit(sur_limg, (result[36][0], result[37][1]))
            timg.blit(sur_rimg, (result[42][0], result[43][1]))
        ball.update()
        lbou.update()
        rbou.update()
        if pg.sprite.collide_mask(ball, lbou):
            ball.updateBou(lbou.cur_angle - 90)
        if pg.sprite.collide_mask(ball, rbou):
            ball.updateBou(rbou.cur_angle - 270)
        timg.blit(ball.image, ball.rect)
        timg.blit(lbou.image, lbou.rect)
        timg.blit(rbou.image, rbou.rect)
        screen.blit(timg, (0,0))
        pg.display.flip()
        clock.tick(30)
    pygame.quit()
if __name__ == '__main__':
    main()