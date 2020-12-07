import sys
import pygame
import cv2
import numpy as np
import paddlehub as hub
import os
import math

img_dir = 'tmpimg'

def load_image(name, colorkey=None):
    fullname = os.path.join(img_dir, name)
    try:
        image = pygame.image.load(fullname)
    except pygame.error:
        print("Cannot load image:", fullname)
        raise SystemExit(str(geterror()))
    
    if image.get_alpha() is None:
        image = image.convert()
    else:
        image = image.convert_alpha()

    if colorkey is not None:
        if colorkey == -1:
            colorkey = image.get_at((0, 0))
        image.set_colorkey(colorkey, pygame.RLEACCEL)
    return image, image.get_rect()

class Ball(pygame.sprite.Sprite):
    def __init__(self, name, vector):
        self.img, self.rect = load_image(name)
        screen = pygame.display.get_surface()
        self.area = screen.get_rect()
        self.vector = vector

    def calcnewpos(self, rect, vector):
        (angle, z) = vector
        (dx, dy) = (z*math.cos(angle), z*math.sin(angle))
        return rect.move(dx, dy)

    def update(self):
        newpos = self.calcnewpos(self.rect, self.vector)
        self.rect = newpos

# 将预测结果写成视频
src_img = cv2.imread('./3.jpg')

shape = src_img.shape
ss = (round(shape[1]/3),round(shape[0]/3))
src_img = cv2.resize(src_img, ss, interpolation=cv2.INTER_AREA)

print(ss)

cv2.imshow("xxx",src_img)

left_img = src_img[150:200,100:150,:]
cv2.imshow("sda", left_img)

module = hub.Module(name="face_landmark_localization")
result = np.array(module.keypoint_detection(images=[src_img])[0]['data'][0], dtype=np.int32)

lefteye1 = [result[36][0], result[37][1]]
lefteye2 = [result[39][0], result[41][1]]
righteye1 = [result[42][0], result[43][1]]
righteye2 = [result[45][0], result[47][1]]

src_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)

print(result)

bounder_pairs =[[11,14],[14,17],[17,25],[25,20],[20,1],[1,4],[4, 7]]

pygame.init()
size = ss
screen = pygame.display.set_mode(size)
pygame.display.set_caption('basic pygame program')

src_img = np.rot90(src_img,k=-1)




img = pygame.surfarray.make_surface(src_img)
img = pygame.transform.flip(img, False, True)

for start, end in bounder_pairs:
    pygame.draw.line(img, (255, 0, 0), (result[start - 1]), (result[end - 1]), 5)

screen.blit(img,(0,0))
ball = Ball("ball.gif",(0.47,3))
clock = pygame.time.Clock()
while 1:
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit() 
    timg = img.copy()
    #screen.blit(img, ball.rect, ball.rect)
    ball.update()
    timg.blit(ball.img, ball.rect)
    screen.blit(timg, ball.rect, ball.rect)
    pygame.display.flip()  
    clock.tick(50)

pygame.quit()

