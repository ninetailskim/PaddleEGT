import pygame
from pygame.locals import *
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


def main():
    pygame.init()
    screen = pygame.display.set_mode((640,480))
    pygame.display.set_caption('basic pygame program')

    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill((0,0,0))

    bo,bo_rect = load_image('123.png')
    
    ball = Ball("123.png",(0.47,13))

    print(bo_rect)
    print(background.get_rect())
    print(640 - bo_rect.width)

    angle = 1

    clock = pygame.time.Clock()

    pygame.display.flip()
    flag = True
    while 1:
        for event in pygame.event.get():
            if event.type == QUIT:
                return 
        
        newbo = pygame.transform.rotate(bo, angle)
        if flag:
            angle += 5
        else:
            angle -= 5
        if angle > 170:
            flag = False
        if angle < 10:
            flag = True
        #print(angle)
        background.fill((0,0,0))
        if angle > 90:
            background.blit(newbo, ((640 - bo_rect.width) / 2, (480 - bo_rect.height) / 2 - math.sin((angle - 90)/180*math.pi) * bo_rect.height))
        else:
            background.blit(newbo, ((640 - bo_rect.width) / 2, (480 - bo_rect.height) / 2))
        ball.update()
        re = ball.rect
        print(re.left)
        print(re.top)
        print(re.width)
        print(re.height)
        background.blit(ball.img, ball.rect)
        
        screen.blit(background,(0,0))
        
        
        pygame.display.flip()
        clock.tick(50)

if __name__ == '__main__':
    main()