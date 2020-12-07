'''
初始化

1拿图像，确定图像
2拿到图像的左眼右眼作为histgram
3制作mask
4转化为pygame可以使用的格式，并画到screen上
'''

'''
while
拿图像，拿到左眼右眼，确定是否操作
如果有操作：
    替换眼睛的贴图
    操作杠杆
'''
    img_mask = np.zeros(ss, np.uint8)
    mask_pairs = [1,4,7,9,11,14,17,25,20]
    mask_point = np.array([res[x - 1] for x in mask_pairs])
    cv2.fillPoly(img_mask, [mask_point], 1)

    mask_pairs = [[28,29,29,28],[30,31,31,30]]
    mask_point = np.array([[res[y - 1][0] + (-2 if i > 1 else 2),res[y - 1][1]] for x in mask_pairs for i, y in enumerate(x,0)])
    mask_point = np.reshape(mask_point, (2,4,2))
    cv2.fillPoly(img_mask, mask_point, 0)

    mask_pairs = [49,50,51,52,53,54,55,56,57,58,59,60]
    mask_point = np.array([res[x - 1] for x in mask_pairs])
    cv2.fillPoly(img_mask, [mask_point], 0)

'''
先取反
然后，看那个线的角度
如果线的角度大于90度，则让她减去180
然后反的角度+2倍的角度

我真的很爱你，嘿嘿，我爱你呦
我草你个傻逼
我真不知道要说啥了
呵呵呵呵
你个大傻逼

'''
class leftBu(pg.sprite.Sprite):
    def __init__(self, name, x, y):
        pg.sprite.Sprite.__init__(self)
        self.image, self.rect = load_image(name)
        self.max_angle = 179;
        self.min_angle = 60;
        self.cur_angle = 60;
        self.controlled = False
        self.transforming = False
        self.rect.left = x
        self.rect.top = y
        self.timage = pg.transform.rotate(self.image, self.cur_angle)
        self.trect = self.rect.copy()

    def up(self):
        self.cur_angle += 0.5
        if self.cur_angle > self.max_angle:
            self.cur_angle = self.max_angle
            self.controlled = False
        self.timage = pg.transform.rotate(self.image, self.cur_angle)
        if self.cur_angle > 90:
            return self.timage, (0, - math.sin(degreeToRad(self.cur_angle - 90)) * self.rect.height)
        else:
            return self.timage, (0, 0)

    def down(self):
        self.cur_angle -= 0.5
        if self.cur_angle < self.min_angle:
            self.cur_angle = self.min_angle
        self.timage = pg.transform.rotate(self.image, self.cur_angle)
        self.transforming = False
        if self.cur_angle > 90:
            return self.timage, (0, - math.sin(degreeToRad(self.cur_angle - 90)) * self.rect.height)
        else:
            return self.timage, (0, 0)

    def update(self):
        print(self.cur_angle)
        if self.controlled:
            _, (tx, ty) = self.up()
            self.trect.top = self.rect.top + ty
            self.trect.left = self.rect.left + tx
        else:
            _, (tx, ty) = self.down()
            self.trect.top = self.rect.top + ty
            self.trect.left = self.rect.left + tx


class rightBu(pg.sprite.Sprite):
    def __init__(self, name, x, y):
        pg.sprite.Sprite.__init__(self)
        self.image, self.rect = load_image(name)
        self.max_angle = 300;
        self.min_angle = 181;
        self.cur_angle = 300;
        self.controlled = False
        self.transforming = False
        self.rect.left = x 
        self.rect.top = y
        self.timage = pg.transform.rotate(self.image, self.cur_angle)
        self.trect = self.rect.copy()

    def up(self):
        self.cur_angle -= 0.5
        if self.cur_angle < self.min_angle:
            self.cur_angle = self.min_angle
            self.controlled = False
        self.timage = pg.transform.rotate(self.image, self.cur_angle)
        if self.cur_angle < 270:
            return self.timage, ((1 - math.cos(degreeToRad(270 - self.cur_angle))) * self.rect.height, -math.sin(degreeToRad(270 - self.cur_angle)) * self.rect.height)
        else:
            return self.timage, ((1 - math.cos(degreeToRad(self.cur_angle - 270))) * self.rect.height, math.sin(degreeToRad(self.cur_angle - 270)) *self.rect.height)

    def down(self):
        self.cur_angle += 0.5
        if self.cur_angle > self.max_angle:
            self.cur_angle = self.max_angle
        self.timage = pg.transform.rotate(self.image,, self.cur_angle)
        self.transforming = False
        if self.cur_angle < 270:
            return self.timage, ((1 - math.cos(degreeToRad(270 - self.cur_angle))) * self.rect.height, -math.sin(degreeToRad(270 - self.cur_angle)) * self.rect.height)
        else:
            return self.timage, ((1 - math.cos(degreeToRad(self.cur_angle - 270))) * self.rect.height, math.sin(degreeToRad(self.cur_angle - 270)) *self.rect.height)

    def update(self):
        print(self.cur_angle)
        if self.controlled:
            _, (tx, ty) = self.up()
            self.trect.top = self.rect.top + ty
            self.trect.left = self.rect.left + tx
        else:
            _, (tx, ty) = self.down()
            self.trect.top = self.rect.top + ty
            self.trect.left = self.rect.left + tx