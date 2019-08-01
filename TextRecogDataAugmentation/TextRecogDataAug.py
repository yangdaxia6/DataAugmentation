# -*- coding=utf-8 -*-
##############################################################
# 包括:
#     1. hsv通道, 亮度等
#     2. 改变对比度, 光照等
#     3. 旋转角度
#     4. 裁剪(随机裁剪某些上下左右四个方向)
#     5. 外扩
#     6. 镜像
#     7. cutout(随机遮挡一部分)
#     8. 运动模糊
#     9. 高斯模糊
#     10.压缩图像质量
#     11.透视(未实现)


import torch
from torchvision import transforms
import cv2
import numpy as np
import types
import math
from numpy import random

random.seed(2)

def is_aug(prob):
    return random.uniform() > 1-prob


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)


class ConvertFromInts(object):
    def __call__(self, image):
        return image.astype(np.float32)


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32)


class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image):
        image = cv2.resize(image, (self.size,
                                 self.size))
        return image


class RandomSaturation(object): #色调HSV, 通道1
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image


class RandomHue(object): #饱和度HSV, 通道0
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image


class RandomLightingNoise(object):  #改变bgr通道
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image


class RandomContrast(object):  #改变通道
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image


class RandomBrightness(object): #随机亮度
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image


class ToCV2Image(object):
    def __call__(self, tensor):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0))


class ToTensor(object):
    def __call__(self, cvimage):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1)


class RandomSampleCrop(object):
    """Crop随机切一边或者多边
    Arguments:
        img (Image): the image being input during training
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img)
            img (Image): the cropped image
    """
    def __init__(self, x_scale = 0.07, y_scale = 0.125, prob = 0.5):
        self.sample_options = (
            # using entire original input image
            'left',
            'right',
            'top',
            'bottom'
        )
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.prob = prob

    def __call__(self, image):
        if is_aug(self.prob):
            height, width, _ = image.shape
            x1, x2, y1, y2 = 0, width, 0, height
            modes = random.choice(self.sample_options, random.randint(1, 5))
            if 'left' in modes:
                x1 = int(width*self.x_scale)
            if 'right' in modes:
                x2 = int(width - width*self.x_scale)
            if 'top' in modes:
                y1 = int(height * self.y_scale)
            if 'bottom' in modes:
                y2 = int(height - height * self.y_scale)
            image = image[y1:y2, x1:x2, :]
        return image


class Expand(object):
    def __init__(self, mean=(104, 117, 123), expand_ratio = 1.2, prob=0.5):
        self.mean = mean
        self.prob = prob
        self.expand_ratio = expand_ratio

    def __call__(self, image):
        if is_aug(self.prob):
            height, width, depth = image.shape
            ratio = random.uniform(1, self.expand_ratio)
            left = random.uniform(0, width*ratio - width)
            top = random.uniform(0, height*ratio - height)

            expand_image = np.zeros(
                (int(height*ratio), int(width*ratio), depth),
                dtype=image.dtype)
            expand_image[:, :, :] = self.mean
            expand_image[int(top):int(top + height),
                         int(left):int(left + width)] = image
            image = expand_image

        return image


class RandomMirror(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image):
        _, width, _ = image.shape
        if is_aug(self.prob):
            image = image[::-1, :, :]

        return image


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image

class RandomRotate(object):
    def __init__(self, rot=10, scale=1.0, prob=0.5):
        self.rotate = rot
        self.scale = scale
        self.prob = prob
        assert self.rotate > 0.0

    def __call__(self, img):
        '''
        参考:https://blog.csdn.net/u014540717/article/details/53301195crop_rate
        输入:
            img:图像array,(h,w,c)
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
            angle:旋转角度
            scale:默认1
        输出:
            rot_img:旋转后的图像array
            rot_bboxes:旋转后的boundingbox坐标list
        '''
        if is_aug(self.prob):
            angle = random.uniform(-self.rotate, self.rotate)
        # ---------------------- 旋转图像 ----------------------
            w = img.shape[1]
            h = img.shape[0]
            # 角度变弧度
            rangle = np.deg2rad(angle)  # angle in radians
            # now calculate new image width and height
            nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * self.scale
            nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * self.scale
            # ask OpenCV for the rotation matrix
            rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, self.scale)
            # calculate the move from the old center to the new center combined with the rotation
            rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
            # the move only affects the translation, so update the translation part of the transform
            rot_mat[0, 2] += rot_move[0]
            rot_mat[1, 2] += rot_move[1]
            # 仿射变换
            rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

            return rot_img
        else:
            return img

class CutOut(object):#随机加黑斑
    def __init__(self, scale=0.5, n_holes=1, mask_value=125, prob = 0.5):
        self.scale = scale
        self.n_holes = n_holes
        self.mask_value = mask_value
        self.prob = prob

    def __call__(self, img):
        #
        if is_aug(self.prob):
            h, w, c = img.shape
            scale = self.scale if h/w<0.35 else 0.25
            print(scale)
            length = int(h*random.choice(random.uniform(0.2, scale, 10)))

            for n in range(self.n_holes):
                y = random.randint(h)
                x = random.randint(w)
                y1 = np.clip(y - length // 2, 0, h)  # numpy.clip(a, a_min, a_max, out=None), clip这个函数将将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min
                y2 = np.clip(y + length // 2, 0, h)
                x1 = np.clip(x - length // 2, 0, w)
                x2 = np.clip(x + length // 2, 0, w)
                img[y1: y2, x1: x2, :] = self.mask_value
        return img


class Motion_blur():
    #运动模糊
    def __init__(self, degree=8, angle=45, prob = 0.5):
        self.degree = degree
        self.angle = angle
        self.prob = prob

    def __call__(self, img):
        '''
        增加运动模糊
        :param image:
        :param degree:
        :param angle:
        :return:
        '''

        # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
        if is_aug(self.prob):
            degree = random.randint(4, self.degree, 1)
            angle = np.clip(self.angle * random.normal(), -self.angle, self.angle)
            M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
            motion_blur_kernel = np.diag(np.ones(degree))
            motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

            motion_blur_kernel = motion_blur_kernel / degree
            img = cv2.filter2D(img, -1, motion_blur_kernel)

        return img

class Gasuss_blur(object):
    def __init__(self, kernel_size=(9, 9), sigma=10, prob=0.5):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.prob = prob

    def __call__(self, image):
        '''
            添加高斯模糊
            kernel_size : 模糊核大小
            sigma : 标准差10大，1小

        '''
        if is_aug(self.prob):
            image = cv2.GaussianBlur(image, self.kernel_size, self.sigma)
        return image

class AdjustJpgQuality(object):
    def __init__(self, scale = 0.5, prob = 0.5):
        self.prob = prob
        self.scale = scale
    def __call__(self, image):
        '''
            调整图像JPG压缩失真程度
            q : 压缩质量 0~100
        '''
        if is_aug(self.prob):
            scale = random.uniform(0.3, self.scale)
            h, w, _ = image.shape
            image = cv2.resize(image, (int(scale * w), int(scale * h)))
            image = cv2.resize(image, (w, h))
        return image

class PhotometricDistort(object):
    def __init__(self, prob=0.5):
        self.pd = [
            RandomContrast(),    #对比度
            ConvertColor(transform='HSV'),
            RandomSaturation(), #随机饱和度
            RandomHue(),        #随机色调
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()   #随机对比度
        ]
        self.rand_brightness = RandomBrightness(delta=64)
        self.prob = prob

    def __call__(self, image):
        im = image.copy()
        im = self.rand_brightness(im)
        if is_aug(self.prob):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im = distort(im)
        return im

class Policy1(object):
    def __init__(self):
        self.augment = Compose([
            ConvertFromInts(),
            PhotometricDistort(),
            AdjustJpgQuality(prob = 0.5),
            Gasuss_blur(prob = 0.2),
            #RandomRotate(),

        ])

    def __call__(self, img):
        return self.augment(img)

def show_pic(img, name):
    '''
    输入:
        img:图像array
        bboxes:图像的所有boudning box list, 格式为[[x_min, y_min, x_max, y_max]....]
        names:每个box对应的名称
    '''

    #cv2.imshow('pic', img)
    cv2.imwrite(name + 'demo.png', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

if __name__ == '__main__':
    import os
    pic_paths = os.listdir('raw')
    #pic_paths = ['raw.jpg']
    for path in pic_paths:
        fn = os.path.join('raw', path)
        img = cv2.imread(fn)

        #show_pic(img, coords)  # 原图
        aug1 = Policy1()


        crop = RandomSampleCrop()
        qual = AdjustJpgQuality()
        #cutout = CutOut(scale=0.5, n_holes=3, mask_value=0)
        for i in range(5):
            im = img.copy()
            auged_img= aug1(img)
            #auged_img= crop(im)
            #auged_img = qual(im)
            show_pic(auged_img, 'policy1/' + str(i)+path)  # 强化后的图
