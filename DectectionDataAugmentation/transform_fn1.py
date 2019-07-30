# -*- coding=utf-8 -*-
##############################################################
# 包括:
#     1. 裁剪(需改变bbox)
#     2. 改变对比度, 光照等
#     4. 加噪声
#     5. 旋转角度(需要改变bbox)
#     6. 镜像
#     7. cutout
#     8. 运动模糊
# 注意:
#     random.seed(),相同的seed,产生的随机数是一样的!!

import torch
from torchvision import transforms
import cv2
import numpy as np
import types
import math
from numpy import random

random.seed(5)

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size,
                                 self.size))
        return image, boxes, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels


class RandomSampleCrop(object):
    """Crop可能会切到bbox
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self, lower=0.0, upper=0.1, crop_scale = 0.7):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (lower, upper),
        )
        self.crop_scale = crop_scale

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(self.crop_scale * width, width)
                h = random.uniform(self.crop_scale * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                #current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, labels


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))

        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels


class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes


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

class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)

class RandomRotate(object):
    def __init__(self, rot=10, scale=1.0):
        self.rotate = rot
        self.scale = scale

    def __call__(self, img, bboxes, labels= None):
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
        if random.randint(2):
            angle = np.clip(self.rotate * random.normal(), -self.rotate, self.rotate)
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

            # ---------------------- 矫正bbox坐标 ----------------------
            # rot_mat是最终的旋转矩阵
            # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
            rot_bboxes = list()
            for bbox in bboxes:
                xmin = bbox[0]
                ymin = bbox[1]
                xmax = bbox[2]
                ymax = bbox[3]
                point1 = np.dot(rot_mat, np.array([xmin, ymin, 1]))
                point2 = np.dot(rot_mat, np.array([xmax, ymin, 1]))
                point3 = np.dot(rot_mat, np.array([xmin, ymax, 1]))
                point4 = np.dot(rot_mat, np.array([xmax, ymax, 1]))
                rx_min, ry_min, rx_max, ry_max = 1e10, 1e10, 0.0, 0.0

                for point in [point1, point2, point3, point4]:
                    rx_min = min(rx_min, point[0])
                    rx_max = max(rx_max, point[0])
                    ry_min = min(ry_min, point[1])
                    ry_max = max(ry_max, point[1])

                # 加入list中
                rot_bboxes.append([rx_min, ry_min, rx_max, ry_max])

            return rot_img, np.array(rot_bboxes), labels
        else:
            return img, np.array(bboxes), labels

class CutOut(object):#随机加黑斑
    def __init__(self, length=100, n_holes=1, threshold=0.5, mask_value=125):
        self.length = length
        self.n_holes = n_holes
        self.threshold = threshold
        self.mask_value = mask_value

    def __call__(self, img, bboxes, labels=None):
        '''
        原版本：https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
        Randomly mask out one or more patches from an image.
        Args:
            img : a 3D numpy array,(h,w,c)
            bboxes : 框的坐标
            n_holes (int): Number of patches to cut out of each image.
            length (int): The length (in pixels) of each square patch.
        '''

        # 得到h和w
        if random.randint(2):
            length = int(random.randint(2, self.length, 1))
            h, w, c = img.shape
            for n in range(self.n_holes):
                chongdie = True  # 看切割的区域是否与box重叠太多
                while chongdie:
                    y = random.randint(h)
                    x = random.randint(w)
                    y1 = np.clip(y - length // 2, 0, h)  # numpy.clip(a, a_min, a_max, out=None), clip这个函数将将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min
                    y2 = np.clip(y + length // 2, 0, h)
                    x1 = np.clip(x - length // 2, 0, w)
                    x2 = np.clip(x + length // 2, 0, w)
                    chongdie = False
                    if min(jaccard_numpy(bboxes, torch.tensor([x1, y1, x2, y2]))) > self.threshold:
                        chongdie = True
                        break
                img[y1: y2, x1: x2, :] = self.mask_value
        return img, bboxes, labels


class Motion_blur():
    def __init__(self, degree=8, angle=45):
        self.degree = degree
        self.angle = angle

    def __call__(self, img, bboxes, labels=None):
        '''
        增加运动模糊
        :param image:
        :param degree:
        :param angle:
        :return:
        '''

        # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
        if random.randint(2):
            degree = random.randint(4, self.degree, 1)
            angle = np.clip(self.angle * random.normal(), -self.angle, self.angle)
            M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
            motion_blur_kernel = np.diag(np.ones(degree))
            motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

            motion_blur_kernel = motion_blur_kernel / degree
            img = cv2.filter2D(img, -1, motion_blur_kernel)

            # convert to uint8
            #cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
            #blurred = np.array(blurred, dtype=np.uint8)
        return img, bboxes, labels

class SSDAugmentation(object):
    def __init__(self, size=300, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean)
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)

def show_pic(img, bboxes, name):
    '''
    输入:
        img:图像array
        bboxes:图像的所有boudning box list, 格式为[[x_min, y_min, x_max, y_max]....]
        names:每个box对应的名称
    '''
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 1)

    #cv2.imshow('pic', img)
    cv2.imwrite(name + 'demo.png', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

class Policy1(object):
    def __init__(self):
        self.augment = Compose([
            ConvertFromInts(),
            PhotometricDistort(),
            RandomMirror(),  # bbox内翻转, 非图片翻转
            Motion_blur(),
            #CutOut(),
            RandomRotate(),
            #Expand(self.mean),  #
            #RandomSampleCrop(),  # 重要

        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)

class Policy2(object):
    def __init__(self):
        self.rot = 10
        self.augment = Compose([
            ConvertFromInts(),
            PhotometricDistort(),
            RandomMirror(),  # bbox内翻转, 非图片翻转
            Motion_blur(),
            #CutOut(),
            RandomRotate(self.rot),
            RandomSampleCrop(),  # 重要

        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)

if __name__ == '__main__':
    pic_path = 'C0013.MP4.mp4_result-4602_02_000_23.56.png'
    coords = torch.tensor([[40.99, 510.73, 131.27, 538.23], [55.02, 550.71, 122.59, 586.47]]).numpy()

    img = cv2.imread(pic_path)

    #show_pic(img, coords)  # 原图
    aug1 = Policy1()
    aug2 = Policy2()
    augssd = SSDAugmentation()
    crop = RandomSampleCrop()
    cutout = CutOut()
    for i in range(10):

        #auged_img, auged_bboxes, _ = augssd(img, coords, labels=None)
        #show_pic(auged_img, auged_bboxes, 'ssd/' + str(i))  # 强化后的图
        #auged_img, auged_bboxes, _ = aug1(img, coords, labels=None)
        #show_pic(auged_img, auged_bboxes, 'policy1/' + str(i))  # 强化后的图
        auged_img, auged_bboxes, _ = aug2(img, coords, labels=None)
        show_pic(auged_img, auged_bboxes, 'policy2/' + str(i))  # 强化后的图
        #print(i, img.shape, auged_img.shape, auged_bboxes)
        #auged_img, auged_bboxes, _ = cutout(img, coords, labels=None)
        #show_pic(auged_img, auged_bboxes, str(i))  # 强化后的图
