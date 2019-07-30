# coding = utf-8
# auth: yangyuxin
# email: yangdx6@126.com
import numpy as np
import cv2

def add_haze(image, t=0.2, A=0.5):
    '''
        添加雾霾
        t : 透视率 0~1
        A : 大气光照
    '''
    out = image*t + A*255*(1-t)
    return out


def add_gasuss_noise(image,sigma = 30):

    row,col,ch= image.shape
    mean = 0
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    out = noisy.astype(np.uint8)
    return out

def add_gasuss_blur(image, kernel_size=(9, 9), sigma=10):
    '''
        添加高斯模糊
        kernel_size : 模糊核大小
        sigma : 标准差10大，1小

    '''
    out = cv2.GaussianBlur(image, kernel_size, sigma)
    return out

def add_motion_blur(image, degree=18, angle=45):
    '''
    增加运动模糊
    :param image:
    :param degree:
    :param angle:
    :return:
    '''
    image = np.array(image)

    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)

    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred

def adjust_image(image, cont=0.3, bright=20):
    '''
        调整对比度与亮度
        cont : 对比度，调节对比度应该与亮度同时调节
        bright : 亮度
    '''
    out = np.uint8(np.clip((cont * image + bright), 0, 255))
    # tmp = np.hstack((img, res))  # 两张图片横向合并（便于对比显示）
    return out

def adjust_image_hsv(image, h=1, s=1, v=0.5):
    '''
        调整HSV通道，调整V通道以调整亮度
        各通道系数
    '''
    HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(HSV)
    H2 = np.uint8(H * h)
    S2 = np.uint8(S * s)
    V2 = np.uint8(V * v)
    hsv_image = cv2.merge([H2, S2, V2])
    out = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return out

def adjust_jpg_quality(image, q=3, save_path=None):
    '''
        调整图像JPG压缩失真程度
        q : 压缩质量 0~100
    '''
    cv2.imwrite("jpg_tmp.jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    out = cv2.imread('jpg_tmp.jpg')
    return out

def test_methods():
    img = cv2.imread('lp.jpg')
    out = add_haze(img)
    cv2.imwrite("add_haze.jpg", out)
    
    out = add_gasuss_noise(img)
    cv2.imwrite("add_gasuss_noise.jpg", out)
    
    out = add_gasuss_blur(img)
    cv2.imwrite("add_gasuss_blur.jpg", out)
    
    out = adjust_image(img)
    cv2.imwrite("adjust_image.jpg", out)
    
    out = adjust_image_hsv(img)
    cv2.imwrite("adjust_image_hsv.jpg", out)
    
    adjust_jpg_quality(img, save_path='adjust_jpg_quality.jpg')
    
    out = add_motion_blur(img)
    cv2.imwrite("motion_blur.jpg", out)
    
def random_process_img(img_name):

    img = cv2.imread(img_name)
    methods = [add_haze, add_gasuss_noise, add_gasuss_blur, add_motion_blur,
              adjust_image, adjust_image_hsv, adjust_jpg_quality]
    idx = np.random.randint(0, 7)
    idx = 1
    method = methods[idx]
    out = method(img)
    '''
    im_i = img.copy()
    for idx in idxs:

        method = methods[idx]
        im_i = method(im_i)
    out = im_i
    cv2.imshow('0', out)
    '''

    cv2.imwrite(str(str(method).split()[1]) + '_' +img_name, out)


#test_methods()
random_process_img('lp_4_7.jpg')