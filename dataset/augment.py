import numpy as np
import cv2
import random


def transform_gaussian_noise(img, mean = 0.0, var = 10.0):
    img = np.array(img)
    height, width, channels = img.shape
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma,(height, width, channels))
    noisy = img + gauss
    cv2.normalize(noisy, noisy, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    noisy = noisy.astype(np.uint8)
    
    return noisy


def _motion_blur(img, kernel_size):
    kernel_v = np.zeros((kernel_size, kernel_size)) 
    kernel_h = np.copy(kernel_v) 
    kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size) 
    kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size) 
    
    # Normalize. 
    kernel_v /= kernel_size 
    kernel_h /= kernel_size 
    if np.random.uniform() > 0.5:
        blurred = cv2.filter2D(img, -1, kernel_v) 
    else:
        blurred = cv2.filter2D(img, -1, kernel_h)

    return blurred


def transform_random_blur(img):
    img = np.array(img)
    flag = np.random.uniform()
    kernel_size = random.choice([3, 5, 7, 9])
    
    if flag >= 0.75:
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), np.random.uniform(0.0, 2.0))
    elif flag >= 0.5:
        img = _motion_blur(img, kernel_size)
    else:
        img = cv2.blur(img, (kernel_size, kernel_size))
        
    return img


def transform_adjust_gamma(image, lower = 0.2, upper = 2.0):
    image = np.array(image)
    gamma = np.random.uniform(lower, upper)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    
    return cv2.LUT(image, table)


def transform_resize(image, resize_range = (24, 112)):
    assert resize_range[0] < resize_range[1], "Lower and higher value not accepted: {} vs {}".format(resize_range[0], resize_range[1])
    resize_value = np.random.randint(resize_range[0], resize_range[1])
    inter = random.choice([cv2.INTER_LINEAR,
                        cv2.INTER_NEAREST])
    height, width, _ = image.shape
    scale = min(resize_value/height, resize_value/width)
    new_width = int(width*scale)
    new_height = int(height*scale)
    resize_image = cv2.resize(image, (new_width, new_height), interpolation = inter)
    
    return cv2.resize(resize_image, (width, height), interpolation = inter)


def adjust_brightness(img, factor):
    # factor > 1: sáng hơn, factor < 1: tối hơn
    return np.clip(img * factor, 0, 255).astype(np.uint8)

def adjust_contrast(img, factor):
    # factor > 1: tăng tương phản, factor < 1: giảm tương phản
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    return np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)

def adjust_saturation(img, factor):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

def adjust_hue(img, factor):
    # factor tính theo tỷ lệ (0.5 = xoay 180°, 1.0 = xoay 360° quay về ban đầu)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + factor * 180) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

def adjust_contrast(img, factor):
    # factor > 1: tăng tương phản, factor < 1: giảm tương phản
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    return np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)

def adjust_saturation(img, factor):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

def adjust_hue(img, factor):
    # factor tính theo tỷ lệ (0.5 = xoay 180°, 1.0 = xoay 360° quay về ban đầu)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + factor * 180) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

def transform_color_jitter_np(img_rgb,
                               p_brightness=0.3, p_contrast=0.3,
                               p_saturation=0.3, p_hue=0.0,
                               range_brightness=0.2, range_contrast=0.2,
                               range_saturation=0.2, range_hue=0.05):
    """
    p_* : xác suất áp dụng phép biến đổi
    range_* : biên độ thay đổi tối đa (nếu được chọn áp dụng)
    """
    img_out = img_rgb.copy()

    ops = []

    if np.random.rand() < p_brightness:
        b_factor = np.random.uniform(1 - range_brightness, 1 + range_brightness)
        ops.append(lambda x: adjust_brightness(x, b_factor))

    if np.random.rand() < p_contrast:
        c_factor = np.random.uniform(1 - range_contrast, 1 + range_contrast)
        ops.append(lambda x: adjust_contrast(x, c_factor))

    if np.random.rand() < p_saturation:
        s_factor = np.random.uniform(1 - range_saturation, 1 + range_saturation)
        ops.append(lambda x: adjust_saturation(x, s_factor))

    if np.random.rand() < p_hue:
        h_factor = np.random.uniform(-range_hue, range_hue)  # tỉ lệ vòng màu
        ops.append(lambda x: adjust_hue(x, h_factor))

    # Xáo trộn thứ tự
    np.random.shuffle(ops)

    # Áp dụng
    for op in ops:
        img_out = op(img_out)

    return img_out


def transform_resize_padding(sample, target_size = [224, 224]):
    sample = np.array(sample)
    height, width, _ = sample.shape
    scale = min(target_size[0]/height, target_size[1]/width)
    new_width = int(width*scale)
    new_height = int(height*scale)
    img = np.zeros((target_size[0], target_size[1], 3), dtype = np.uint8)
    img[:new_height, :new_width] = cv2.resize(sample, (new_width, new_height))
    
    return img


def random_mask_rgb(image_rgb, max_ratio=0.2):
    """
    Che ngẫu nhiên một phần ảnh RGB bằng màu ngẫu nhiên.
    Diện tích che <= max_ratio (tỉ lệ ảnh).
    """
    h, w, _ = image_rgb.shape
    area = h * w

    # Random diện tích che
    mask_area = random.uniform(0.05, max_ratio) * area

    # Random tỷ lệ khung mask
    mask_aspect_ratio = random.uniform(0.5, 2.0)
    mask_h = int(round((mask_area * mask_aspect_ratio) ** 0.5))
    mask_w = int(round((mask_area / mask_aspect_ratio) ** 0.5))

    # Giới hạn kích thước mask
    mask_h = min(mask_h, h)
    mask_w = min(mask_w, w)

    # Random vị trí
    top = random.randint(0, h - mask_h)
    left = random.randint(0, w - mask_w)

    # Random màu RGB
    random_color = tuple(np.random.randint(0, 256, size=3).tolist())

    # Che ảnh
    image_rgb[top:top+mask_h, left:left+mask_w] = random_color

    return image_rgb

def random_augment(sample, cfg):
    img = np.array(sample)
    # 1. Color Jitter (brightness / contrast / saturation / hue)
    if np.random.rand() < cfg.color_jitter.prob:
        img = transform_color_jitter_np(
            img_rgb=img,
            p_brightness=cfg.color_jitter.ratio[0],  # bên trong sẽ random order
            p_contrast=cfg.color_jitter.ratio[1],
            p_saturation=cfg.color_jitter.ratio[2],
            p_hue=cfg.color_jitter.ratio[3],
            range_brightness=cfg.color_jitter.range[0],
            range_contrast=cfg.color_jitter.range[1],
            range_saturation=cfg.color_jitter.range[2],
            range_hue=cfg.color_jitter.range[3]
        )

    # 2. Resize ngẫu nhiên (thu/phóng rồi trả về kích thước ban đầu)
    if np.random.rand() < cfg.resize.prob:
        img = transform_resize(img, resize_range=cfg.resize.range)

    # 3. Gaussian Noise
    if np.random.rand() < cfg.gaussian_noise.prob:
        img = transform_gaussian_noise(img, mean=cfg.gaussian_noise.mean, var=cfg.gaussian_noise.var)

    # 4. Flip trái–phải
    if np.random.rand() < cfg.flip.prob:
        img = cv2.flip(img, cfg.flip.direction) 

    # 5. Blur
    if np.random.rand() < cfg.blur.prob:
        img = transform_random_blur(img)

    # 6. Mask     
    if np.random.rand() < cfg.mask.prob:
        img = random_mask_rgb(img, max_ratio=cfg.mask.max_ratio)

    return img