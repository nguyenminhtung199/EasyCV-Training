import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
import cv2
import random
from io import BytesIO

def resize_and_pad(img, target_size=224):
    aspect_ratio = img.width / img.height
    if img.width > img.height:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_height = target_size
        new_width = int(target_size * aspect_ratio)
    
    img = img.resize((new_width, new_height))
    padding_left = (target_size - img.size[0]) // 2
    padding_top = (target_size - img.size[1]) // 2
    padding_right = target_size - img.size[0] - padding_left
    padding_bottom = target_size - img.size[1] - padding_top
    img = F.pad(img, (padding_left, padding_top, padding_right, padding_bottom), 0, 'constant')
    return img
    
def transform_JPEGcompression(image, compress_range = (30, 100)):
    '''
        Perform random JPEG Compression
    '''
    assert compress_range[0] < compress_range[1], "Lower and higher value not accepted: {} vs {}".format(compress_range[0], compress_range[1])
    jpegcompress_value = random.randint(compress_range[0], compress_range[1])
    out = BytesIO()
    image.save(out, 'JPEG', quality=jpegcompress_value)
    out.seek(0)
    rgb_image = Image.open(out)
    return rgb_image


def transform_gaussian_noise(img, mean = 0.0, var = 10.0):
    '''
        Perform random gaussian noise
    '''
    img = np.array(img)
    height, width, channels = img.shape
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma,(height, width, channels))
    noisy = img + gauss
    cv2.normalize(noisy, noisy, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    noisy = noisy.astype(np.uint8)
    return Image.fromarray(noisy)


def _motion_blur(img, kernel_size):
    # Specify the kernel size. 
    # The greater the size, the more the motion. 
    # Create the vertical kernel. 
    kernel_v = np.zeros((kernel_size, kernel_size)) 
    # Create a copy of the same for creating the horizontal kernel. 
    kernel_h = np.copy(kernel_v) 
    # Fill the middle row with ones. 
    kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size) 
    kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size) 
    # Normalize. 
    kernel_v /= kernel_size 
    kernel_h /= kernel_size 
    if np.random.uniform() > 0.5:
        # Apply the vertical kernel. 
        blurred = cv2.filter2D(img, -1, kernel_v) 
    else:
        # Apply the horizontal kernel. 
        blurred = cv2.filter2D(img, -1, kernel_h) 
    return blurred

def transform_random_blur(img):
    img = np.array(img)
    flag = np.random.uniform()
    kernel_size = random.choice([3, 5, 7, 9, 11])
    if flag >= 0.6:
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), random.choice([0, 1]))
    elif flag >= 0.20:
        img = _motion_blur(img, kernel_size)
    else:
        img = cv2.blur(img, (kernel_size, kernel_size))
   
    return Image.fromarray(img)

def transform_adjust_gamma(image):
    image = np.array(image)
    gamma = np.random.uniform(0.2, 2.0)
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return Image.fromarray(cv2.LUT(image, table))

def transform_blur(img):
    flag = np.random.uniform()
    kernel_size = random.choice([3, 5, 7, 9])
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return img

def transform_to_gray(img):
    '''
        Perform random gaussian noise
    '''
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(gray)

def transform_resize(image, resize_range = (24, 224), target_size = 224):
    assert resize_range[0] < resize_range[1], "Lower and higher value not accepted: {} vs {}".format(resize_range[0], resize_range[1])
    resize_value = random.randint(resize_range[0], resize_range[1])
    resize_image = image.resize((resize_value, resize_value))
    return resize_image.resize((target_size, target_size))


def transform_eraser(image):
    if np.random.uniform() < 0.1:
        mask_range = random.randint(0, 3)
        image_array = np.array(image, dtype=np.uint8)
        image_array[(7-mask_range)*16:, :, :] = 0
        return Image.fromarray(image_array)
    else:
        return image

def transform_color_jiter(sample, photometric):
    fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            photometric.get_params(photometric.brightness, photometric.contrast,
                                                  photometric.saturation, photometric.hue)
    for fn_id in fn_idx:
        if fn_id == 0 and brightness_factor is not None:
            sample = F.adjust_brightness(sample, brightness_factor)
        elif fn_id == 1 and contrast_factor is not None:
            sample = F.adjust_contrast(sample, contrast_factor)
        elif fn_id == 2 and saturation_factor is not None:
            sample = F.adjust_saturation(sample, saturation_factor)
        elif fn_id == 3 and hue_factor is not None:
            sample = F.adjust_hue(sample, hue_factor)
    return sample

def random_augment(sample, photometric):
    # Blur augmentation
    if np.random.uniform() < 0.3:
        sample = transform_random_blur(sample)

    # # Downscale augmentation
    # if np.random.uniform() < 0.3:
    #     sample = transform_resize(sample, resize_range = (24, 224), target_size = 224)

    # Color augmentation
    if np.random.uniform() < 0.3:
        sample = transform_color_jiter(sample, photometric)
    if np.random.uniform() < 0.3:
        sample = transform_adjust_gamma(sample)

    # Noise augmentation
    if np.random.uniform() < 0.15:
        sample = transform_gaussian_noise(sample, mean = 0.0, var = 10.0)

    # Gray augmentation
    if np.random.uniform() < 0.2:
        sample = transform_to_gray(sample)
    
    # JPEG augmentation
    if np.random.uniform() < 0.5:
        sample = transform_JPEGcompression(sample, compress_range = (15, 100))
    return sample

def jpeg_compression(img, quality=75):
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    return Image.open(buffer)
