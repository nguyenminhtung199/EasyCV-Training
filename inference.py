import onnxruntime
import cv2
import numpy as np

def resize_and_pad(img, target_size=224):
    aspect_ratio = img.width / img.height
    
    if img.width > img.height:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_height = target_size
        new_width = int(target_size * aspect_ratio)
    
    img = cv2.resize(img, (new_width, new_height))
    
    img_array = np.array(img)
    
    padding_left = (target_size - new_width) // 2
    padding_top = (target_size - new_height) // 2
    padding_right = target_size - new_width - padding_left
    padding_bottom = target_size - new_height - padding_top
    
    if len(img_array.shape) == 2:
        padded_img = np.pad(
            img_array,
            ((padding_top, padding_bottom), (padding_left, padding_right)),
            mode='constant',
            constant_values=0
        )
    else: 
        padded_img = np.pad(
            img_array,
            ((padding_top, padding_bottom), (padding_left, padding_right), (0, 0)),
            mode='constant',
            constant_values=0
        )    
    return padded_img

def preprocess(image_path, target_size):
    image = cv2.imread(image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # image = resize_and_pad(image, target_size)
    image = cv2.resize(image, target_size)

    image = np.float32(image)

    image = np.transpose(image, (2, 0, 1))
    
    image = np.expand_dims(image, axis=0)

    return image


image_path = "/home1/data/tungcao/face_attribute/trainning/photo_2025-08-16_11-21-57.jpg"
onnx_path = "/home1/data/tungcao/face_attribute/trainning/weight/model_v1/model_mask.onnx"

image_processed = preprocess(image_path, target_size=(224,224))
ort_session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
ort_inputs = {ort_session.get_inputs()[0].name: image_processed}
ort_outs = ort_session.run(None, ort_inputs)

print(ort_outs)
