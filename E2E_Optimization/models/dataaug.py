from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def random_resized_crop(image, size):
    h, w = image.shape[:2]
    new_h, new_w = size
    
    #print(new_h, new_w)

    # Randomly choose top-left corner for cropping
    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    # Perform cropping and resizing
    cropped_image = image[top: top + new_h, left: left + new_w]

    return cropped_image

def random_rotation(image, angle_range=(-50, 50)):
    # Randomly choose rotation angle
    angle = np.random.uniform(angle_range[0], angle_range[1])

    # Perform rotation
    rotated_image = Image.fromarray(image).rotate(angle, resample=Image.BICUBIC)

    return np.array(rotated_image)
