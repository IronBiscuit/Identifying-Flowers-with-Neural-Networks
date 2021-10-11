from PIL import Image
import numpy as np
import torch

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    with Image.open(image) as im:
        
        size = 256, 256
        im.thumbnail(size, Image.ANTIALIAS)
        
        width, height = im.size
        cropped_width = 224
        cropped_height = 224
        left = (width - cropped_width)/2
        top = (height - cropped_height)/2
        right = (width + cropped_width)/2
        bottom = (height + cropped_height)/2
        im = im.crop((left, top, right, bottom))
    np_image = np.array(im)
    np_image = np_image / 255
    for i in range(224):
        for j in range(224):
            np_image[i][j] = (np_image[i][j] - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            
    np_image = np.transpose(np_image, (2, 0, 1)) 
    return torch.from_numpy(np_image)