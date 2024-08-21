#!/usr/bin/python3

from PIL import Image
import io
import numpy as np

def topic_to_image(msg):
    img_raw = np.frombuffer(msg.data, dtype=np.uint8)
    img = img_raw.reshape((msg.height, msg.width, 3))
    return img

def topic_to_depth(msg, CAMERA_DATA):
    depth_raw = np.frombuffer(msg.data, dtype=np.uint16)
    depth = depth_raw.reshape((msg.height, msg.width))
    depth = depth / (2**16-1) * (CAMERA_DATA["max_range"]- CAMERA_DATA["min_range"]) + CAMERA_DATA["min_range"]
    return depth

def interpolate_depth(arr, coords):
    assert coords.shape[1] == 2

    depth = []
    
    for coord in coords:
        x, y = coord
        assert x >= 0 and x < arr.shape[1]
        assert y >= 0 and y < arr.shape[0]
        
        top_left = arr[int(y), int(x)] * (1 - (x - int(x))) * (1 - (y - int(y)))
        top_right = arr[int(y), int(x+1)] * (x - int(x)) * (1 - (y - int(y)))
        bottom_left = arr[int(y+1), int(x)] * (1 - (x - int(x))) * (y - int(y))
        bottom_right = arr[int(y+1), int(x+1)] * (x - int(x)) * (y - int(y))
        
        depth.append(top_left + top_right + bottom_left + bottom_right)
    
    return np.array(depth)
        