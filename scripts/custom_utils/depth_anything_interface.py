#!/usr/bin/python3

#import sys
#sys.path.append("/Depth-Anything-V2")

#import os
#os.chdir("/Depth-Anything-V2")
#print(os.getcwd())

import cv2
import torch
import numpy as np
from scipy.optimize import curve_fit

# These functions aim to fit based on disparity map

def get_model(DEVICE, MODEL_PATH, model_type = "base", encoder='vitl', max_depth=20.0):
    if model_type == "base":
        import sys
        sys.path.append("/Depth-Anything-V2")
        from depth_anything_v2.dpt import DepthAnythingV2

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        model = DepthAnythingV2(**model_configs[encoder])
    elif model_type == "metric":
        import sys
        sys.path.append("/Depth-Anything-V2/metric_depth")

        from depth_anything_v2.dpt import DepthAnythingV2

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }

        model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    else:
        print("Invalid Model Type")
        return None

    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.to(DEVICE).eval()
    return model

def estimated_depth_model(x, a, b, c):
    return (a / (x + b)) + c

def get_pred_depth(depth, est_depth, CAMERA_DATA, verbose=False):
    depth_flatten = depth.flatten()
    est_depth_flatten = est_depth.flatten()
    #Ignore pixels with 0 depth in depth image
    est_depth_flatten = est_depth_flatten[depth_flatten!=CAMERA_DATA["min_range"]]
    depth_flatten = depth_flatten[depth_flatten!=CAMERA_DATA["min_range"]]

    popt, pcov = curve_fit(estimated_depth_model, est_depth_flatten, depth_flatten)
    a_opt, b_opt, c_opt = popt

    if verbose: print(f"a: {a_opt}, b: {b_opt}, c: {c_opt}")

    pred_depth = estimated_depth_model(est_depth, a_opt, b_opt, c_opt)

    # Calcualte r2
    if verbose:
        r2 = np.corrcoef(estimated_depth_model(est_depth_flatten,a_opt,b_opt,c_opt), depth_flatten)[0, 1]
        print(f"R2: {r2}")
    
    return pred_depth, popt