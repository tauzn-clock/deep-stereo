import os
import cv2
import torch
import numpy as np
import rosbag
import matplotlib.pyplot as plt
import custom_utils.data_conversion as data_conversion
import custom_utils.depth_anything_interface as depth_anything_interface

DATAFILE = "/scratchdata/stationary"
FRAME_INDEX = 0
SQUARE_SIZE = 25 #mm
CHECKERBOARD = (8,6)
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
MODEL = depth_anything_interface.get_model(DEVICE)

# Open bag file
bag_file_path = os.path.join(DATAFILE, "raw.bag")
bag = rosbag.Bag(bag_file_path)

for topic, msg, t in bag.read_messages(topics=["/camera/color/camera_info"]):
    D = np.array(msg.D)
    K = np.array(msg.K).reshape((3, 3))
    P = np.array(msg.P).reshape((3, 4))
    R = np.array(msg.R).reshape((3, 3))
    height = msg.height
    width = msg.width
    distortion_model = msg.distortion_model
    break

depth = None
img = None

for topic, msg, t in bag.read_messages(topics=["/camera/color/image_raw", "/camera/depth/image_raw"]):
    if topic == "/camera/color/image_raw":
        img = data_conversion.topic_to_image(msg)
    elif topic == "/camera/depth/image_raw":
        depth = data_conversion.topic_to_depth(msg)
        
    if depth is not None and img is not None:
        
        #Estimate gt depth from camera
        
        # Undistort camera
        img_undistorted = cv2.undistort(img, K, D, P)

        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(img_undistorted, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH)
        if not ret:
            depth = None
            img = None
            continue
        # Estimate distance via PnP
        objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        objp *= SQUARE_SIZE  # Scale by the size of the squares

        ret, rvec, tvec = cv2.solvePnP(objp, corners, K, D)
        
        # Find norm distance between objp and tvec

        camera_estimated_depth = []
        for p in objp:
            camera_estimated_depth.append(np.linalg.norm(p - tvec.reshape(3)))

        camera_estimated_depth = np.array(camera_estimated_depth) / 1000
        
        # Read corresponding depth in raw depth measurements
        raw_depth = data_conversion.interpolate_depth(depth,corners.reshape(-1, 2))
        
        # Estimate depth with depth anything
        est_depth = MODEL.infer_image(np.array(img)) # HxW raw depth map in numpy
        pred_depth, _ = depth_anything_interface.get_pred_depth(depth, est_depth)
        depth_anything_depth = data_conversion.interpolate_depth(pred_depth,corners.reshape(-1, 2))
    
        print("Camera vs Raw", abs(raw_depth - camera_estimated_depth).mean())
        print("Camera vs Depth Anything", abs(depth_anything_depth - camera_estimated_depth).mean())
    
        depth = None
        img = None