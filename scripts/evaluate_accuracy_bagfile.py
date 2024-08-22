import os
import cv2
import torch
import time
import json
import numpy as np
import rosbag
import custom_utils.data_conversion as data_conversion
import custom_utils.depth_anything_interface as depth_anything_interface

MODEL_TYPE = "metric"
DATASET = "hypersim"
ENCODER = "vitl"
DATAFILE = "/scratchdata/moving_2L"
CAMERA_JSON = "/scratchdata/gemini_2l.json"
MODEL_PATH = f"/scratchdata/depth_anything_v2_{ENCODER}.pth"
with open(CAMERA_JSON, 'r') as f:
    CAMERA_DATA = json.load(f)
SQUARE_SIZE = 24 #mm
CHECKERBOARD = (8,6)
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
MODEL = depth_anything_interface.get_model(DEVICE, MODEL_PATH, model_type = MODEL_TYPE, encoder=ENCODER)
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
prev_time = time.time()

# Store accuracy per frame
data = []

for topic, msg, t in bag.read_messages(topics=["/camera/color/image_raw", "/camera/depth/image_raw"]):
    if topic == "/camera/color/image_raw":
        img = data_conversion.topic_to_image(msg)
    elif topic == "/camera/depth/image_raw":
        depth = data_conversion.topic_to_depth(msg,CAMERA_DATA)
        
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
        
        pred_depth, _ = depth_anything_interface.get_pred_depth(depth, est_depth, CAMERA_DATA, depth_anything_interface.estimated_metric_depth_model)
        depth_anything_depth = data_conversion.interpolate_depth(pred_depth,corners.reshape(-1, 2))
    
        diff_raw = abs(raw_depth - camera_estimated_depth)
        diff_depth_anything = abs(depth_anything_depth - camera_estimated_depth)
    
        # Store data
        data.append((time.time() - prev_time, diff_raw.mean(), diff_raw.max(), diff_raw.std(), diff_depth_anything.mean(), diff_depth_anything.max(), diff_depth_anything.std()))

        depth = None
        img = None
        prev_time = time.time()

# Write data to csv

data = np.array(data)
np.savetxt(os.path.join(DATAFILE, "accuracy.csv"), data, delimiter=",", header="time,mean_raw,max_raw,std_raw,mean_depth_anything,max_depth_anything,std_depth_anything", comments='')