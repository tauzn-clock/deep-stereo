import os
import time
import open3d as o3d

DATAFILE = "/scratchdata/moving_2L/depth"
#DATAFILE = "/scratchdata/moving_2L/est_depth"

# Read all pcd files in the directory
pcd_files = [f for f in os.listdir(DATAFILE) if f.endswith('.ply')]

# Sort by lexical order
pcd_files.sort()

vis = o3d.visualization.Visualizer()
vis.create_window()

geometry = o3d.io.read_point_cloud(os.path.join(DATAFILE, pcd_files[0]))
vis.add_geometry(geometry)

frame = 0
while True:
    geometry.points = o3d.io.read_point_cloud(os.path.join(DATAFILE, pcd_files[frame])).points
    geometry.colors = o3d.io.read_point_cloud(os.path.join(DATAFILE, pcd_files[frame])).colors
    vis.update_geometry(geometry)
    vis.poll_events()
    vis.update_renderer()
    
    frame += 1
    if frame == len(pcd_files):
        frame = 0
    
    time.sleep(0.1)