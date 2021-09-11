#!/usr/bin/env python3


import sys, os
import numpy as np
import rospy, roslib
import imutils
import std_msgs.msg as std_msgs
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2, PointField, sensor_msgs, Image
from ros_numpy import point_cloud2 as pc2
from sensor_msgs import point_cloud2
import time
# import pyrealsense2 as rs
from skimage.measure import LineModelND, ransac
import pandas as pd
from cv_bridge import CvBridge, CvBridgeError
import time
# import pyvista as pv


def read_process_pc(input_path, skiprows=10):
    point_cloud = np.loadtxt(input_path, skiprows=10, max_rows=10000000)
    mean_Z = np.mean(point_cloud, axis=0)[2]
    spatial_query = point_cloud[abs(point_cloud[:, 2] - mean_Z) < 1]
    xyz = spatial_query[:, :3]
    rgb = spatial_query[:, 3:]
    return xyz, rgb


def ret_points(axis, df):
    min_val = round(df[axis].min(), 4)
    max_val = round(df[axis].max(), 4)
    diff = max_val - min_val
    end = True
    incre = diff / 8
    low = min_val
    all_limits = []
    for i in range(20):
        low = round(low, 3)
        high = low + diff / 3
        high = round(high, 4)
        if high <= max_val:
            all_limits.append((low, high))
        else:
            break
        low += incre
    return all_limits


def get_max_segment(df, limits, axis):
    max_nPoints = 0
    low = None
    high = None
    segmented_df = None
    i = j = 0
    for (i, j) in limits:
        nPoints = len(df[(df[axis] >= i) & (df[axis] < j)])
        if nPoints >= max_nPoints:
            max_nPoints = nPoints
            low = i
            high = j
            segmented_df = df[(df[axis] >= low) & (df[axis] < high)]
    print(axis, low, high, len(segmented_df))
    return segmented_df


def run_Ransac(df, type_):
    points = df.values
    if 'wire' in type_.lower():
        threshold = 0.004
    elif 'post' in type_.lower():
        threshold = 0.006
    # 1st iteration for fitting horizontal_line
    model_robust, inliers = ransac(points, LineModelND, min_samples=2,
                                   residual_threshold=threshold, max_trials=1000)
    outliers = inliers == False
    return points[inliers], points[outliers]


def write_txt(f_name, df):
    print("Inside Save File Function")
    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df)
    in_file = open(f_name, "w")
    L = ["""ply
format ascii 1.0
element vertex nPoints
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header\n""".replace('nPoints', str(len(df)))]
    in_file.writelines(L)
    in_file.close()  # to change file access modes
    if df.shape[1] <= 3:
        for col in ['r', 'g', 'b']:
            if col == 'r':
                df.insert(df.shape[1], col, 255)
            else:
                df.insert(df.shape[1], col, 0)
    # Writing arrays to .ply files
    df.to_csv(f_name, header=None, index=None, sep=' ', mode='a')
    print(os.getcwd())
    print("Saved {}".format(f_name))

def array_2_pointcloud2(inliers_array):
    from sensor_msgs.msg import PointCloud2
    import std_msgs.msg
    import sensor_msgs.point_cloud2 as pcl2
   
    if isinstance(inliers_array, np.ndarray):
        temp_df1 = pd.DataFrame(inliers_array)
        #filling pointcloud header
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'camera_realsense_gazebo'
        #create pcl from points
        new_pointcloud = pcl2.create_cloud_xyz32(header, inliers_array)
        print('Converted to pointcloud')
        return new_pointcloud


# def array_to_point_cloud(points, parent_frame="camera_realsense_gazebo"):
#     """ Creates a point cloud message.
#     Args:
#         points: Nx6 array of xyz positions (m) and rgb colors (0..1)
#         parent_frame: frame in which the point cloud is defined
#     Returns:
#         sensor_msgs/PointCloud2 message
#     """
#     import sensor_msgs.msg as sensor_msgs
#     import std_msgs.msg as std_msgs
#     ros_dtype = sensor_msgs.PointField.FLOAT32
#     dtype = np.float32
#     itemsize = np.dtype(dtype).itemsize

#     data = points.astype(dtype).tobytes()

#     fields = [sensor_msgs.PointField(
#         name=n, offset=i * itemsize, datatype=ros_dtype, count=1)
#         for i, n in enumerate('xyzrgb')]

#     header = std_msgs.Header(frame_id=parent_frame, stamp=rospy.Time.now())

#     return sensor_msgs.PointCloud2(
#         header=header,
#         height=1,
#         width=points.shape[0],
#         is_dense=False,
#         is_bigendian=False,
#         fields=fields,
#         point_step=(itemsize * 6),
#         row_step=(itemsize * 6 * points.shape[0]),
#         data=data
#     )


def ransacCallback(data):
    print("data is publishing")
    assert isinstance(data, PointCloud2)
    gen = point_cloud2.read_points(data, field_names=("x", "y", "z"), skip_nans=True)
    time.sleep(0.5)
    xyz_1 = np.array(list(gen))
    if len(xyz_1) > 0:
        temp_df = pd.DataFrame(xyz_1)
        # write_txt('all.ply', my_array)
        v_limits = ret_points(0, temp_df)
        post_df = get_max_segment(temp_df, v_limits, 0)
        h_limits = ret_points(1, temp_df)
        wire_df = get_max_segment(temp_df, h_limits, 1)
        wire_inliers_array, outliers_array = run_Ransac(wire_df, 'wire_df')
        post_inliers_array, outliers_array = run_Ransac(post_df, 'post_df')
        write_txt('wire_inliers_array.ply', wire_inliers_array)
        write_txt('post_inliers_array.ply', post_inliers_array)
        combined_array = np.concatenate((wire_inliers_array, post_inliers_array))  ## Contains Trellis & Frame
        write_txt('combined_array.ply', combined_array)
        # cloud_pc = pv.PolyData(combined_array)

        cloud_pc = array_2_pointcloud2(combined_array)
        image_pub.publish(cloud_pc)
        print("Ransac Pointcloud Published")


def listener():
    print("creating a node")
    rospy.init_node("ransac_pointcloud", anonymous=True)
    rospy.Subscriber('/topic1/mega_pointcloud', PointCloud2, ransacCallback)
    rospy.spin()


if __name__ == "__main__":
    image_pub = rospy.Publisher("/ransac/ransac_pointcloud", PointCloud2, queue_size=10)
    listener()
    

