# -*- coding: utf-8 -*-
# http://pointclouds.org/documentation/tutorials/planar_segmentation.php#planar-segmentation

import pcl
import numpy as np
import random
import math
import cv2
import glob

import pcl.pcl_visualization

def read_calib2(filepath, return_as_dict=False):
    calib_params = dict() if return_as_dict else []

    fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
    if fs.isOpened():
        num_prefixes = int(fs.getNode("num_prefixes").real())
        resize_dims = (int(fs.getNode("resize_dims").at(0).real()), int(fs.getNode("resize_dims").at(1).real()))

        for i in range(num_prefixes):
            prefix = fs.getNode(f"prefix-{i+1}").string()

            calib_prefix = dict(
                camera_matrix=fs.getNode(f"camera_matrix-{i+1}").mat(),
                dist_coeffs=fs.getNode(f"dist_coeffs-{i+1}").mat()
            )

            if i > 0:
                calib_prefix["R"] = fs.getNode("R").mat()
                calib_prefix["t"] = fs.getNode("T").mat()

            if return_as_dict:
                calib_params[prefix] = calib_prefix
            else:
                calib_prefix["resize_dims"] = resize_dims
                calib_params += [calib_prefix]

        if return_as_dict:
            calib_params["resize_dims"] = resize_dims
            calib_params["sync_delay"] = fs.getNode("sync_delay").real()
            calib_params["flags"] = int(fs.getNode("flags").real())
            calib_params["rms"] = fs.getNode("rms").real()

        fs.release()

        return calib_params

    return None


def camera_matrix_to_intrinsics(K):
    '''

    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]

    :param K: camera matrix
    :return: (fx, fy), (cx, cy)
    '''
    return (K[0,0], K[1,1]), (K[0,2], K[1,2])

def prj2rw(D, Kd, keep_organized=False, cutoff=5., scale_d=1.0000000474974513e-03):
    '''
    Algin some other modality to depth-floor-1. The slow more-readable version of this code whould be: align_to_depth_slow,
    which is commented underneath.
    :param D: depth-floor-1 frame
    :param Kd: depth-floor-1's modality camera matrix
    :param Ko: other's modality camera matrix
    :param scale_d: a scaling factor to convert depth-floor-1 values into meters
    :param R: other-to-depth-floor-1 rotation matrix
    :param t: other-to-depth-floor-1 translation vector
    :return: map_x and map_y that can be used in OpenCV's cv2.remap(...)
    '''
    (fx_d, fy_d), (cx_d, cy_d) = camera_matrix_to_intrinsics(Kd.astype(np.float32))

    i = np.repeat(np.linspace(0, D.shape[0]-1, D.shape[0], dtype=np.float32), D.shape[1])
    j = np.tile(np.linspace(0, D.shape[1]-1, D.shape[1], dtype=np.float32), D.shape[0])
    d = np.reshape(D, [np.prod(D.shape),]).astype(np.float32)

    z = d * scale_d
    x = ((j - cx_d) * z) / fx_d
    y = ((i - cy_d) * z) / fy_d

    if cutoff > 0.:
        if keep_organized:
            z[z < cutoff] = 0
        else:
            x = x[z < cutoff]
            y = y[z < cutoff]
            z = z[z < cutoff]

    D_rw = np.concatenate([x[:,np.newaxis], y[:,np.newaxis], z[:,np.newaxis]], axis=1)

    return np.reshape(D_rw, D.shape + (3,)) if keep_organized and cutoff == 0. else D_rw


def make_bg(imgs_bg, consensus="mean"):
    D_mean = np.mean(np.array(imgs_bg), axis=0)
    D_std = np.std(np.array(imgs_bg), axis=0)
    imgs_bg_f = []
    for im in imgs_bg:
        bottom = im < (D_mean - D_std)
        sup = im > (D_mean + D_std)
        im[bottom | sup] = np.nan
        imgs_bg_f += [im]

    if consensus == "mean":
        B = np.nanmean(np.array(imgs_bg_f), axis=0)
    elif consensus == "median":
        B = np.nanmedian(np.array(imgs_bg_f), axis=0)

    return B

def main():
    files_bg = sorted(glob.glob("depth-floor-2/*.png"))
    imgs_bg = [cv2.imread(bg, cv2.IMREAD_UNCHANGED).astype(np.float32) for bg in files_bg]

    D_prj = make_bg(imgs_bg, consensus="median")

    calib_params = read_calib2('depth-floor-2/calibration.yml', return_as_dict=True)
    Kd = calib_params["rs/color"]["camera_matrix"]
    D_rw = prj2rw(D_prj, Kd, keep_organized=True)
    #   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    #
    #   // Fill in the cloud data
    #   cloud->width  = 15;
    #   cloud->height = 1;
    #   cloud->points.resize (cloud->width * cloud->height);
    #
    #   // Generate the data
    #   for (size_t i = 0; i < cloud->points.size (); ++i)
    #   {
    #     cloud->points[i].x = 1024 * rand () / (RAND_MAX + 1.0f);
    #     cloud->points[i].y = 1024 * rand () / (RAND_MAX + 1.0f);
    #     cloud->points[i].z = 1.0;
    #   }
    #
    #   // Set a few outliers
    #   cloud->points[0].z = 2.0;
    #   cloud->points[3].z = -2.0;
    #   cloud->points[6].z = 4.0;
    ###
    cloud = pcl.PointCloud()

    # points = np.zeros((15, 3), dtype=np.float32)
    # RAND_MAX = 1024.0
    # for i in range(0, 15):
    #     points[i][0] = 1024 * random.random() / (RAND_MAX + 1.0)
    #     points[i][1] = 1024 * random.random() / (RAND_MAX + 1.0)
    #     points[i][2] = 1.0
    #
    # points[0][2] = 2.0
    # points[3][2] = -2.0
    # points[6][2] = 4.0

    # c_val = 255 << 16 | 255 << 8 | 255
    # color_col = np.array([c_val] * D_rw.shape[0], dtype=np.float32)
    # D_rw = np.concatenate([D_rw, color_col[:,np.newaxis]], axis=1)

    cloud.from_array(D_rw)

    # #   std::cerr << "Point cloud data: " << cloud->points.size () << " points" << std::endl;
    # #   for (size_t i = 0; i < cloud->points.size (); ++i)
    # #     std::cerr << "    " << cloud->points[i].x << " "
    # #                         << cloud->points[i].y << " "
    # #                         << cloud->points[i].z << std::endl;
    # #
    # print('Point cloud data: ' + str(cloud.size) + ' points')
    # for i in range(0, cloud.size):
    #     print('x: ' + str(cloud[i][0]) + ', y : ' +
    #           str(cloud[i][1]) + ', z : ' + str(cloud[i][2]))
    #
    # #   pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    # #   pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    # #   // Create the segmentation object
    # #   pcl::SACSegmentation<pcl::PointXYZ> seg;
    # #   // Optional
    # #   seg.setOptimizeCoefficients (true);
    # #   // Mandatory
    # #   seg.setModelType (pcl::SACMODEL_PLANE);
    # #   seg.setMethodType (pcl::SAC_RANSAC);
    # #   seg.setDistanceThreshold (0.01);
    # #
    # #   seg.setInputCloud (cloud);
    # #   seg.segment (*inliers, *coefficients);
    # ###
    # # http://www.pcl-users.org/pcl-SACMODEL-CYLINDER-is-not-working-td4037530.html
    # # NG?
    # # seg = cloud.make_segmenter()
    # # seg.set_optimize_coefficients(True)
    # # seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
    # # seg.set_method_type(pcl.SAC_RANSAC)
    # # seg.set_distance_threshold(0.01)
    # # indices, coefficients = seg.segment()
    seg = cloud.make_segmenter()
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_PERPENDICULAR_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_axis(0., 1.0, 1.0)
    seg.set_eps_angle(math.pi/4)
    seg.set_distance_threshold(0.1)
    # seg.set_normal_distance_weight(0.01)
    # seg.set_max_iterations(100)
    indices, coefficients = seg.segment()
    #
    # #   if (inliers->indices.size () == 0)
    # #   {
    # #     PCL_ERROR ("Could not estimate a planar model for the given dataset.");
    # #     return (-1);
    # #   }
    # #   std::cerr << "Model coefficients: " << coefficients->values[0] << " "
    # #                                       << coefficients->values[1] << " "
    # #                                       << coefficients->values[2] << " "
    # #                                       << coefficients->values[3] << std::endl;
    # ###
    # if len(indices) == 0:
    #     print('Could not estimate a planar model for the given dataset.')
    #     exit(0)
    #
    print('Model coefficients: ' + str(coefficients[0]) + ' ' + str(
        coefficients[1]) + ' ' + str(coefficients[2]) + ' ' + str(coefficients[3]))
    #
    # #   std::cerr << "Model inliers: " << inliers->indices.size () << std::endl;
    # #   for (size_t i = 0; i < inliers->indices.size (); ++i)
    # #     std::cerr << inliers->indices[i] << "    " << cloud->points[inliers->indices[i]].x << " "
    # #                                                << cloud->points[inliers->indices[i]].y << " "
    # #                                                << cloud->points[inliers->indices[i]].z << std::endl;
    # ###
    # print('Model inliers: ' + str(len(indices)))
    # max_dist = 0
    # for i in range(0, len(indices)):
    #     # print(str(indices[i]) + ', x: ' + str(cloud[indices[i]][0]) + ', y : ' +
    #     #       str(cloud[indices[i]][1]) + ', z : ' + str(cloud[indices[i]][2]))
    #     dist = np.abs(np.dot(D_rw[indices[i]][:3], coefficients[:3]) + coefficients[-1])
    #     if dist > max_dist:
    #         max_dist = dist
    #     # D_rw[indices[i]][3] = 255 << 16 | 0 << 8 | 255

    max_dist = 0
    dist = np.abs(np.dot(D_rw[:,:3], coefficients[:3]) + coefficients[-1])
    # for i in range(0, cloud.size):
    #     if dist > max_dist:
    #         max_dist = dist
    max_dist = np.max(dist)

    # dist = np.abs(np.dot(D_rw[:,:3], coefficients[:3]) + coefficients[-1])
    for i in range(0, cloud.size):
        D_rw[i][3] = 255 << 16 | int((dist[i]/max_dist)*255) << 8 | int(255-(dist[i]/max_dist)*255)

    for i in range(0, len(indices)):
        # print(str(indices[i]) + ', x: ' + str(cloud[indices[i]][0]) + ', y : ' +
        #       str(cloud[indices[i]][1]) + ', z : ' + str(cloud[indices[i]][2]))
        D_rw[indices[i]][3] = 255 << 16 | 0 << 8 | 0

    seg_cloud = pcl.PointCloud_PointXYZRGB()
    seg_cloud.from_array(D_rw)

    visual = pcl.pcl_visualization.CloudViewing()
    visual.ShowColorCloud(seg_cloud)

    while True:
        if visual.WasStopped():
            break

if __name__ == "__main__":
    # import cProfile
    # cProfile.run('main()', sort='time')
    main()
