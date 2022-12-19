import open3d as o3d
import numpy as np
import copy
from kitti360scripts.viewer.kitti360Viewer3DRaw import Kitti360Viewer3DRaw
from PIL import Image
import matplotlib.pyplot as plt
from kitti360scripts.devkits.commons.loadCalibration import loadCalibrationCameraToPose, loadCalibrationRigid
from kitti360scripts.helpers.project import CameraPerspective, CameraFisheye
import os
from SemanticSegmentation import get_parser, SegmentationNetwork, check

import os
# import logging
# import argparse

import cv2
# import torch
# import torch.backends.cudnn as cudnn
# import torch.nn.functional as F
# import torch.nn.parallel
# import torch.utils.data
#
# from semseg.util import config
# from semseg.util.util import colorize
# from semseg.model.pspnet import PSPNet
# from semseg.model.psanet import PSANet

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


class projectVeloToImage:
    # Constructor
    def __init__(self, cam_id=0, seq=0, kitti360Path=""):
        kitti360Path = kitti360Path
        self.sequence = '2013_05_28_drive_%04d_sync'%seq
        self.cam_id = cam_id
        self.seq = seq
        # perspective camera
        self.camera = CameraPerspective(kitti360Path, self.sequence, cam_id)

        # object for parsing 3d raw data
        self.velo = Kitti360Viewer3DRaw(mode='velodyne', seq=seq, kitti360Path=kitti360Path)

        # cam_0 to velo
        fileCameraToVelo = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_velo.txt')
        TrCam0ToVelo = loadCalibrationRigid(fileCameraToVelo)

        # all cameras to system center
        fileCameraToPose = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_pose.txt')
        TrCamToPose = loadCalibrationCameraToPose(fileCameraToPose)

        # velodyne to all cameras
        TrVeloToCam = {}
        for k, v in TrCamToPose.items():
            # Tr(cam_k -> velo) = Tr(cam_k -> cam_0) @ Tr(cam_0 -> velo)
            TrCamkToCam0 = np.linalg.inv(TrCamToPose['image_00']) @ TrCamToPose[k]
            TrCamToVelo = TrCam0ToVelo @ TrCamkToCam0
            # Tr(velo -> cam_k)
            TrVeloToCam[k] = np.linalg.inv(TrCamToVelo)

        # take the rectification into account for perspective cameras
        self.TrVeloToRect = np.matmul(self.camera.R_rect, TrVeloToCam['image_%02d' % cam_id])

        args = get_parser()
        check(args)
        self.semantic_net = SegmentationNetwork(args)

    def cam2image(self, points):
        ndim = points.ndim
        if ndim == 2:
            points = np.expand_dims(points, 0)
        points_proj = np.matmul(self.camera.K[:3,:3].reshape([1,3,3]), points)
        depth = points_proj[:,2,:]
        depth[depth==0] = -1e-6
        u = np.round(points_proj[:,0,:]/np.abs(depth)).astype(np.int)
        v = np.round(points_proj[:,1,:]/np.abs(depth)).astype(np.int)

        if ndim==2:
            u = u[0]; v=v[0]; depth=depth[0]
        return u, v, depth

    def project (self, points):
        # visualize a set of frame
        # for each frame, load the raw 3D scan and project to image plane
        points = np.concatenate((points, np.ones((points.shape[0], 1))), axis = 1)
        # transfrom velodyne points to camera coordinate
        pointsCam = np.matmul(self.TrVeloToRect, points.T).T
        pointsCam = pointsCam[:,:3]
        # project to image space
        u,v, depth= self.camera.cam2image(pointsCam.T)
        u = u.astype(int)
        v = v.astype(int)

        return u, v, depth

    def read_RGB_img(self, frame):
        sub_dir = 'data_rect'

        # load RGB image for visualization
        imagePath = os.path.join(kitti360Path, 'data_2d_raw', self.sequence, 'image_%02d' % self.cam_id, sub_dir, '%010d.png' % frame)
        if not os.path.isfile(imagePath):
            raise RuntimeError('Image file %s does not exist!' % imagePath)

        colorImage = np.array(Image.open(imagePath))

        return colorImage

    def visualize(self, frame, u, v, depth, colorImage):
        # prepare depth map for visualization
        mask = np.logical_and(np.logical_and(np.logical_and(u>=0, u<self.camera.width), v>=0), v<self.camera.height)
        # visualize points within 30 meters
        mask = np.logical_and(np.logical_and(mask, depth>0), depth<30)

        semanticImage = self.semantic_net.test(colorImage)
        semanticImage = np.array(semanticImage.convert('RGB'))/ 255.
        colorImage = colorImage/ 255.

        return semanticImage[v[mask],u[mask]], mask

        depthMap = np.zeros((self.camera.height, self.camera.width))
        depthImage = np.zeros((self.camera.height, self.camera.width, 3))

        depthMap[v[mask],u[mask]] = depth[mask]

        depthImage = cm(depthMap/depthMap.max())[...,:3]
        colorImage[depthMap<= 0] = 0


        layout = (2,1)
        fig, axs = plt.subplots(*layout, figsize=(18,12))
        # color map for visualizing depth map
        cm = plt.get_cmap('jet')
        axs[0].imshow(depthMap, cmap='jet')
        axs[0].title.set_text('Projected Depth')
        axs[0].axis('off')
        axs[1].imshow(colorImage)
        axs[1].title.set_text('Projected Depth Overlaid on Image')
        axs[1].axis('off')
        plt.suptitle('Sequence %04d, Camera %02d, Frame %010d' % (self.seq, self.cam_id, frame))
        plt.show()

def pairwise_registration(source, target, initial_guess):
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, initial_guess,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, max_correspondence_distance_fine, transformation_icp)

    return transformation_icp, evaluation

def merge_map(pcd_combined, pcd, transformation, voxel_size, frame_id):
    u, v, depth = projector.project(np.asarray(pcd.points))

    colorImage = projector.read_RGB_img(frame_id)
    color, mask = projector.visualize(frame_id, u, v, depth, colorImage)

    pcd_temp = o3d.geometry.PointCloud()
    pcd_temp.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[mask, :3])
    pcd_temp.colors = o3d.utility.Vector3dVector(color)
    pcd_temp.transform(transformation)
    pcd_combined += pcd_temp
    pcd_combined = pcd_combined.voxel_down_sample(voxel_size=voxel_size)

    cv2.imshow('image', cv2.cvtColor(colorImage, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)
    return pcd_combined

def GetColour(v,vmin,vmax):
   r, g, b = 1, 1, 1
   if v < vmin:
      v = vmin

   if v > vmax:
      v = vmax
   dv = vmax - vmin

   if v < (vmin + 0.25 * dv):
      r = 0
      g = 4 * (v - vmin) / dv
   elif v < (vmin + 0.5 * dv):
      r = 0
      b = 1 + 4 * (vmin + 0.25 * dv - v) / dv
   elif v < (vmin + 0.75 * dv):
      r = 4 * (v - vmin - 0.5 * dv) / dv
      b = 0
   else:
      g = 1 + 4 * (vmin + 0.75 * dv - v) / dv
      b = 0

   return [r, g, b]


if __name__ == "__main__":
    mode = 'velodyne'
    voxel_size = 0.02
    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5

    data_set_path = "/media/bear/T7/KITTI-360/"

    kitti360Path="/media/bear/T7/KITTI-360/"
    ve = Kitti360Viewer3DRaw(seq=0, mode=mode, kitti360Path=kitti360Path)


    T = np.eye(4)
    points_curr = ve.loadVelodyneData(0)
    # depth = points_curr[:, 2]

    pcd_curr = o3d.geometry.PointCloud()
    pcd_curr.points = o3d.utility.Vector3dVector(points_curr[:, :3])
    pcd_curr = pcd_curr.voxel_down_sample(voxel_size=voxel_size)
    pcd_curr.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # 0000011517.bin
    odometry = np.eye(4)
    projector = projectVeloToImage(cam_id=0, seq=0, kitti360Path=kitti360Path)
    pcd_combined = o3d.geometry.PointCloud()
    pcd_combined = merge_map(pcd_combined, pcd_curr, np.linalg.inv(odometry), voxel_size, 0)
    transformation_icp = np.identity(4)
    rmse = 0
    fitness = 1

    min_bound = np.asarray([-30, -30, -5])
    max_bound = np.asarray([30, 30, 5])

    t = np.linalg.inv(odometry)[0:3, 3]


    # min_bound_curr = np.linalg.inv(odometry) @ min_bound
    # max_bound_curr = np.linalg.inv(odometry) @ max_bound

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=tuple(t+ min_bound),
                                               max_bound=tuple(t+ max_bound))

    pcb_cropped = pcd_combined.crop(bbox)
    pcb_cropped.transform(odometry)

    # pcb_cropped = pcd_combined

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcb_cropped)
    # ctr.set_lookat([0, 0, -1])
    # vis.run()

    for frame_id in range(1, 1000, 7):
        pcd_prev = copy.deepcopy(pcd_curr)

        points_curr = ve.loadVelodyneData(frame_id)
        pcd_curr = o3d.geometry.PointCloud()
        # depth = points_curr[:, 2]
        pcd_curr.points = o3d.utility.Vector3dVector((points_curr[:, :3]))
        pcd_curr = pcd_curr.voxel_down_sample(voxel_size=voxel_size)
        pcd_curr.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        transformation_icp, evaluation = pairwise_registration(pcd_prev, pcd_curr, transformation_icp)
        if evaluation.fitness < fitness:
            fitness = evaluation.fitness

        if evaluation.inlier_rmse > rmse:
            rmse = evaluation.inlier_rmse
        odometry = transformation_icp @ odometry
        vis.remove_geometry(pcb_cropped)
        pcd_combined = merge_map(pcd_combined, pcd_curr, np.linalg.inv(odometry), voxel_size, frame_id)

        # min_bound_curr = np.linalg.inv(odometry) @ min_bound
        # max_bound_curr = np.linalg.inv(odometry) @ max_bound

        # print(odometry)
        # print(min_bound_curr)
        # print(max_bound_curr)

        t = np.linalg.inv(odometry)[0:3, 3]

        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=tuple(t + min_bound),
                                                   max_bound=tuple(t + max_bound))

        pcb_cropped = pcd_combined.crop(bbox)
        pcb_cropped.transform(odometry)

        vis.add_geometry(pcb_cropped)
        vis.poll_events()
        vis.update_renderer()
    vis.destroy_window()
    cv2.destroyWindow("image")
    print(rmse, fitness)
    # combined_points = np.asarray(pcd_combined.points)
    print(pcd_combined.get_max_bound())
    print(pcd_combined.get_min_bound())
    # color = np.zeros_like(combined_points)
    # vmax = pcd_combined.get_max_bound()[2]
    # vmin = pcd_combined.get_min_bound()[2]
    #
    # for i in range(combined_points.shape[0]):
    #     depth = combined_points[i, 2]
    #     color[i, :] = GetColour(depth, vmin, vmax)
    #
    # pcd_combined.colors = o3d.utility.Vector3dVector(color)
    # o3d.visualization.draw_geometries([pcd_combined])


    #o3d.visualization.draw_geometries([pcd])
    # projector = projectVeloToImage(cam_id=0, seq=0, kitti360Path=kitti360Path)
    #
    # u,v, depth = projector.project(0)
    # projector.visualize(0, u, v, depth)
    #


