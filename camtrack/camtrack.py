#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Tuple

import numpy as np
import cv2
import sortednp as snp

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import *


max_reprojection_error=1.
min_depth=0.1


def _initialize_with_history(corners_prev, corners, intrinsic_mat, triang_pars):
    correspondences = build_correspondences(corners_prev, corners)

    if len(correspondences.ids) <= 5:
        return None, np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    E, mask = cv2.findEssentialMat(correspondences.points_1, correspondences.points_2,
                                   cameraMatrix=intrinsic_mat, method=cv2.RANSAC, prob=0.99, threshold=1.)
    H, hmask = cv2.findHomography(correspondences.points_1, correspondences.points_2,
                                   method=cv2.RANSAC, ransacReprojThreshold=1., confidence=0.99)
    if np.sum(mask) / np.sum(hmask) < 0.8:
        return None, np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    r1, r2, t = cv2.decomposeEssentialMat(E)
    correspondences = remove_correspondences_with_ids(correspondences, np.where(mask == 0)[0])
    view_mat, points, ids = None, np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    for r in [r1, r2]:
        for tt in [t.reshape(-1), -t.reshape(-1)]:
            view_ = pose_to_view_mat3x4(Pose(r, tt))
            pts_, ids_ = triangulate_correspondences(correspondences, eye3x4(), view_, intrinsic_mat, triang_pars)
            if pts_.size > points.size:
                view_mat, points, ids = view_, pts_, ids_

    return view_mat, points, ids


def _initialize_cloud(corner_storage, intrinsic_mat, triang_pars):
    res = (None, None, np.array([], dtype=np.int32), np.array([], dtype=np.int32))
    max_sz = -5
    for frame, corners in enumerate(corner_storage[1:], start=1):
        view, points, ids = _initialize_with_history(corner_storage[0], corners, intrinsic_mat, triang_pars)
        res_ = (frame, view, points, ids)
        if len(ids) > max_sz:
            max_sz = len(ids)
            res = res_
    builder = PointCloudBuilder()
    builder.add_points(res[3], res[2])
    return res[0], res[1], builder


def _track_camera(corner_storage: CornerStorage,
                  intrinsic_mat: np.ndarray, min_triangulation_angle_deg=0.7) \
        -> Tuple[List[np.ndarray], PointCloudBuilder]:

    triang_pars = TriangulationParameters(
        max_reprojection_error=max_reprojection_error,
        min_triangulation_angle_deg=min_triangulation_angle_deg,
        min_depth=min_depth)
    views = [eye3x4()]
    init_frame, init_view, builder = _initialize_cloud(corner_storage, intrinsic_mat, triang_pars)

    print('Min triang angle: {}'.format(min_triangulation_angle_deg))

    for frame, corner in enumerate(corner_storage[1:], start=1):
        if frame == init_frame:
            views.append(init_view)
            continue

        _, (obj_ids, imgs_ids) = snp.intersect(
            builder.ids.flatten(),
            corner.ids.flatten(),
            indices=True
        )

        if len(builder.points[obj_ids]) < 4:
            print("Tracking failed with len < 4!")
            if min_triangulation_angle_deg <= 0.3:
                return [], PointCloudBuilder()
            return _track_camera(corner_storage, intrinsic_mat, min_triangulation_angle_deg - 0.1)

        obj = builder.points[obj_ids]
        imgs = corner.points[imgs_ids]
        res, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj.reshape((obj.shape[0], 1, obj.shape[1])), imgs.reshape((imgs.shape[0], 1, imgs.shape[1])),
            intrinsic_mat, distCoeffs=None,
        )

        if not res:
            print("Tracking failed with no res!")
            if min_triangulation_angle_deg <= 0.3:
                return [], PointCloudBuilder()
            return _track_camera(corner_storage, intrinsic_mat, min_triangulation_angle_deg - 0.1)

        res_ids = np.delete(builder.ids[obj_ids], inliers, axis=0)
        views.append(rodrigues_and_translation_to_view_mat3x4(rvec, tvec))
        delta = 0
        for other_frame in range(frame):
            correspondences = build_correspondences(corner_storage[other_frame], corner, ids_to_remove=res_ids)
            if len(correspondences.ids) == 0:
                continue
            new_points, new_ids = triangulate_correspondences(correspondences,
                                                              views[other_frame], views[frame],
                                                              intrinsic_mat, triang_pars
                                                              )

            delta += len(new_ids)
            builder.add_points(new_ids, new_points)

        if frame % 10 == 0:
            print('Frame {}: new points {}, total {}'.format(frame, delta, builder.points.shape[0]))

    return views, builder


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    view_mats, point_cloud_builder = _track_camera(
        corner_storage,
        intrinsic_mat
    )
    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    create_cli(track_and_calc_colors)()