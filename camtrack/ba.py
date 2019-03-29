from typing import List

import numpy as np
from recordclass import recordclass
import sortednp as snp
from scipy.optimize import approx_fprime
from scipy.sparse import csr_matrix

from corners import FrameCorners
from _camtrack import *


Corners = recordclass('Corners', ('frame', 'point3d', 'point2d'))


def run_bundle_adjustment(intrinsic_mat: np.ndarray,
                          list_of_corners: List[FrameCorners],
                          max_inlier_reprojection_error: float,
                          views: List[np.ndarray],
                          pc_builder: PointCloudBuilder) -> List[np.ndarray]:
    print(f'Running bundle adjustment with max_error={max_inlier_reprojection_error}')
    inliers = []
    used_3ds = set()
    for frame, (corners, view) in enumerate(zip(list_of_corners, views)):
        _, (ids_3d, ids_2d) = snp.intersect(
            pc_builder.ids.flatten(),
            corners.ids.flatten(),
            indices=True
        )
        indices = calc_inlier_indices(
            pc_builder.points[ids_3d],
            corners.points[ids_2d],
            intrinsic_mat @ view,
            max_inlier_reprojection_error
        )

        points = corners.points[ids_2d]
        for id in indices:
            id_3d = pc_builder.ids[ids_3d[id], 0]
            inliers.append(Corners(frame, id_3d, points[id]))
            used_3ds.add(id_3d)

    if len(used_3ds) == 0:
        return views

    used_3ds = list(sorted(used_3ds))
    for i, (_, point3d, _) in enumerate(inliers):
        inliers[i].point3d = used_3ds.index(point3d)
    p = _optimize_parameters(views, used_3ds, pc_builder, intrinsic_mat, inliers)

    res = []
    for id in range(len(views)):
        pid = 6 * id
        r_vec = p[pid : pid + 3].reshape(3, 1)
        t_vec = p[pid + 3 : pid + 6].reshape(3, 1)
        res.append(rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec))
    pc_builder.update_points(np.array(used_3ds), p[6 * len(views):].reshape(-1, 3))
    return res


def _reprojection_err(p, point2d, intrinsic_mat):
    view_mat = rodrigues_and_translation_to_view_mat3x4(p[0:3].reshape(3, 1), p[3:6].reshape(3, 1))
    proj_mat = np.dot(intrinsic_mat, view_mat)
    point3d_hom = np.hstack((p[6:9], 1))
    proj_point2d = np.dot(proj_mat, point3d_hom)
    proj_point2d = proj_point2d / proj_point2d[2]
    proj_point2d = proj_point2d.T[:2]
    err = (point2d - proj_point2d).reshape(-1)
    return np.linalg.norm(err)


def _reprojection_errs(p, inliers, views_num, intrinsic_mat):
    errors = []
    for i, (frame, point3d, point2d) in enumerate(inliers):
        p1 = np.hstack((
            p[6 * frame : 6 * frame + 6],
            p[views_num + 3 * point3d : views_num + 3 * point3d + 3]
        ))
        errors.append(_reprojection_err(p1[:9], point2d, intrinsic_mat))
    return np.array(errors)


def compute_jacobian(p, views_num, used_num, inliers, intrinsic_mat):
    vals = []
    rows = []
    cols = []
    for i, (frame, point3d, point2d) in enumerate(inliers):
        frame *= 6
        pid = views_num + 3 * point3d
        tmp = np.hstack((p[frame : frame + 6], p[pid:pid+3]))
        prime = approx_fprime(tmp, lambda pp: _reprojection_err(pp[:9], point2d, intrinsic_mat), np.full(tmp.size, 1e-10))
        for j in range(6):
            rows.append(i)
            cols.append(frame + j)
            vals.append(prime[j])
        for j in range(3):
            rows.append(i)
            cols.append(pid + j)
            vals.append(prime[j + 6])
    return csr_matrix((vals, (rows, cols)), shape=(len(inliers), views_num + 3 * used_num))

def _count_p(views, used_3ds, pc_builder):
    views_num = 6 * len(views)
    p = np.zeros(views_num + 3 * len(used_3ds))
    for frame_num in range(len(views)):
        p_id = 6 * frame_num
        r_vec, t_vec = view_mat3x4_to_rodrigues_and_translation(views[frame_num])
        p[p_id:p_id + 3] = r_vec.reshape(-1)
        p[p_id + 3:p_id + 6] = t_vec.reshape(-1)
    _, (indices, _) = snp.intersect(pc_builder.ids.flatten(), np.array(used_3ds), indices=True)
    p[views_num:] = pc_builder.points[indices].reshape(-1)
    return p


def _optimize_parameters(views, used_3ds, pc_builder, intrinsic_mat, inliers, max_steps=15):
    views_num = 6 * len(views)
    p = _count_p(views, used_3ds, pc_builder)

    print('Optimization started')
    J = compute_jacobian(p, views_num, len(used_3ds), inliers, intrinsic_mat)
    lmbd = 1.
    for step in range(max_steps):
        print("Step", step)
        errors = _reprojection_errs(p, inliers, views_num, intrinsic_mat)
        start_err = errors @ errors

        jmat = J.T.dot(J).toarray()
        jmat +=lmbd * np.diag(np.diagonal(jmat))
        U = jmat[:views_num, :views_num]
        W = jmat[:views_num, views_num:]
        V = jmat[views_num:, views_num:]
        V = np.linalg.inv(V)

        g = J.toarray().T.dot(_reprojection_errs(p, inliers, views_num, intrinsic_mat))
        A = U - W.dot(V).dot(W.T)
        B = W.dot(V).dot(g[views_num:]) - g[:views_num]

        try:
            delta_c = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            lmbd *= 5.
            continue

        delta_x = V.dot(-g[views_num:] - W.T.dot(delta_c))
        tmp = p.copy() + np.hstack((delta_c, delta_x))
        errors = _reprojection_errs(tmp, inliers, views_num, intrinsic_mat)
        err = errors @ errors
        if err < start_err:
            p[:] = tmp
            J = compute_jacobian(p, views_num, len(used_3ds), inliers, intrinsic_mat)
            lmbd /= 5.
        else:
            lmbd *= 5.
    return p