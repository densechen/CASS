import math
import random

import numpy as np
import open3d as o3d
import trimesh


def save_obj(vertex: np.array, path: str):
    """ vertex: [N x 3]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertex)
    o3d.io.write_point_cloud(path, pcd)


def load_obj(path):
    """return np.array
    """
    pcd_load = o3d.io.read_point_cloud(path)
    return np.asarray(pcd_load.points)


def estimateSimilarityTranform(source: np.array, target: np.array):
    source_hom = np.transpose(
        np.hstack([source, np.ones([source.shape[0], 1])]))
    target_hom = np.transpose(
        np.hstack([target, np.ones([source.shape[0], 1])]))

    # auto-parameter selection based on source-target heuritics
    target_norm = np.mean(np.linalg.norm(target, axis=1))
    source_norm = np.mean(np.linalg.norm(source, axis=1))
    ratio_TS = (target_norm / source_norm)
    ratio_ST = (source_norm / target_norm)

    pass_T = ratio_ST if ratio_ST > ratio_TS else ratio_TS
    stop_T = pass_T / 100.
    n_iter = 100

    source_inliers_hom, target_inliers_hom, best_inlier_ratio = getRANSACInliers(
        source_hom, target_hom, max_iterations=n_iter, pass_threshold=pass_T, stop_threshold=stop_T)
    if best_inlier_ratio < 0.1:
        return None, None, None, None

    scales, rotation, translation, out_transform = estimateSimilarityUmeyama(
        source_inliers_hom, target_inliers_hom)

    return scales, rotation, translation, out_transform


def getRANSACInliers(source_hom, target_hom, max_iterations=100, pass_threshold=200, stop_threshold=1):
    best_residual = 1e10
    best_inlier_ratio = 0
    best_inlier_idx = np.arange(source_hom.shape[1])
    for _ in range(max_iterations):
        # pick up 5 random (but corresponding) points from source and target
        rand_idx = np.random.randint(source_hom.shape[1], size=5)
        _, _, _, out_transform = estimateSimilarityUmeyama(
            source_hom[:, rand_idx], target_hom[:, rand_idx])
        residual, inlier_ratio, inlier_idx = evaluateModel(
            out_transform, source_hom, target_hom, pass_threshold)
        if residual < best_residual:
            best_residual = residual
            best_inlier_ratio = inlier_ratio
            best_inlier_idx = inlier_idx
        if best_residual < stop_threshold:
            break
    return source_hom[:, best_inlier_idx], target_hom[:, best_inlier_idx], best_inlier_ratio


def evaluateModel(out_transform, source_hom, target_hom, pass_threshold):
    diff = target_hom - np.matmul(out_transform, source_hom)
    residual_vec = np.linalg.norm(diff[:3, :], axis=0)
    residual = np.linalg.norm(residual_vec)
    inlier_idx = np.where(residual_vec < pass_threshold)
    n_inliers = np.count_nonzero(inlier_idx)
    inliner_ratio = n_inliers / source_hom.shape[1]
    return residual, inliner_ratio, inlier_idx[0]


def estimateSimilarityUmeyama(source_hom, target_hom):
    source_centroid = np.mean(source_hom[:3, :], axis=1)
    target_centroid = np.mean(target_hom[:3, :], axis=1)
    n_points = source_hom.shape[1]

    centered_source = source_hom[:3, :] - \
        np.tile(source_centroid, (n_points, 1)).transpose()
    centered_target = target_hom[:3, :] - \
        np.tile(target_centroid, (n_points, 1)).transpose()

    cov_matrix = np.matmul(
        centered_target, np.transpose(centered_source)) / n_points

    if np.isnan(cov_matrix).any():
        raise RuntimeError("There are NaNs in the input.")

    U, D, Vh = np.linalg.svd(cov_matrix, full_matrices=True)
    d = (np.linalg.det(U) * np.linalg.det(Vh)) < 0.0
    if d:
        D[-1] = -D[-1]
        U[:, -1] = -U[:, -1]

    rotation = np.matmul(U, Vh).T

    var_p = np.var(source_hom[:3, :], axis=1).sum()
    scale_fact = 1 / var_p * np.sum(D)
    scales = np.array([scale_fact, scale_fact, scale_fact])
    scale_matrix = np.diag(scales)

    translation = target_hom[:3, :].mean(
        axis=1) - source_hom[:3, :].mean(axis=1).dot(scale_fact * rotation)

    out_transform = np.identity(4)
    out_transform[:3, :3] = scale_matrix @ rotation
    out_transform[:3, 3] = translation

    return scales, rotation, translation, out_transform


def backproject(depth, intr, mask):
    intr_inv = np.linalg.inv(intr)

    non_zero_mask = depth > 0
    final_instance_mask = np.logical_and(mask, non_zero_mask)

    idxs = np.where(final_instance_mask)
    grid = np.array([idxs[1], idxs[0]])

    length = grid.shape[1]
    ones = np.ones([1, length])
    uv_grid = np.concatenate([grid, ones], axis=0)

    xyz = intr_inv @ uv_grid
    xyz = np.transpose(xyz)

    z = depth[idxs[0], idxs[1]]

    pts = xyz * z[:, np.newaxis] / xyz[:, -1:]
    pts[:, 0] = -pts[:, 0]
    pts[:, 1] = -pts[:, 1]
    return pts, idxs


def triangle_area(v1, v2, v3):
    a = np.array(v2) - np.array(v1)
    b = np.array(v3) - np.array(v1)
    domain = np.dot(a, a) * np.dot(b, b) - (np.dot(a, b) ** 2)
    domain = domain if domain > 0 else 0.0

    return math.sqrt(domain) / 2.0


def cal_surface_area(mesh):
    areas = []
    if hasattr(mesh, "faces"):
        for face in mesh.faces:
            v1, v2, v3 = face
            v1 = mesh.vertices[v1]
            v2 = mesh.vertices[v2]
            v3 = mesh.vertices[v3]

            areas += [triangle_area(v1, v2, v3)]
    else:
        for face in mesh.triangles:
            v1, v2, v3 = face

            areas += [triangle_area(v1, v2, v3)]
    return np.array(areas)


def sample_obj(path, num_points, norm):
    """sample uniform point from .obj mesh file.
    if norm, we ill normalize it.
    """
    mesh = trimesh.load(path)
    areas = cal_surface_area(mesh)
    prefix_sum = np.cumsum(areas)

    total_area = prefix_sum[-1]
    sample_points = []

    for _ in range(num_points):
        prob = random.random()
        sample_pos = prob * total_area

        # binary search
        left_bound, right_bound = 0, len(areas) - 1
        while left_bound < right_bound:
            mid = (left_bound + right_bound) // 2
            if sample_pos <= prefix_sum[mid]:
                right_bound = mid
            else:
                left_bound = mid + 1

        target_surface = right_bound

        # sampel point
        if hasattr(mesh, "faces"):
            v1, v2, v3 = mesh.faces[target_surface]

            v1, v2, v3 = mesh.vertices[v1], mesh.vertices[v2], mesh.vertices[v3]
        else:
            v1, v2, v3 = mesh.triangles[target_surface]

        edge_vec1 = np.array(v2) - np.array(v1)
        edge_vec2 = np.array(v3) - np.array(v1)

        prob_vec1, prob_vec2 = random.random(), random.random()
        if prob_vec1 + prob_vec2 > 1:
            prob_vec1 = 1 - prob_vec1
            prob_vec2 = 1 - prob_vec2

        target_point = np.array(
            v1) + (edge_vec1 * prob_vec1 + edge_vec2 * prob_vec2)

        sample_points.append(target_point)
    sample_points = np.stack(sample_points, axis=0)

    if norm:
        min_ = np.min(sample_points, axis=0)
        max_ = np.max(sample_points, axis=0)
        dis_ = max_ - min_

        scale = 1 / np.sqrt(np.sum(dis_ * dis_))

        sample_points *= scale

    return sample_points
