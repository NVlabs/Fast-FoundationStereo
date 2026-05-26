'''

Dataset management for Synthetic stereo dataset.

Loads RealSense IR stereo pairs left, right images
and Depth depthmap and computes the ground-truth from chess baord pattern in the left image.
The png file have 3 channels: first chnnel is left, second is right, third is depth.

The left channel contains images of the chess board pattern captured by the left RealSense camera.
, which is used for computing the synthetic depth ground truth. 
The right channel contains the corresponding images from the right RealSense camera. 
The depth channel contains the depth maps obtained from the RealSense sensor, 
which can be used for comparison against the synthetic depth computed from the chessboard pattern.

Expected directory layout (one or more session folders under root):
    <root>/
      <session>/
        405/
          <type_1>/
            image_d16_<idx>.png          # left, right and depth image  (uint16)
            image_d16_<idx>.png          # left, right and depth image  (uint16)
          <type_2>/
            image_d16_<idx>.png          # left, right and depth image  (uint16)            


Only samples that have BOTH a realsense pair AND a matching zivid depth are
included. Sessions that lack a zivid subfolder (e.g. freedrive-only captures)
are silently skipped.

Output dict keys (same as faro_data_manager for compatibility):
    left        : numpy array  (H, W)   uint8/uint16 IR
    right       : numpy array  (H, W)   uint8/uint16 IR
    depth_syn   : numpy array  (H, W)   float32, mm  ← Synthetic depth from chessboard pattern (empty / zeros if absent)
    depth_rs    : numpy array  (H, W)   float32, mm  (empty / zeros if absent)

'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob
import unittest
import logging as log
import yaml

from object_chessboard import ObjectChessboard

# --------------------------------
# 405 / 1280x720
CAMERA_MATRIX_RS = np.array([
    [644.471, 0, 649.253],
    [0, 644.471, 365.398],
    [0, 0, 1]
])
DIST_COEFFS_RS = np.array([ 0.0,  -0.0,     -0.0,     0.0,    -0.0])



# --------------------------------
#%% Data source
class DataSource:

    def __init__(self):
        self.gray_scale_input   = False
        self.depth_estimator    = ObjectChessboard()  # for synthetic GT depth estimation from chessboard pattern
        self.imgs               = []   # list of dicts: {packed_png}
        log.info('Source is defined')

    def init_directory(self, input_rectified='', gray_scale_input=False, sub_indexes=None):
        """Scan root for packed synthetic PNG files and populate self.imgs.

        Expected layout (one or more sessions):
            <root>/<session>/405/<type>/image_d16_<idx>.png
        """
        if len(input_rectified) < 3:
            input_rectified = r'C:\Work\Data\DepthRS\ffs'

        self.gray_scale_input = gray_scale_input
        self.imgs = []

        if not os.path.isdir(input_rectified):
            log.error(f"Directory not found: {input_rectified}")
            return 0

        patterns = [
            os.path.join(input_rectified, '**', '405', '*', 'image_d16_*.png'),  # recursive, supports <root>/405/... and <root>/<session>/405/...   # legacy/session path
        ]

        packed_paths = []
        for pattern in patterns:
            packed_paths.extend(glob.glob(pattern, recursive=True))
        packed_paths = sorted(set(packed_paths))

        for packed_path in packed_paths:
            self.imgs.append({'packed_png': packed_path})

        if sub_indexes is not None:
            self.imgs = [self.imgs[i] for i in sub_indexes]

        log.info(f"DataSource: found {len(self.imgs)} samples in {input_rectified}")
        return len(self.imgs)

    def get_item(self, index: int, debug: bool = False):
        """Return one sample as a dict with left, right, depth_syn, depth_rs."""
        output_str = {
            "left": [],
            "right": [],
            "depth_syn": [],
            "depth_rs": [],
            "rgb": np.array([], dtype=np.uint8)
        }

        entry = self.imgs[index]

        packed_img = cv2.imread(entry['packed_png'], cv2.IMREAD_UNCHANGED)
        if packed_img is None:
            log.warning(f"Failed to load sample {index}: {entry['packed_png']}")
            return output_str

        if packed_img.ndim != 3 or packed_img.shape[2] < 3:
            log.warning(f"Invalid packed PNG format (expected 3 channels): {entry['packed_png']}")
            return output_str

        left_img  = packed_img[:, :, 0]
        right_img = packed_img[:, :, 1]
        depth_rs  = packed_img[:, :, 2].astype(np.float32)

        # Synthetic GT can be computed from chessboard if available;
        depth_syn                 = self.get_synthetic_depth(left_img)

        output_str["left"]        = left_img
        output_str["right"]       = right_img
        output_str["depth_syn"]   = depth_syn
        output_str["depth_rs"]    = depth_rs

        if debug:
            img_list = [left_img, right_img, depth_rs, depth_syn]
            ttl_list = ['left', 'right', 'depth RS (mm)', 'depth SYN (mm)']
            self.show_subset(img_list, ttl_list)

        return output_str
    
    def get_synthetic_depth(self, left_img):
        """Compute synthetic depth from chessboard pattern in the left image."""
        result = self.depth_estimator.estimate_camera_pose(left_img, camera_matrix = CAMERA_MATRIX_RS, dist_coeffs = DIST_COEFFS_RS)
        if result["success"]:
            XYZ, projected_points = self.depth_estimator.get_grid_in_camera_coordinates(
                rvec=result['rvec'],
                tvec=result['tvec'],
                camera_matrix=CAMERA_MATRIX_RS,
                dist_coeffs=DIST_COEFFS_RS
            )
            depth_syn = self.project_3d_to_camera(XYZ, CAMERA_MATRIX_RS, DIST_COEFFS_RS, frame_size = left_img.shape)  # Project back to image space to get depth map
            return depth_syn
        else:
            log.warning("Failed to estimate camera pose for synthetic depth computation.")
            return np.zeros_like(left_img, dtype=np.float32)

    def get_item_projected(self, index: int, debug: bool = False):
        """Compatibility wrapper for synthetic data; returns the same as get_item."""
        return self.get_item(index=index, debug=debug)

    def compute_depth_error(self, depth_pred, depth_gt, depth_mask=None):
        """Compute absolute depth error between prediction and GT."""
        depth_pred = depth_pred.astype(np.float32)
        depth_gt   = depth_gt.astype(np.float32)
        depth_error = np.zeros_like(depth_pred)
        mask = np.ones_like(depth_pred, dtype=bool) if depth_mask is None else depth_mask
        valid = np.logical_and(depth_gt > 0, mask)
        valid = np.logical_and(depth_pred > 0, valid)
        depth_error[valid] = (depth_pred[valid] - depth_gt[valid])
        return depth_error

    def show_subset(self, img_list, ttl_list, vmin=None, vmax=None, save_path='', fig_name=''):
        """Display a list of images in a grid."""
        img_num = len(img_list)
        col_num = min(img_num, 3)
        row_num = (img_num + col_num - 1) // col_num
        fig, axes = plt.subplots(row_num, col_num, sharey=True, sharex=True)
        axes = np.array(axes).reshape(row_num, col_num)
        for k in range(img_num):
            ri, ci = k // col_num, k % col_num
            axes[ri, ci].imshow(img_list[k], vmin=vmin, vmax=vmax)
            axes[ri, ci].set_title(ttl_list[k])
        for k in range(img_num, row_num * col_num):
            axes[k // col_num, k % col_num].axis('off')
        if save_path and os.path.exists(save_path):
            fig.savefig(os.path.join(save_path, fig_name + ".png"))
        plt.show(block=False)

    def save_data_to_folder(self, output_str, output_directory):
        """Save sample dict to PNG files on disk."""
        os.makedirs(output_directory, exist_ok=True)

        paths = {
            "img_left.png":        output_str["left"],
            "img_right.png":       output_str["right"],
            "img_depth_syn.png":   output_str["depth_syn"].astype(np.uint16),
            "img_depth_rs.png":    output_str["depth_rs"].astype(np.uint16),
        }
        success = True
        for fname, img in paths.items():
            out = cv2.imwrite(os.path.join(output_directory, fname), img,
                              [cv2.IMWRITE_PNG_COMPRESSION, 0])
            success = success and out

        if output_str["rgb"] is not None and np.asarray(output_str["rgb"]).size > 0:
            cv2.imwrite(os.path.join(output_directory, "img_rgb.png"),
                        output_str["rgb"], [cv2.IMWRITE_PNG_COMPRESSION, 0])

        return success
    

    def save_to_ply(self, points: np.ndarray, filename: str):
        """Save a point cloud to a PLY file for visualization."""
        with open(filename, 'w') as f:
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {len(points)}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('end_header\n')
            for x, y, z in points:
                f.write(f'{x} {y} {z}\n')

    def project_camera_to_3d(self, depth_img_mm: np.ndarray, cam_matrix: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
        """Project 2D pixel coordinates with depth to 3D points in camera space."""
        h, w = depth_img_mm.shape
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32), indexing='xy')

        # OpenCV expects Nx1x2 contiguous float32/float64 image points in (x, y) order.
        distorted_points = np.stack([xs, ys], axis=-1).reshape(-1, 1, 2).astype(np.float32)
        undistorted_points = cv2.undistortPoints(distorted_points,  cam_matrix.astype(np.float32),  dist_coeffs.astype(np.float32) )

        uv = undistorted_points.reshape(-1, 2)
        Z = depth_img_mm.reshape(-1).astype(np.float32)
        valid = np.isfinite(Z) & (Z > 0)
        if not np.any(valid):
            return np.zeros((0, 3), dtype=np.float32)

        uv      = uv[valid]
        Z       = Z[valid]
        X       = uv[:, 0] * Z
        Y       = uv[:, 1] * Z

        # save to ply point cloud for visualization
        XYZ     = np.stack([X, Y, Z], axis=1).astype(np.float32)

        return XYZ

    def project_3d_to_camera(self, points_3d: np.ndarray, cam_matrix: np.ndarray, dist_coeffs: np.ndarray, frame_size = (480,640)) -> np.ndarray:
        """Project 3D points in camera space back to 2D pixel coordinates."""
        if points_3d.shape[1] != 3:
            raise ValueError("Input points_3d must have shape (N, 3)")
        
        projected_pts, _ = cv2.projectPoints(
            points_3d.reshape(-1, 1, 3),
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            cam_matrix.astype(np.float32),
            dist_coeffs.astype(np.float32),
        )

        uv_rs = projected_pts.reshape(-1, 2)
        u_idx = np.rint(uv_rs[:, 0]).astype(np.int32)
        v_idx = np.rint(uv_rs[:, 1]).astype(np.int32)

        h_rs, w_rs = frame_size
        in_bounds = (u_idx >= 0) & (u_idx < w_rs) & (v_idx >= 0) & (v_idx < h_rs)
        if not np.any(in_bounds):
            return np.zeros((h_rs, w_rs), dtype=np.float32)

        u_idx = u_idx[in_bounds]
        v_idx = v_idx[in_bounds]
        z_vals = points_3d[in_bounds, 2]  # Z values of the valid points

        # Rasterize by nearest pixel; if multiple points hit a pixel, keep the closest depth.
        lin             = v_idx * w_rs + u_idx
        depth_buffer    = np.full(h_rs * w_rs, np.inf, dtype=np.float32)
        np.minimum.at(depth_buffer, lin, z_vals)
        depth_projected = depth_buffer.reshape(h_rs, w_rs)
        depth_projected[~np.isfinite(depth_projected)] = 0.0
        return depth_projected

    # project from zivid depth patrix to point cloud and back to depth matrix with rs intrinsics and distortion to get "zivid GT as seen by RealSense" for pixel-level comparison
    def project_depth_zivid_to_rs(self,depth_zivid_mm: np.ndarray, depth_rs_mm: np.ndarray, finx = 0) -> np.ndarray:
        # create 3D point cloud from zivid depth
        XYZ = self.project_camera_to_3d(depth_zivid_mm, CAMERA_MATRIX_RS, DIST_COEFFS_RS)  # (N, 3) array of 3D points in Zivid camera space
        # save to ply point cloud for visualization
        #save_to_ply(XYZ/1000, f'zivid_original_points_{finx:03d}.ply') # save in meters for visualization

        # project back on imaage RS
        depth_zivid_projected_mm = self.project_3d_to_camera(XYZ, CAMERA_MATRIX_RS, DIST_COEFFS_RS, frame_size = depth_rs_mm.shape)  # (H, W) depth map of Zivid points projected into RealSense pixel space

        XYZ_RS = self.project_camera_to_3d(depth_zivid_projected_mm, CAMERA_MATRIX_RS, DIST_COEFFS_RS)
            # save to ply point cloud for visualization
        #save_to_ply(XYZ_RS/1000, f'zivid_projected_points_{finx:03d}.ply') # save in meters for visualization

        return depth_zivid_projected_mm    
    
    def show_projection(self, rs_map, zv_map, zv_valid, idx):
        fig, axes = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(8,4))
        axes[0].imshow(rs_map, vmin=-10, vmax=1000),axes[0].set_title(f"RealSense Depth Diff (mm)"),
        axes[1].imshow(zv_map, vmin=-10, vmax=1000),axes[1].set_title(f"Zivid Projected Depth Diff (mm)"),
        axes[2].imshow(zv_valid, cmap='gray'),axes[2].set_title(f"Valid Mask (Zivid Projection)"),
        plt.suptitle(f"Sample {idx:03d} Depth Difference Maps and Valid Mask", fontsize=16)
        plt.tight_layout()
        plt.show()


# --------------------------------
#%% Tests
class TestDataSource(unittest.TestCase):

    def test_init_directory(self):
        p       = DataSource()
        img_num = p.init_directory()
        self.assertTrue(img_num > 0)

    def test_get_item(self):
        p       = DataSource()
        img_num = p.init_directory()
        self.assertTrue(img_num > 0)
        out = p.get_item(0, debug=True)
        self.assertTrue(len(out["left"]) > 0)

    def test_show_images(self):
        p       = DataSource()
        img_num = p.init_directory()
        if img_num == 0:
            log.warning("No images found.")
            return
        for k in np.random.randint(0, img_num, size=min(4, img_num)):
            out = p.get_item(int(k), debug=True)
            self.assertTrue(len(out["left"]) > 0)
            p.show_subset([out["left"], out["right"], out["depth_rs"]],
                          ['left (RS)', 'right (RS)', 'depth RS (mm)'])

        plt.show()

    def test_get_item_projected(self):
        p       = DataSource()
        img_num = p.init_directory()
        self.assertTrue(img_num > 0)
        out     = p.get_item_projected(80, debug=False)
        err     = p.compute_depth_error(out["depth_rs"], out["depth_syn"])
        self.assertTrue(len(out["left"]) > 0)
        p.show_subset([out["left"], out["right"], out["depth_rs"], out["depth_syn"], err],
                          ['left (RS)', 'right (RS)', 'depth RS (mm)', 'depth SYN (mm)', 'error (mm)'], vmax=None)
        plt.show()


# --------------------------------
#%% Run Test
def RunTest():
    tst = TestDataSource()
    #tst.test_init_directory()
    #tst.test_get_item()
    #tst.test_show_images()
    tst.test_get_item_projected()


if __name__ == '__main__':
    RunTest()
