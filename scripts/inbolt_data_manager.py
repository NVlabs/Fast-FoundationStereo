'''

Dataset management for Inbolt stereo dataset.

Loads RealSense IR stereo pairs (mono0/mono1) as left/right images
and Zivid depthmap as ground-truth depth.

Expected directory layout (one or more session folders under root):
    <root>/
      <session>/
        realsense/
          <idx>/
            mono0.png          # left IR image  (uint8 or uint16)
            mono1.png          # right IR image (uint8 or uint16)
        zivid/
          <idx>/
            depthmap_mm.png    # GT depth in mm (uint16)
            color.png          # optional RGB

Only samples that have BOTH a realsense pair AND a matching zivid depth are
included. Sessions that lack a zivid subfolder (e.g. freedrive-only captures)
are silently skipped.

Output dict keys (same as faro_data_manager for compatibility):
    left        : numpy array  (H, W)   uint8/uint16 IR
    right       : numpy array  (H, W)   uint8/uint16 IR
    depth_faro  : numpy array  (H, W)   float32, mm  ← Zivid GT
    depth_rs    : numpy array  (H, W)   float32, mm  (empty / zeros if absent)
    rgb         : numpy array  (H, W, 3) uint8        (Zivid color, or empty)

'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob
import unittest
import logging as log


# --------------------------------
#%% Data source
class DataSource:

    def __init__(self):
        self.gray_scale_input = False
        self.imgs = []   # list of dicts: {left, right, depth_zivid, rgb}
        log.info('Source is defined')

    def init_directory(self, input_rectified='', gray_scale_input=False, sub_indexes=None):
        """Scan root for (realsense, zivid) sample pairs and populate self.imgs."""
        if len(input_rectified) < 3:
            input_rectified = (
                r'/mnt/algonas/Local/Data/new_depth_stereo_datasets/'
                r'Inbolt_datasets/Data Collection-20260322T091926Z-1-001/Data Collection'
            )

        self.gray_scale_input = gray_scale_input
        self.imgs = []

        IGNORED_SESSIONS = {'dataset_y16_freedrive', 'dataset_y8_freedrive'}

        # Each immediate sub-directory is a session
        try:
            sessions = sorted([
                os.path.join(input_rectified, d)
                for d in os.listdir(input_rectified)
                if os.path.isdir(os.path.join(input_rectified, d))
                and d not in IGNORED_SESSIONS
            ])
        except FileNotFoundError:
            log.error(f"Directory not found: {input_rectified}")
            return 0

        for session in sessions:
            rs_root    = os.path.join(session, 'realsense')
            zivid_root = os.path.join(session, 'zivid')

            if not os.path.isdir(rs_root) or not os.path.isdir(zivid_root):
                continue  # session has no stereo+GT pair

            # Find all left images; match by index folder name
            left_paths = sorted(glob.glob(os.path.join(rs_root, '*', 'mono0.png')))
            for left_path in left_paths:
                idx             = os.path.basename(os.path.dirname(left_path))
                right_path      = os.path.join(rs_root, idx, 'mono1.png')
                depth_rs_path    = os.path.join(rs_root, idx, 'depthmap_mm.png')
                depth_zivid_path = os.path.join(zivid_root, idx, 'depthmap_mm.png')
                rgb_path        = os.path.join(zivid_root, idx, 'color.png')

                if not os.path.isfile(depth_rs_path) or not os.path.isfile(depth_zivid_path):
                    continue  # skip incomplete samples

                self.imgs.append({
                    'left':  left_path,
                    'right': right_path,
                    'depth_rs': depth_rs_path,
                    'depth_zivid': depth_zivid_path,
                    'rgb':   rgb_path if os.path.isfile(rgb_path) else None,
                })

        if sub_indexes is not None:
            self.imgs = [self.imgs[i] for i in sub_indexes]

        log.info(f"DataSource: found {len(self.imgs)} samples in {input_rectified}")
        return len(self.imgs)

    def get_item(self, index: int, debug: bool = False):
        """Return one sample as a dict with left, right, depth_faro, depth_rs, rgb."""
        output_str = {"left": [], "right": [], "depth_zivid": [], "depth_rs": [], "rgb": []}

        entry = self.imgs[index]

        left_img  = cv2.imread(entry['left'],  cv2.IMREAD_UNCHANGED)
        right_img = cv2.imread(entry['right'], cv2.IMREAD_UNCHANGED)
        depth_rs_img = cv2.imread(entry['depth_rs'], cv2.IMREAD_UNCHANGED)
        depth_zivid_img = cv2.imread(entry['depth_zivid'], cv2.IMREAD_UNCHANGED)

        if left_img is None or right_img is None or depth_rs_img is None or depth_zivid_img is None:
            log.warning(f"Failed to load sample {index}: {entry}")
            return output_str

        

        rgb_img = np.array([], dtype=np.uint8)
        if entry['rgb'] is not None:
            rgb_img = cv2.imread(entry['rgb'], cv2.IMREAD_COLOR)
            if rgb_img is None:
                rgb_img = np.array([], dtype=np.uint8)

        depth_rs = depth_rs_img.astype(np.float32)
        depth_zivid = depth_zivid_img.astype(np.float32)   # uint16 mm → float32 mm

        output_str["left"]       = left_img
        output_str["right"]      = right_img
        output_str["depth_zivid"] = depth_zivid   # Zivid GT
        output_str["depth_rs"]   = depth_rs
        output_str["rgb"]        = rgb_img

        if debug:
            img_list = [left_img, right_img, depth_rs, depth_zivid]
            ttl_list = ['left (RS)', 'right (RS)', 'depth RS (mm)', 'depth Zivid (mm)']
            if rgb_img.size > 0:
                img_list.append(rgb_img)
                ttl_list.append('rgb (Zivid)')
            self.show_subset(img_list, ttl_list)

        return output_str

    def compute_depth_error(self, depth_pred, depth_gt, depth_mask=None):
        """Compute absolute depth error between prediction and GT."""
        depth_pred = depth_pred.astype(np.float32)
        depth_gt   = depth_gt.astype(np.float32)
        depth_error = np.zeros_like(depth_pred)
        mask = np.ones_like(depth_pred, dtype=bool) if depth_mask is None else depth_mask
        valid = np.logical_and(depth_gt > 0, mask)
        valid = np.logical_and(depth_pred > 0, valid)
        depth_error[valid] = np.abs(depth_pred[valid] - depth_gt[valid])
        return depth_error

    def show_subset(self, img_list, ttl_list, vmin=None, vmax=None, save_path='', fig_name=''):
        """Display a list of images in a grid."""
        img_num = len(img_list)
        col_num = min(img_num, 4)
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
            "img_depth_zivid.png": output_str["depth_zivid"].astype(np.uint16),
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
            p.show_subset([out["left"], out["right"], out["depth_zivid"], out["depth_rs"], out["rgb"]],
                          ['left (RS)', 'right (RS)', 'depth Zivid (mm)', 'depth RS (mm)', 'rgb (Zivid)'])

        plt.show()


# --------------------------------
#%% Run Test
def RunTest():
    tst = TestDataSource()
    #tst.test_get_item()
    tst.test_show_images()


if __name__ == '__main__':
    RunTest()
