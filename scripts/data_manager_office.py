'''
Dataset management for packed Office stereo frames.

Reads `image_d16_*.png` files where channels are packed as:
    channel 0 -> left IR
    channel 1 -> right IR
    channel 2 -> depth (mm)

Expected directory layout:
    <root>/
      image_d16_000.png
      image_d16_001.png
      ...

Default root:
    C:\Work\Data\DepthRS\data\pattern_cube

Output dict keys (kept compatible with existing benchmark code):
    left         : numpy array (H, W) uint16
    right        : numpy array (H, W) uint16
    depth_rs     : numpy array (H, W) float32, mm
    depth_zivid  : numpy array (H, W) float32, mm (mirrors depth_rs for this dataset)
    rgb          : empty array (no RGB in packed d16 files)
'''

import glob
import logging as log
import os
import unittest

import cv2
import matplotlib.pyplot as plt
import numpy as np


log.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=log.INFO)

# ---------------------------------
# D405
CAMERA_MATRIX_RS = np.array([
    [638.77, 0, 644.23],
    [0, 638.77, 358.049],
    [0, 0, 1]
])

DIST_COEFFS_RS = np.array([
    0.0,
    -0.0,
    -0.0,
    0.0,
    -0.0
])

class DataSource:
    def __init__(self):
        self.gray_scale_input = False
        self.imgs = []  # list of packed d16 PNG paths
        log.info('Source is defined')

    def init_directory(self, input_rectified='', gray_scale_input=False, sub_indexes=None):
        """Scan root for packed d16 files and populate self.imgs."""
        if len(input_rectified) < 3:
            input_rectified = r'C:\Work\Data\DepthRS\data\pattern_cube'

        self.gray_scale_input = gray_scale_input

        if not os.path.isdir(input_rectified):
            log.error(f"Directory not found: {input_rectified}")
            self.imgs = []
            return 0

        # Support flat and nested layouts.
        self.imgs = sorted(glob.glob(os.path.join(input_rectified, '**', 'image_d16_*.png'), recursive=True))

        if sub_indexes is not None:
            self.imgs = [self.imgs[i] for i in sub_indexes]

        log.info(f"DataSource: found {len(self.imgs)} samples in {input_rectified}")
        return len(self.imgs)

    def get_item(self, index: int, debug: bool = False):
        """Return one sample from packed d16 file as left/right/depth maps."""
        output_str = {
            "left": [],
            "right": [],
            "depth_rs": [],
        }

        packed_path = self.imgs[index]
        packed_img = cv2.imread(packed_path, cv2.IMREAD_UNCHANGED)

        if packed_img is None:
            log.warning(f"Failed to load sample {index}: {packed_path}")
            return output_str

        if packed_img.ndim != 3 or packed_img.shape[2] < 3:
            log.warning(f"Expected 3-channel packed image, got shape={packed_img.shape} at: {packed_path}")
            return output_str

        left_img = packed_img[:, :, 0]
        right_img = packed_img[:, :, 1]
        depth_img = packed_img[:, :, 2].astype(np.float32)

        output_str["left"] = left_img
        output_str["right"] = right_img
        output_str["depth_rs"] = depth_img


        if debug:
            self.show_subset(
                [output_str["left"], output_str["right"], output_str["depth_rs"] ],
                ['left (packed ch0)', 'right (packed ch1)', 'depth RS (packed ch2, mm)']
            )

        return output_str

    def get_item_projected(self, index: int, debug: bool = False):
        """Compatibility wrapper for datasets without Zivid.

        For packed d16 files, depth_zivid is mirrored from depth_rs, so projection is not required.
        """
        return self.get_item(index=index, debug=debug)

    def compute_depth_error(self, depth_pred, depth_gt, depth_mask=None):
        """Compute signed depth error: pred - gt (mm) on valid pixels only."""
        depth_pred = depth_pred.astype(np.float32)
        depth_gt = depth_gt.astype(np.float32)
        depth_error = np.zeros_like(depth_pred)
        mask = np.ones_like(depth_pred, dtype=bool) if depth_mask is None else depth_mask
        valid = np.logical_and(depth_gt > 0, mask)
        valid = np.logical_and(depth_pred > 0, valid)
        depth_error[valid] = depth_pred[valid] - depth_gt[valid]
        return depth_error

    def show_subset(self, img_list, ttl_list, vmin=None, vmax=None, save_path='', fig_name=''):
        """Display a list of images in a compact grid."""
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
            fig.savefig(os.path.join(save_path, fig_name + '.png'))

        plt.show(block=False)

    def save_data_to_folder(self, output_str, output_directory):
        """Save sample dict to PNG files on disk."""
        os.makedirs(output_directory, exist_ok=True)

        paths = {
            'img_left.png': output_str['left'],
            'img_right.png': output_str['right'],
            'img_depth_rs.png': output_str['depth_rs'].astype(np.uint16),
        }

        success = True
        for fname, img in paths.items():
            out = cv2.imwrite(os.path.join(output_directory, fname), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            success = success and out

        if output_str['rgb'] is not None and np.asarray(output_str['rgb']).size > 0:
            cv2.imwrite(
                os.path.join(output_directory, 'img_rgb.png'),
                output_str['rgb'],
                [cv2.IMWRITE_PNG_COMPRESSION, 0],
            )

        return success


class TestDataSource(unittest.TestCase):
    def test_init_directory(self):
        p = DataSource()
        img_num = p.init_directory(r'C:\Work\Data\DepthRS\data\pattern_cube')
        self.assertTrue(img_num > 0)

    def test_get_item(self):
        p = DataSource()
        img_num = p.init_directory(r'C:\Work\Data\DepthRS\data\pattern_cube')
        self.assertTrue(img_num > 0)
        out = p.get_item(0, debug=True)
        self.assertTrue(len(out['left']) > 0)

    def test_show_images(self):
        p = DataSource()
        img_num = p.init_directory(r'C:\Work\Data\DepthRS\data\pattern_cube')
        if img_num == 0:
            log.warning('No images found.')
            return

        for k in np.random.randint(0, img_num, size=min(8, img_num)):
            out = p.get_item(int(k), debug=True)
            self.assertTrue(len(out['left']) > 0)
            p.show_subset(
                [out['left'], out['right'], out['depth_rs']],
                ['left', 'right', 'depth_rs(mm)'],
            )

        plt.show()

    def test_get_item_projected(self):
        p = DataSource()
        img_num = p.init_directory(r'C:\Work\Data\DepthRS\data\pattern_cube')
        self.assertTrue(img_num > 0)
        for k in np.random.randint(0, img_num, size=min(6, img_num)):
            out = p.get_item_projected(int(k), debug=True)
            err = p.compute_depth_error(out['depth_rs'], out['depth_zivid'])
            self.assertTrue(len(out['left']) > 0)
            p.show_subset(
                [out['left'], out['right'], out['depth_zivid'], out['depth_rs'], err],
                ['left', 'right', 'depth_zivid(mm)', 'depth_rs(mm)', 'error(mm)'],
            )
        plt.show()


def RunTest():
    tst = TestDataSource()
    # tst.test_get_item()
    tst.test_show_images()
    #tst.test_get_item_projected()


if __name__ == '__main__':
    RunTest()
