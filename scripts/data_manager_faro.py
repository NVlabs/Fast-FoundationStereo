''' 

Dataset management from different image source.
Can read ros bag files, bin files, mp4 files and even image stream from the camera

Output : 
    Depth, Left, Right  or orther image types

Usage:

  Usage examples:
  # Training set — excludes KITCHEN paths
  ds = DataSource()
  ds.init_directory(root, test_keywords=['KITCHEN'], split='train')

  # Test set — only KITCHEN paths
  ds = DataSource()
  ds.init_directory(root, test_keywords=['KITCHEN'], split='test')

  # Access counts directly
  ds.train_imgs  # list of train paths
  ds.test_imgs   # list of test paths

Environment : 
    C:\\Users\\udubin\\Documents\\Envs\\barcode

Install : 
    See README.md


'''

from copyreg import pickle
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import os
import glob
import re
import scipy.io as sio
import unittest

 # importing common Use modules 
# import sys 
# sys.path.append(r'..\Utils\src')
#from logger import log
import logging as log

# --------------------------------
#%% Data source
class DataSource:

    def __init__(self):

        # params
        self.gray_scale_input = False
        self.imgs = []


        log.info('Source is defined')

    def init_directory(self, input_rectified = '', gray_scale_input = False, sub_indexes = None,
                       test_keywords = None, split = 'all'):
        """Load directory and optionally split into train/test by path keywords.

        Args:
            test_keywords: list of strings (case-insensitive); paths containing any
                           of these keywords are assigned to the test split.
                           E.g. ['KITCHEN', 'BATHROOM']
            split: 'all'   – return all images (default, backward-compatible)
                   'train' – return only images whose path matches no keyword
                   'test'  – return only images whose path matches at least one keyword
        """
        if len(input_rectified) < 3:
            input_rectified = r'/mnt/algonas/Local'

        all_imgs = glob.glob(os.path.join(input_rectified, "**/L_images/L_Img_**.mat"), recursive=True)
        self.gray_scale_input = gray_scale_input

        if test_keywords:
            keywords_upper = [kw.upper() for kw in test_keywords]
            def is_test(path):
                p = path.upper()
                return any(kw in p for kw in keywords_upper)

            self.train_imgs = [p for p in all_imgs if     is_test(p)]
            self.test_imgs  = [p for p in all_imgs if not is_test(p)]
        else:
            self.train_imgs = all_imgs
            self.test_imgs  = []

        if split == 'train':
            self.imgs = self.train_imgs
        elif split == 'test':
            self.imgs = self.test_imgs
        else:  # 'all'
            self.imgs = all_imgs

        if sub_indexes is not None:
            self.imgs = [self.imgs[idx] for idx in sub_indexes]

        log.info(f"Split='{split}': {len(self.imgs)} images "
                 f"(train={len(self.train_imgs)}, test={len(self.test_imgs)})")
        return len(self.imgs)


    def get_item(self, index: int, debug: bool = False):
        "get one item from the dataset"
        output_str          = {"left": [],  "right": [],   "depth_faro": [],   "depth_rs": [],   "rgb": []  }    

        # find path
        left_path           = self.imgs[index]
        right_path          = left_path.replace("L_images", "R_images").replace("L_Img", "R_Img")
        rgb_path            = left_path.replace("L_images", "RGB_images").replace("L_Img", "RGB_Img")
        depth_faro_path     = left_path.replace("L_images", "Depth").replace("L_Img", "Depth_Img")
        depth_rs_path       = left_path.replace("L_images", "Z_Intel_Depth").replace("L_Img", "Z_Intel_Img")

        base_folder         = left_path[:left_path.rfind("L_images") - 1]
        gt_test_folder      = os.path.join(base_folder, "Disparity")

        # right image
        filename_r          = os.path.basename(right_path)
        filename_r_wo_ext   = os.path.splitext(filename_r)[0]
        m                   = re.search('R_Img_(\d+)', filename_r_wo_ext)
        file_idx            = int(m.group(1))

        # read img
        left_img            = sio.loadmat(left_path)['Il']
        right_img           = sio.loadmat(right_path)['Ir']
        rgb_img             = sio.loadmat(rgb_path)['I_RGB']
        depth_faro_img      = sio.loadmat(depth_faro_path)['depth']
        depth_rs_img        = sio.loadmat(depth_rs_path)['Z_im']
        #left_img = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
        #right_img = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)

        if left_img is None or right_img is None or rgb_img is None:
            return output_str
        
        # if self.gray_scale_input:
        #     left_img = cv2.cvtColor(left_img.astype("uint8"), cv2.COLOR_BGR2GRAY)[None, :, :]
        #     right_img = cv2.cvtColor(right_img.astype("uint8"), cv2.COLOR_BGR2GRAY)[None, :, :]

        # test_name = f"{base_folder}"
        # prefix = f"{test_name}/{os.path.basename(left_path)}"
        # file_sources = {
        #     "left_path": left_path,
        #     "prefix": os.path.basename(prefix),
        #     "right_path": right_path,
        #     "left_disp_path": left_disp_filename
        # }

        
        left_img, right_img, rgb_img, depth_rs_img, depth_faro_img = left_img, right_img, rgb_img, depth_rs_img.astype(np.float32), depth_faro_img.astype(np.float32)

        output_str["left"]          = left_img
        output_str["right"]         = right_img
        output_str["depth_faro"]    = depth_faro_img
        output_str["depth_rs"]      = depth_rs_img
        output_str["rgb"]           = rgb_img

        if debug:
            depth_error     = self.compute_depth_error(depth_rs_img, depth_faro_img)
            img_list        = [left_img, right_img, rgb_img, depth_rs_img, depth_faro_img, depth_error]
            ttl_list        = ['left','right','rgb','depth rs','depth faro','depth error']
            self.show_subset(img_list, ttl_list)        

        return output_str          

    def compute_depth_error(self, depth_rs_img, depth_faro_img, depth_mask = None) :
        "compute depth error"
        depth_rs_img, depth_faro_img = depth_rs_img.astype(np.float32), depth_faro_img.astype(np.float32)
        depth_error = np.zeros_like(depth_rs_img)
        depth_mask  = np.ones_like(depth_rs_img,dtype=bool) if depth_mask is None else depth_mask
        
        #depth_valid = depth_faro_img > 0 if depth_mask is None else depth_mask # depth_rs_img > 0
        depth_valid = np.logical_and(depth_faro_img > 0, depth_mask)
        depth_valid = np.logical_and(depth_rs_img > 0, depth_valid)
        depth_error[depth_valid] = np.abs(depth_rs_img[depth_valid] - depth_faro_img[depth_valid])
        return depth_error
    
    def show_subset(self, img_list, ttl_list, vmin=None, vmax=None, save_path='', fig_name=''):
        "show some images"
        img_num  = len(img_list)
        row_num  = int(img_num/4) +1
        col_num  = int(img_num/row_num)
        fig, axes = plt.subplots(row_num, col_num, sharey=True, sharex=True)
        axes      = axes.reshape((row_num,col_num))
        do_save   = os.path.exists(save_path)
        for k in range(img_num):
            ri, ci = int(k / col_num), k % col_num
            pcm = axes[ri, ci].imshow(img_list[k], vmin=vmin, vmax=vmax)
            axes[ri, ci].set_title(ttl_list[k])     
            #fig.colorbar(pcm, ax=axes[ri, ci])  
        
        if do_save:
            fig.savefig(os.path.join(save_path, fig_name + ".png"))
        
        plt.show(block=False)

    def save_data_to_folder(self, output_str, output_directory):
        "save data dict to disk "

        # 3. Ensure the output directory exists
        # exist_ok=True prevents an error if the directory already exists
        os.makedirs(output_directory, exist_ok=True) 

        img_left            = output_str["left"].astype(np.uint16)    
        img_right           = output_str["right"].astype(np.uint16)
        depth_faro_img      = output_str["depth_faro"].astype(np.uint16)
        depth_rs_img        = output_str["depth_rs"].astype(np.uint16)
        rgb_img             = output_str["rgb"].astype(np.uint16)

        # 4. Create the full output path
        output_path         = os.path.join(output_directory, "img_left.png")
        success             = cv2.imwrite(output_path, img_left, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        output_path         = os.path.join(output_directory, "img_right.png")
        success             = cv2.imwrite(output_path, img_right, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        output_path         = os.path.join(output_directory, "img_depth_faro.png")
        success             = cv2.imwrite(output_path, depth_faro_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        output_path         = os.path.join(output_directory, "img_depth_rs.png")
        success             = cv2.imwrite(output_path, depth_rs_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        output_path         = os.path.join(output_directory, "img_rgb.png")
        success             = cv2.imwrite(output_path, rgb_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        return success
    
 
    

# --------------------------------        
#%% Tests
class TestDataSource(unittest.TestCase):

    def test_init_directory(self):
        "check image are in data source"
        p           = DataSource()
        img_num     = p.init_directory()
        self.assertTrue(img_num > 0)

    def test_get_item(self):
        "check image from data source"
        p           = DataSource()
        img_num     = p.init_directory()
        out_data    = p.get_item(7, debug = True)
        self.assertTrue(len(out_data["left"]) > 0)    

    def test_show_images(self):
        "show image from video file"
        p           = DataSource()
        img_num     = p.init_directory()
        if img_num == 0:
            log.warning("No images found in the directory.")
            return
        img_index   = np.random.randint(0, high=img_num, size=8)
        for k in img_index:
            out_data    = p.get_item(k, debug = True)
            self.assertTrue(len(out_data["left"]) > 0) 
        plt.show() 

    def test_show_images_and_save(self):
        "show image from data files and ssaves them to disk"
        p           = DataSource()
        img_num     = p.init_directory()
        if img_num == 0:
            log.warning("No images found in the directory.")
            return
        img_index   = np.random.randint(0, high=img_num, size=16)
        for k in img_index:
            out_data    = p.get_item(k, debug = True)
            out_folder  = f"C:\\Work\\Projects\\Deploy\\data\\fs\\index_{k:04d}"
            ret         = p.save_data_to_folder(out_data, output_directory = out_folder)
            self.assertTrue(ret) 
        plt.show() 

# --------------------------------
#%% Run Test
def RunTest():

    tst = TestDataSource()
    #tst.test_init_directory()
    tst.test_get_item()
    #tst.test_show_images()
    #tst.test_show_images_and_save()


#%%
if __name__ == '__main__':
    #print (__doc__)
    RunTest()