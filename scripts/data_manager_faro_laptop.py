''' 

Dataset management from different image source.
Can read png files created from FARO dataset

Output : 
    Depth, Left, Right  or orther image types

Usage:

Environment : 
    docker fs

Install : 
    See README.md


'''

import numpy as np
import cv2 
import matplotlib.pyplot as plt
import os
import unittest
#from torch.utils.data import DataLoader

 # importing common Use modules



# --------------------------------
#%% Data source
class DataSource:

    def __init__(self):

        # params
        self.input_dir        = ''
        self.gray_scale_input = False
        self.dirs               = []
        self.count            = 0

        print('Source is defined')

    def get_bf(self):
        "image baseline for faro"
        return 49470.45

    def init_directory(self,  gray_scale_input = False, sub_indexes = None):
        "load entire directory"

        input_rectified      = r'./data/faro'
        dir_list             = os.listdir(input_rectified)
        self.dirs            = [file for file in dir_list if file.startswith('index')]

        #self.imgs            = glob.glob(os.path.join(input_rectified,  f"/**/*.png"),  recursive=True)
        self.gray_scale_input = gray_scale_input
        if sub_indexes is not None:
            self.dirs = [self.dirs[idx] for idx in sub_indexes]

        self.input_dir      = input_rectified
        print(f'Total directories {len(self.dirs)}')
        return len(self.dirs)
    
    def get_image_from_directory(self, dir_path):
        "get an d,l,r image from a training directory"

        # check if initialized
        fpath               = dir_path

        #print(f'Reading files from {fpath}.....')
        files               = os.listdir(fpath)
        file_extensions     = ['.png'] #['.png','.jpg','.bmp','z.v_0.png']
        file_name_part     = self.file_names[0] # depth name
        filtered_files      = [file for file in files if file.endswith(tuple(file_extensions))]
        file_list           = filtered_files
        frame_count         = 0
        print('Found %d.' %len(filtered_files))

        file_num            = len(file_list)
        if file_num < 1 or self.frame_count >= file_num:
            print('No image files are found')
            return False, None

        # Iterate over files and process them
        file_name           = file_list[self.frame_count]
        file_path           = os.path.join(fpath, file_name)
        img_array_d         = cv2.imread(file_path, cv2.IMREAD_UNCHANGED) 

        # depth with left
        file_path           = os.path.join(fpath, file_name.replace(self.file_names[0],self.file_names[1]))
        img_array_l         = cv2.imread(file_path, cv2.IMREAD_UNCHANGED).astype(img_array_d.dtype) 

        # depth with right
        file_path           = os.path.join(fpath, file_name.replace(self.file_names[0],self.file_names[2]))
        img_array_r         = cv2.imread(file_path, cv2.IMREAD_UNCHANGED).astype(img_array_d.dtype) 

        # assign data
        self.frame_gray     = img_array_d               
        self.frame_left     = img_array_l
        self.frame_right    = img_array_r        
        img_array           = np.stack((img_array_l,img_array_r,img_array_d),2)


        # check the number of files : self.direct_count can be 0,+1,-1
        self.frame_count    = (self.frame_count + self.direct_count) #% file_num
        self.frame_name     = file_name

        return True, img_array   


    def get_item(self, index: int, debug: bool = False):
        "get one item from the dataset"
        output_str          = {"img_left": [],  "img_right": [],   "img_depth_faro": [],   "img_depth_rs": [],   "img_rgb": []  }    

        if  index > len(self.dirs):           
            print(f'bad directory {base_folder}')
            return output_str

        # find path
        base_folder         = os.path.join(self.input_dir ,self.dirs[index])
        #print(f'Reading data from {base_folder}')
        img_path            = os.path.join(base_folder, "img_left.png")
        left_img            = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img_path            = os.path.join(base_folder, "img_right.png")
        right_img           = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img_path            = os.path.join(base_folder, "img_depth_faro.png")
        depth_faro_img      = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img_path            = os.path.join(base_folder, "img_depth_rs.png")
        depth_rs_img        = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img_path            = os.path.join(base_folder, "img_rgb.png")
        rgb_img             = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)        

        #left_img = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
        #right_img = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)

        if left_img is None or right_img is None or rgb_img is None:
            print(f'bad directory {base_folder}')
            return output_str
        if len(left_img)<1  or len(right_img) < 1:
            print(f'bad directory {base_folder}')
            return output_str        
        
        #print(right_img)
        
        #left_img, right_img, rgb_img, depth_rs_img, depth_faro_img = left_img, right_img, rgb_img, depth_rs_img.astype(np.float32), depth_faro_img.astype(np.float32)

        output_str["img_left"]          = left_img
        output_str["img_right"]         = right_img
        output_str["img_depth_faro"]    = depth_faro_img
        output_str["img_depth_rs"]      = depth_rs_img
        output_str["img_rgb"]           = rgb_img

        if debug:
            depth_error     = self.compute_depth_error(depth_rs_img, depth_faro_img)
            img_list        = [left_img, right_img, rgb_img, depth_rs_img, depth_faro_img, depth_error]
            ttl_list        = ['left','right','rgb','depth rs','depth faro','depth error']
            self.show_subset(img_list, ttl_list)        

        return output_str        

    def load_specific_files(self, debug = True):
        "specific files to load"  

        # find path
        base_folder         = r'./data/roi'
        #print(f'Reading data from {base_folder}')
        img_path            = os.path.join(base_folder, "2_Infrared.png")
        left_img            = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img_path            = os.path.join(base_folder, "1_Color.png")
        right_img           = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img_path            = os.path.join(base_folder, "1_Color.png")
        depth_faro_img      = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img_path            = os.path.join(base_folder, "2_Infrared.png")
        depth_rs_img        = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img_path            = os.path.join(base_folder, "1_Color.png")
        rgb_img             = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)          

        output_str          = {"img_left": [],  "img_right": [],   "img_depth_faro": [],   "img_depth_rs": [],   "img_rgb": []  }   
        output_str["img_left"]          = left_img
        output_str["img_right"]         = right_img
        output_str["img_depth_faro"]    = depth_faro_img[:,:,0]
        output_str["img_depth_rs"]      = depth_rs_img[:,:,0]
        output_str["img_rgb"]           = rgb_img

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
        
        #plt.show(block=False)
        plt.show()

    def save_image(self, frame, fname = ''):
        fn = './image_%03d_%s.png' % (self.count, fname)
        frame = frame.astype(np.uint16) #cv.cvtColor(frame, cv.CV_16U)
        cv2.imwrite(fn, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(fn, 'saved')
        self.count += 1          
    
 
    

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
        self.assertTrue(len(out_data["img_left"]) > 0)    

    def test_show_images(self):
        "show image from video file"
        p           = DataSource()
        img_num     = p.init_directory()
        img_index   = np.random.randint(0,img_num,8)
        for k in img_index:
            out_data    = p.get_item(k, debug = True)
            self.assertTrue(len(out_data["img_left"]) > 0) 
        plt.show() 



# --------------------------------
#%% Run Test
def RunTest():

    tst = TestDataSource()
    #tst.test_init_directory()
    #tst.test_get_item()
    tst.test_show_images()


#%%
if __name__ == '__main__':
    #print (__doc__)
    RunTest()