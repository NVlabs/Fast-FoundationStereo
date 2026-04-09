'''

Chessboard Object Manager.
Supported detection of the chess board object in the image and video. It is used for testing and demonstration of the pose estimation.:



Usage :
    python object_chessboard.py


Environemt : 
    .\\envs\\pyqt5g

Install : 


'''


import numpy as np
import cv2

import sys, os
current_dir = os.path.dirname(__file__)
parent_dir  = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

 # importing common Use modules 
from src.logger                     import log


# --------------------------------
#%% ObjectManager - manages the entire experiment
class ObjectChessboard:

    def __init__(self, config = None):

        # params
        self.config             = config
        self.name               = 'chessboard'
        self.frame              = []
        self.resolution         = (1280,720)
        self.square_size         = 10                  # size in mm of the pattern square         
        self.debug_on            = True
        self.pattern_size       = (9,6)
        
        
        self.init()
        log.info("Chess Object is Created")

            
    def init(self):
        # can be also string from file    
        
        # should not be done - connect to external
        if self.config is None:
            log.info("Configuration is not connected")
        else:
            self.pattern_size       = self.config.get('chessboard_size', self.pattern_size)
            
        ret = True
        return ret
            
    def set_square_size(self,sqSize = 21.8):
        # set chessboard size
        if sqSize < 0 or sqSize > 100:
            log.info('Square size should be in range 0.1:100 mm')
            return
         
        self.square_size = sqSize # 21mm, 
        
        log.info('Square size is %4.2f mm' % self.square_size)
        

    def set_pattern_size(self, pattern_size = (9,6)):
        # set chessboard pattern size
        if pattern_size[0] < 2 or pattern_size[1] < 2:
            log.info('Pattern size should be at least 2x2')
            return False
        
        self.pattern_size = pattern_size # (9,6) for 9x6 chessboard
        
        log.info('Pattern size is %dx%d' % self.pattern_size)
        return True

    def get_object_points(self):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        a = self.pattern_size[0]
        b = self.pattern_size[1]
        s = self.square_size # 21.8 # 21mm, but i want the units to be in meters
         
        objCorners        = np.zeros((b*a,3), np.float32)
        objCorners[:,:2]  = np.mgrid[0:a,0:b].T.reshape(-1,2)*s 
        return objCorners

    def get_image_points(self, img):
        "detect corners in the image and return their coordinates"

        # reduce size
        gray            = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # find the chess board (calibration pattern) corners
        flags           = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        ret, imgCorners = cv2.findChessboardCornersSB(gray, self.pattern_size, flags=flags)
        if ret:
            # Refine the corners of the detected corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            imgCorners2 = cv2.cornerSubPix(gray,imgCorners,(11,11),(-1,-1),criteria)
        else:
            imgCorners2= []
            #log.info('Chessboard corners not found in the image')
        return imgCorners2

    def detect(self, img):
        """Detect chessboard corners and return a unified detection dictionary.

        Args:
            img: BGR image.

        Returns:
            dict: Detection result with keys:
                - success (bool)
                - image_points (np.ndarray or list)
                - object_points (np.ndarray)
                - quality (float)
                - reason (str, when failed)
        """
        if img is None:
            return {
                'success': False,
                'reason': 'input image is None',
                'image_points': [],
                'object_points': self.get_object_points(),
                'quality': 0.0,
            }

        try:
            img_points = self.get_image_points(img)
        except Exception as e:
            return {
                'success': False,
                'reason': f'chessboard detection error: {e}',
                'image_points': [],
                'object_points': self.get_object_points(),
                'quality': 0.0,
            }

        has_points = len(img_points) > 0
        detected_count = int(len(img_points)) if has_points else 0
        expected_count = int(self.pattern_size[0] * self.pattern_size[1])
        quality = float(detected_count / max(expected_count, 1))

        result = {
            'success': has_points,
            'image_points': img_points,
            'object_points': self.get_object_points(),
            'quality': quality,
        }
        if not has_points:
            result['reason'] = 'chessboard corners not found'

        return result


    def draw_corners(self, img, corners):
        # draw corners on the image
        if len(corners) == 0:
            return img
        
        img_drawn = cv2.drawChessboardCorners(img, self.pattern_size, corners, True)
        return img_drawn

    def show_corners(self, img, corners):
        # show results
        if not self.debug_on:
            return False
        
        img = self.draw_corners(img, corners)
        
        cv2.imshow('Image with Corners',img)
                    
        #press q if you want to end the loop
        ret = cv2.waitKey(0) & 0xFF == ord('q')
        return ret


# ----------------------
#%% Tests
class TestObjectChessboard():

    def __init__(self):
        "init test"
        self.s = ObjectChessboard()
        log.info('TestObjectChessboard tests started')

    def assertTrue(self, isOk = True):
        "assert true"
        if not isOk:
            raise AssertionError("Test failed")

    def assertFalse(self, isOk = False):
        "assert false"
        if isOk:
            raise AssertionError("Test failed")        


    def test_object_detect_single_image(self):
        """
        Function that loads images, does measurement and shows final result
        """
        file_path           = r"data\calib_robot_0006.jpg"
        img                 = cv2.imread(file_path)
        self.assertTrue(img is not None)

        img_points         = self.s.get_image_points(img)
        self.assertTrue(len(img_points) > 0)
        
        isOk                = self.s.show_corners(img, img_points)
        self.assertTrue(isOk)

    def test_object_detect_video(self):
        """
        Function that does processing using video file
        """
        object_path         = r"D:\RobotAI\Customers\Plasel\Objects\plasel_gray-01"
        file_path           = r"D:\RobotAI\Customers\Plasel\Objects\plasel_gray-01\videos\object_0002.mp4"

        isOk                = self.s.pose6d.ObjectSelectSingle(object_path)
        self.assertTrue(isOk)
        
        isOk                = self.s.pose6d.TestRunFile(file_path)
        self.assertTrue(isOk)

    def test_create_new_object(self):
        """
        Function that creates new object by copying existing one
        """    
        import os
        object_path         = r"C:\Projects\Plasel\Objects\plasel_gray_ids-05"
        new_object_name     = 'plasel_blue-01'
        isOk                = self.s.pose6d.ObjectSelectSingle(object_path)
        self.assertTrue(isOk)

        current_object_path   = self.s.config.get_working_folder()
        isOk                = self.s.pose6d.ObjectCopy(current_object_path, new_object_name)
        self.assertTrue(isOk) 
        
        current_directory, current_name = os.path.split(current_object_path)
        new_object_path    = os.path.join(current_directory, new_object_name)
        isOk                = os.path.exists(new_object_path)
        self.assertTrue(isOk)
                   
    def test_object_configuration(self):
        """
        Function that loads yaml file for the object using settings
        
        """    
        import os
        current_directory   = self.s.config.get_working_folder()
        file_name            = 'plasel_gray-01'
        current_object_path  = os.path.join(current_directory, file_name)

        isOk                = self.s.pose6d.ObjectSelectSingle(current_object_path)
        self.assertTrue(isOk) 
        
        file_path           = r"D:\RobotAI\Customers\Plasel\Objects\plasel_gray-01\videos\object_0006\img37.jpg"
        isOk                = self.s.pose6d.TestSingleImage(file_path)
        self.assertTrue(isOk)  
        
    def test_rs_camera_connection(self):
        """
        Function that connects to IDS camera and shows live stream
        """
        camera_params       = self.s.config.get_camera_parameters()
        self.assertTrue(camera_params['cameraid']==3) 

        isOk                = self.s.camera_connect()
        self.assertTrue(isOk) 
        
        isOk                = self.s.camera_check_connected()
        self.assertTrue(isOk) 
        
        frame               = self.s.camera_read()
        self.assertTrue(frame is not None) 

        self.s.camera.ShowLive()   
        
        isOk                = self.s.camera_disconnect()
        self.assertTrue(isOk)       
           

# ----------------------------------------------------
#%% Run Test
def RunTest():
    "Run all tests in the MainApp class"
    tst = TestObjectChessboard()

    tst.test_object_detect_single_image() # ok

    

#%% Run ALL
if __name__ == '__main__':
    #print(__doc__)
    RunTest()
