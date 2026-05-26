#!/usr/bin/env python

'''
Tester for multi planar plain detector with foundation stereo
==================

Using depth image to compute depth planes locally for specific ROI.


Usage:

Environemt : 
    ..\\docker

Install : 



'''

import sys 
import numpy as np
import cv2 as cv
import random
import unittest
#from scipy.spatial.transform import Rotation as Rot
import matplotlib.pyplot as plt
import logging 
log = logging.getLogger("robot")
log.setLevel(logging.DEBUG)
from opencv_realsense_camera import RealSense, draw_str
from run_fast_foundation_with_rs import convert_disparity_to_depth, foundation_stereo_algo_init, foundation_stereo_algo, process_arguments

#!/usr/bin/env python






#%% Main
class PlaneDetector:
    def __init__(self, detect_type = 'p', image_size = (1280,720)):

        self.detect_type    = detect_type   # plane

        self.frame_size     = image_size
        self.img            = None
        self.cam_matrix     = np.array([[1000,0,self.frame_size[0]/2],[0,1000,self.frame_size[1]/2],[0,0,1]], dtype = np.float32)
        self.cam_distort    = np.array([0,0,0,0,0],dtype = np.float32)

        self.img3d          = None  # contains x,y and depth plains
        self.img_xyz        = None  # comntains X,Y,Z information after depth image to XYZ transform
        self.img_mask       = None  # which pixels belongs to the plain
        self.rect           = None  # roi
        self.img_roi        = None  # roi image
        self.img_roi_normal = None  # normals at roi image

        # detector type     
        self.matrix_inv     = None     # holds inverse params of the 
        self.rect_z         = None     # flat z for ROI         
        self.rect_dir       = None     # direct u,v,1 for ROI
        self.rect_xyz       = None     # direct u,v,1 multiplied by z ROI 
        self.full_dir       = None     # direct u,v,1 for entire image
        self.full_xyz       = None     # direct u,v,1 multiplied by z entire image         
        self.roi_index      = None     # index of the points og an ROI in the original image     
        self.plane_params   = None     # rvec not normalized
        self.plane_center   = None     # tvec
        self.plane_confidence = 0      # reliability of the detcetion

        #self.corner_ind     = [0, 10, 40, 50]  # corner of the rectnagle for the projection
        self.rect_3d        = None    # roi but projected on 3D 

        # params
        self.MIN_SPLIT_SIZE  = 32
        self.MIN_STD_ERROR   = 0.01

        # color for the mask
        self.color_mask     = np.random.randint(0,255,3) # random color

        # help variable
        self.ang_vec     = np.zeros((3,1))  # help variable

    def init_image(self, img = None):
        "load image"

        self.img            = img
        h,w                 = img.shape[:2]
        self.frame_size     = (w,h)
        self.img_mask       = np.zeros((h,w))
        return True

    def init_roi(self, roi_type = 1):
        "load the test case"
        w,h     = self.frame_size[0],self.frame_size[1]
        w2,h2   = w>>1,h>>1
        roi     = [0,0,w,h]
        if roi_type == 1:
            roi = [w2-3,h2-3,w2+3,h2+3] # xlu, ylu, xrb, yrb
        elif roi_type == 2:
            roi = [300,220,340,260] # xlu, ylu, xrb, yrb
        elif roi_type == 3:
            roi = [280,200,360,280] # xlu, ylu, xrb, yrb            
        elif roi_type == 4:
            roi = [220,140,420,340] # xlu, ylu, xrb, yrb      
        elif roi_type == 5:
            roi = [200,120,440,360] # xlu, ylu, xrb, yrb    
        elif roi_type == 11:
            roi = [w2-16,h2-16,w2+16,h2+16] # xlu, ylu, xrb, yrb             
        elif roi_type == 12:
            roi = [w2-32,h2-32,w2+32,h2+32] # xlu, ylu, xrb, yrb    
        elif roi_type == 13:
            roi = [w2-64,h2-64,w2+64,h2+64] # xlu, ylu, xrb, yrb      
        elif roi_type == 14:
            roi = [w2-64,h2-48,w2+64,h2+48] # xlu, ylu, xrb, yrb      
        elif roi_type == 21: # lower center image position
            roi = [w2-64,h2+128,w2+64,h2+196] # xlu, ylu, xrb, yrb                
        elif roi_type == 22: # lower left image position
            roi = [w2-400,h2+128,w2-272,h2+196] # xlu, ylu, xrb, yrb  
        elif roi_type == 23: # lower right image position
            roi = [w2+272,h2+128,w2+400,h2+196] # xlu, ylu, xrb, yrb       
        elif roi_type == 31: # upper center image position
            #roi = [w2-64,h2-196,w2+64,h2-128] # xlu, ylu, xrb, yrb         
            roi = [w2+64,h2-128,w2+128,h2-64] # xlu, ylu, xrb, yrb                
        elif roi_type == 32: # upper left image position
            roi = [w2-400,h2-196,w2-272,h2-128] # xlu, ylu, xrb, yrb  
        elif roi_type == 33: # upper right image position
            roi = [w2+302,h2-196,w2+430,h2-128] # xlu, ylu, xrb, yrb    
        elif roi_type == 41: # center center image position
            #roi = [w2-64,h2+32,w2+64,h2+96] # xlu, ylu, xrb, yrb    
            roi = [w2-128,h2+64,w2+128,h2+196] # xlu, ylu, xrb, yrb    
        elif roi_type == 42: # center left image position
            roi = [w2-200,h2+32,w2-72,h2+96] # xlu, ylu, xrb, yrb 
        elif roi_type == 43: # center right image position
            roi = [w2+72,h2+32,w2+200,h2+96] # xlu, ylu, xrb, yrb   
        elif roi_type == 43: # center right image position
            roi = [w2+72,h2+32,w2+200,h2+96] # xlu, ylu, xrb, yrb      
        elif roi_type == 52: # upper left image position for data 432
            roi = [w2-256,h2-256,w2-128,h2-192] # xlu, ylu, xrb, yrb  
        elif roi_type == 53: # upper center image position
            roi = [w2-128,h2+16,w2+128,h2+300] # xlu, ylu, xrb, yrb                                            
        
        elif roi_type == 60: # 422 on cube smaller
            roi = [645,395,665,415] # xlu, ylu, xrb, yrb          
        elif roi_type == 61: # 422 on cube
            roi = [640,390,670,420] # xlu, ylu, xrb, yrb   
        elif roi_type == 62: # 422 right cube side
            roi = [690,390,720,420] # xlu, ylu, xrb, yrb     
        elif roi_type == 71: # set 422 - cube 
            roi = [630,h2+72,690,h2+128] # xlu, ylu, xrb, yrb    
        elif roi_type == 72: # set 422 - cube 
            roi = [430,h2-30,890,h2+228] # xlu, ylu, xrb, yrb   
        elif roi_type == 73: # set 422 - down side of the cube 
            roi = [630,h2+140,690,h2+190] # xlu, ylu, xrb, yrb             
        elif roi_type == 74: # set 422 - right side of the cube 
            roi = [700,h2+64,800,h2+164] # xlu, ylu, xrb, yrb  
        elif roi_type == 75: # set 422 - up side of the cube 
            roi = [700,h2+20,800,h2+96] # xlu, ylu, xrb, yrb            
        self.rect = roi       
        #self.rect_3d        = [[-w,-h,0],[w,-h,0],[w,h,0],[-w,h,0],[-w,-h,0]]                                                           
        log.info(f'Using ROI : {roi}')         
        return roi    

    def preprocess(self, img = None):
        "image preprocessing - extracts roi and converts from uint8 to float using log function"
        if img is None:
            log.info('No image provided')
            return False        

        if self.img_mask is None:
            ret = self.init_image(img)

        if self.rect is None: # use entire image
            roi = self.init_roi(4)

        # init params of the inverse
        if self.full_dir is None:
            self.fit_plane_init()              
            
        #x0, y0, x1, y1  = self.rect
        if len(img.shape) > 2:
            #img_roi        = img[y0:y1,x0:x1,2].astype(np.float32)
            img_roi        = img[:,:,2].astype(np.float32)
        else:
            #img_roi        = img[y0:y1,x0:x1].astype(np.float32)
            img_roi        = img.astype(np.float32)
        return img_roi         

    def init_img3d(self, img = None):
        "initializes xyz coordinates for each point"
        img     = self.img if img is None else img
        h,w     = img.shape[:2]
        x       = np.arange(w)
        y       = np.arange(h)
        x,y     = np.meshgrid(x,y)
        fx      = self.cam_matrix[0,0]
        fy      = self.cam_matrix[1,1]
        
        xy      = np.hstack((x.reshape(-1,1),y.reshape(-1,1)))
        xy      = np.expand_dims(xy, axis=1).astype(np.float32)
        xy_undistorted = cv.undistortPoints(xy, self.cam_matrix, self.cam_distort)

        u       = xy_undistorted[:,0,0].reshape((h,w))
        v       = xy_undistorted[:,0,1].reshape((h,w))
        z3d     = img.astype(np.float32)
        x3d     = z3d.copy()
        y3d     = z3d.copy()

        #ii        = np.logical_and(z3d> 1e-6 , np.isfinite(z3d))
        ii        = z3d > 5
        x3d[ii]   = u[ii]*z3d[ii] #/fx
        y3d[ii]   = v[ii]*z3d[ii] #/fy
        z3d[ii]   = z3d[ii]

        #self.img3d = np.stack((u/fx,v/fy,z3d), axis = 2)
        self.img3d      = np.stack((u,v,z3d), axis = 2)
        self.img_mask   = np.zeros((h,w))
        return self.img3d
    
    def compute_img3d(self, img = None):
        "compute xyz coordinates for each point using prvious init"
        img         = self.img if img is None else img
        xyz         = self.img3d
        if xyz is None:
            xyz = self.init_img3d(img)

        if np.any(img.shape[:2] != xyz.shape[:2]):
            print('Image dimension change')
            return 

        imgXYZ      = self.img3d.copy()

        z3d         = img.astype(np.float32)
        x3d         = self.img3d[:,:,0].copy()  # u/f
        y3d         = self.img3d[:,:,1].copy()  # v/f

        # filter bad z values
        #ii          = np.logical_and(z3d > 1e-6 , np.isfinite(z3d))
        ii          = z3d > 15
        x3d[ii]     = x3d[ii]*z3d[ii]
        y3d[ii]     = y3d[ii]*z3d[ii]
        z3d[ii]     = z3d[ii]

        # x,y,z coordinates in 3D
        imgXYZ[:,:,0] = x3d
        imgXYZ[:,:,1] = y3d
        imgXYZ[:,:,2] = z3d

        self.img_xyz = imgXYZ
        return imgXYZ

    def check_error(self, xyz1_mtrx, vnorm):
        "checking the error norm"
        err         = np.dot(xyz1_mtrx, vnorm)
        err_std     = err.std()
        return err_std
    
    def get_plane_params(self):
        "for external interface support"
        rvec_left              = self.plane_params
        tvec_left              = self.plane_center
        conf_left              = self.plane_confidence
        return tvec_left, rvec_left, conf_left

    def convert_plane_params(self, plane_equation):
        "convert plane params to rvec"
        # 4. Convert plane parameters to rvec and tvec
        #    - The plane normal vector is (A, B, C).
        #    - We can use the normal vector to get the rotation.
        #    - A point on the plane can be used for the translation vector.

        # Normalize the plane normal vector
        normal      = plane_equation #np.array([plane_equation[0], plane_equation[1], plane_equation[2]])
        normal_norm = np.linalg.norm(normal)
        if normal_norm == 0:
            log.error("Error: Zero norm for plane normal vector.")
            return None
        normal = normal / normal_norm

        # Use the normalized normal vector to get the rotation matrix
        # This is a common method, but there are other ways to do this.
        z_axis        = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, normal)
        rotation_angle = np.arccos(np.dot(z_axis, normal))

        # Handle the case where the rotation axis is zero (normal is parallel to z-axis)
        if np.linalg.norm(rotation_axis) < 1e-6:
            if normal[2] > 0:
                rvec = np.zeros(3)  # Rotation is identity
            else:
                rvec = np.array([0, np.pi, 0]) # Rotation by 180 degrees around X or Y.
        else:
            rvec, _ = cv.Rodrigues(rotation_axis * rotation_angle)
            rvec, _ = cv.Rodrigues(rvec)

        return rvec

    def convert_plane_params_to_pose(self, plane_params = None, plane_center = None):
        "converting params of the plane to the pose vector"

        plane_params = self.plane_params if plane_params is None else plane_params[:3].flatten()
        plane_center = self.plane_center if plane_center is None else plane_center[:3].flatten()

        tvec       = plane_center.reshape((1,-1))
        rvec       = plane_params.reshape((1,-1)) #reshape((-1,1))
        rvec       = rvec/np.linalg.norm(rvec.flatten())

        pose_norm  = np.hstack((tvec, rvec))
        #log.info('roi to pose')
        return pose_norm #.flatten()

    def fit_plane_init(self):
        "prepares data for real time fit a*x+b*y+c = z"
        if self.cam_matrix is None:
            self.cam_matrix   = np.array([[650,0,self.frame_size[0]/2],[0,650,self.frame_size[1]/2],[0,0,1]], dtype = np.float32)
            self.cam_distort  = np.array([0,0,0,0,0],dtype = np.float32)
            log.info('Camera matrix is initialized to default.')

        x0,y0,x1,y1     = 0,0,self.frame_size[0],self.frame_size[1] #self.rect 
        h,w             = y1-y0, x1-x0
        x_grid          = np.arange(x0, x1, 1)
        y_grid          = np.arange(y0, y1, 1)
        x, y            = np.meshgrid(x_grid, y_grid)  

        # remember corner indexes for reprojection [0 .... h*(w-1))
        #                                           .        .
        #                                           h ......h*w-1]
        #self.corner_ind = [0, h,  h*w-1, h*(w-1), 0]
        #h2,w2           = h>>1, w>>1
        #self.rect_3d    = [[-w,-h,0],[w,-h,0],[w,h,0],[-w,h,0],[-w,-h,0]]

        # camera coordinates
        xy              = np.hstack((x.reshape(-1,1),y.reshape(-1,1)))
        xy              = np.expand_dims(xy, axis=1).astype(np.float32)
        xy_undistorted  = cv.undistortPoints(xy, self.cam_matrix, self.cam_distort)

        u               = xy_undistorted[:,0,0].reshape((h,w)).reshape(-1,1)
        v               = xy_undistorted[:,0,1].reshape((h,w)).reshape(-1,1)

        # check
        #u, v            = u*self.cam_matrix[0,0], v*self.cam_matrix[1,1]

        self.full_dir   = np.hstack((u,v,u*0+1))
        #self.matrix_inv = np.linalg.pinv(self.rect_dir)

    def fit_plane_init_old(self):
        "prepares data for real time fit a*x+b*y+c = z"
        self.cam_matrix   = np.array([[650,0,self.frame_size[0]/2],[0,650,self.frame_size[1]/2],[0,0,1]], dtype = np.float32)
        self.cam_distort  = np.array([0,0,0,0,0],dtype = np.float32)

        x0,y0,x1,y1     = self.rect 
        h,w             = y1-y0, x1-x0
        x_grid          = np.arange(x0, x1, 1)
        y_grid          = np.arange(y0, y1, 1)
        x, y            = np.meshgrid(x_grid, y_grid)  

        # remember corner indexes for reprojection [0 .... h*(w-1))
        #                                           .        .
        #                                           h ......h*w-1]
        #self.corner_ind = [0, h,  h*w-1, h*(w-1), 0]
        h2,w2           = h>>1, w>>1
        self.rect_3d    = [[-w,-h,0],[w,-h,0],[w,h,0],[-w,h,0],[-w,-h,0]]

        # camera coordinates
        xy              = np.hstack((x.reshape(-1,1),y.reshape(-1,1)))
        xy              = np.expand_dims(xy, axis=1).astype(np.float32)
        xy_undistorted  = cv.undistortPoints(xy, self.cam_matrix, self.cam_distort)

        u               = xy_undistorted[:,0,0].reshape((h,w)).reshape(-1,1)
        v               = xy_undistorted[:,0,1].reshape((h,w)).reshape(-1,1)

        # check
        #u, v            = u*self.cam_matrix[0,0], v*self.cam_matrix[1,1]

        self.rect_dir = np.hstack((u,v,u*0+1))
        #self.matrix_inv = np.linalg.pinv(self.rect_dir)

    def convert_roi_to_points(self, img, point_num = 30, step_size = 1, roi_rect = None):
        "converting roi to pts in XYZ - Nx3 array. point_num - is the target point number"

        # init params of the inverse
        if self.full_dir is None:  # do not use mtrix_dir - initialized before
            self.fit_plane_init()  

        # deal iwth different rect options
        roi_rect            = self.rect if roi_rect is None else roi_rect
        x0, y0, x1, y1      = roi_rect

        # make rectangle 
        h,w                 = (y1-y0)>>1, (x1-x0)>>1
        self.rect_3d        = [[-w,-h,0],[w,-h,0],[w,h,0],[-w,h,0],[-w,-h,0]]

        # extract roi - must be compatible with image dimensions
        # n,m                 = img.shape[:2]
        # img_roi_mask        = np.zeros((n,m), dtype = np.bool_)
        # img_roi_mask[y0:y1,x0:x1] = True  
        # valid_bool          = img_roi_mask > 0 & img > 0

        # check if roi is valid. +1 to grow in positive x and y since arange does not include x1,y1
        x_grid              = np.arange(x0, x1, 1)
        y_grid              = np.arange(y0, y1, 1)
        x, y                = np.meshgrid(x_grid, y_grid) 
        #flat_indices        = np.ravel_multi_index((y, x), img.shape[:2]).reshape((-1,1))         
        flat_indices        = y * self.frame_size[0] + x
        flat_indices        = flat_indices.ravel().astype(np.int32)
        # valid under mask
        #valid_bool          = img.flat[flat_indices] > 0        
        #ii                  = flat_indices[valid_bool]
        img_roi             = img[y0:y1,x0:x1].flatten() #.astype(np.float32).reshape((-1,1)) 
        valid_bool          = img_roi > 1 # valid pixels in the roi
        ii                  = np.where(valid_bool)[0]
  
        valid_point_num     = len(ii)
        if valid_point_num < 5:
            return np.zeros((0,3))
        
        step_size           = np.maximum(step_size, np.int32(valid_point_num/point_num))
        ii                  = ii[::step_size]

        # plane params - using only valid
        z                   = img_roi[ii].reshape((-1,1))
        jj                  = flat_indices[ii].flatten()
        uv1_matrix          = self.full_dir[jj,:]
        xyz_matrix          = uv1_matrix[:,:3]*z  # keep 1 intact

        #self.plane_center   = xyz_center.flatten() 
        self.rect_z          = z
        self.rect_dir        = uv1_matrix
        self.rect_xyz        = xyz_matrix
        self.roi_index       = jj
        self.img_roi         = img[y0:y1,x0:x1]

        return xyz_matrix


    def convert_roi_to_points_old(self, img_roi, point_num = 30, step_size = 1):
        "converting roi to pts in XYZ - Nx3 array. point_num - is the target point number"
        # x1,y1       = self.img_xyz.shape[:2]
        # roi_area    = x1*y1

        # # reduce size of the grid for speed
        # if step_size < 1 and roi_area > 100:
        #     step_size   = np.maximum(1,int(np.sqrt(roi_area)/10))

          
        # #roi3d       = self.img_xyz[y0:y1:step_size,x0:x1:step_size,:]   
        # roi3d       = self.img_xyz[::step_size,::step_size,:]           
        # x,y,z       = roi3d[:,:,0].reshape((-1,1)), roi3d[:,:,1].reshape((-1,1)), roi3d[:,:,2].reshape((-1,1)) 
        # xyz_matrix  = np.hstack((x,y,z)) 
        # 
        
        # init params of the inverse
        if self.rect_dir is None:
            self.fit_plane_init_old()  

        # extract roi 

        n,m                 = img_roi.shape[:2]
        img_roi             = img_roi.reshape((-1,1))
        valid_bool          = img_roi > 0
        valid_bool          = valid_bool.flatten()
        #log.info(f'Timing : 1')  

        # all non valid
        ii                  = np.where(valid_bool)[0]
        valid_point_num     = len(ii)
        if valid_point_num < 5:
            return None
        step_size           = np.maximum(step_size, np.int32(valid_point_num/point_num))
        ii                  = ii[::step_size]

        # plane params - using only valid
        z                   = img_roi[ii]
        xyz_matrix          = self.rect_dir[ii,:]
        xyz_matrix[:,:3]    = xyz_matrix[:,:3]*z  # keep 1 intact

        # update corners of the rect in 3d
        #self.rect_3d        = self.rect_dir[self.corner_ind,:]*img_roi[self.corner_ind]
        # rect to show
        x0, y0, x1, y1      = self.rect
        h,w                 = y1-y0, x1-x0
        self.rect_3d        = [[-w,-h,0],[w,-h,0],[w,h,0],[-w,h,0],[-w,-h,0]]
        # substract mean
        #xyz_center          = xyz_matrix[:,:3].mean(axis=0)
        #xyz_matrix          = xyz_matrix - xyz_center   
        #log.info(f'Timing : 2')     

        # mtrx_dir            = np.hstack((self.rect_dir[valid_bool,0]*z,self.rect_dir[valid_bool,1]*z,z*0+1))
        # mtrx_inv            = np.linalg.pinv(mtrx_dir)
        # #mtrx_inv            = self.matrix_inv[:,valid_bool]
        # plane_params        = np.dot(mtrx_inv,z)

        # decimate to make it run faster  reduce size of the grid for speed. 1000 pix - 30x30 - step 1, 10000 pix - step=3
        #roi_area            = n*m
        #step_size           = int(np.sqrt(roi_area)/7) if roi_area > 1000 else 1  

        #self.plane_center   = xyz_center.flatten()   
        self.rect_xyz      = xyz_matrix          

        return xyz_matrix

    def fit_plane_svd(self, img_roi):
        "estimates mean and std of the plane fit"

        # roi converted to points with step size on the grid
        xyz_matrix          = self.convert_roi_to_points(img_roi, point_num = 600, step_size = 1)    

        # some problem with points
        if xyz_matrix.shape[0] < 2:
            log.warning('Not enough points in the ROI')
            return 0, 0    

        # substract mean
        xyz_center          = xyz_matrix[:,:3].mean(axis=0)
        xyz_matrix          = xyz_matrix - xyz_center   
        #log.info(f'Timing : 2')     

        # mtrx_dir            = np.hstack((self.rect_dir[valid_bool,0]*z,self.rect_dir[valid_bool,1]*z,z*0+1))
        # mtrx_inv            = np.linalg.pinv(mtrx_dir)
        # #mtrx_inv            = self.matrix_inv[:,valid_bool]
        # plane_params        = np.dot(mtrx_inv,z)

        # decimate to make it run faster  reduce size of the grid for speed. 1000 pix - 30x30 - step 1, 10000 pix - step=3
        #roi_area            = n*m
        #step_size           = int(np.sqrt(roi_area)/7) if roi_area > 1000 else 1
        
        # using svd to make the fit
        U, S, Vh            = np.linalg.svd(xyz_matrix, full_matrices=True)
        ii                  = np.argmin(S)
        vnorm               = Vh[ii,:]
        #log.info(f'Timing : 3') 

        # keep orientation
        plane_params       = vnorm*np.sign(vnorm[2])

        # estimate error
        err                = np.dot(xyz_matrix,plane_params)
        #z_est              = z + err + xyz_center[2]

        img_mean           = xyz_center[2] #z_est.mean()
        img_std            = err.std()
        self.plane_params  = plane_params[:3].flatten()
        self.plane_center  = xyz_center.flatten()

        #log.info(f'Plane : {self.plane_params}, error {img_std:.3f}, step {step_size}')
        
        return img_mean, img_std  
    
    def fit_plane_svd_old(self, img_roi):
        "estimates mean and std of the plane fit"
        # n,m             = img_roi.shape[:2]
        # img_roi         = img_roi.reshape((-1,1))
        # valid_bool      = img_roi > 0
        # valid_bool      = valid_bool.flatten()
        # #log.info(f'Timing : 1')  

        # # init params of the inverse
        # if self.matrix_inv is None:
        #     self.fit_plane_init()

        # # plane params - using only valid
        # z                   = img_roi[valid_bool]
        # xyz_matrix          = self.rect_dir[valid_bool,:]
        # xyz_matrix[:,:3]    = xyz_matrix[:,:3]*z  # keep 1 intact

        # update corners of the rect in 3d
        #self.rect_3d        = self.rect_dir[self.corner_ind,:]*img_roi[self.corner_ind]

        # roi converted to points with step size on the grid
        #xyz_matrix          = self.convert_roi_to_points(img_roi, point_num = 1e4, step_size = 1)    
        xyz_matrix          = self.convert_roi_to_points_old(img_roi, point_num = 1e4, step_size = 1) 


        # some problem with points
        if xyz_matrix.shape[0] < 2:
            log.warning('Not enough points in the ROI')
            return 0, 0                 

        # substract mean
        xyz_center          = xyz_matrix[:,:3].mean(axis=0)
        xyz_matrix          = xyz_matrix - xyz_center   
        #log.info(f'Timing : 2')     

        # mtrx_dir            = np.hstack((self.rect_dir[valid_bool,0]*z,self.rect_dir[valid_bool,1]*z,z*0+1))
        # mtrx_inv            = np.linalg.pinv(mtrx_dir)
        # #mtrx_inv            = self.matrix_inv[:,valid_bool]
        # plane_params        = np.dot(mtrx_inv,z)

        # decimate to make it run faster  reduce size of the grid for speed. 1000 pix - 30x30 - step 1, 10000 pix - step=3
        #roi_area            = n*m
        #step_size           = int(np.sqrt(roi_area)/7) if roi_area > 1000 else 1
        
        # using svd to make the fit
        U, S, Vh            = np.linalg.svd(xyz_matrix, full_matrices=True)
        ii                  = np.argmin(S)
        vnorm               = Vh[ii,:]
        #log.info(f'Timing : 3') 

        # keep orientation
        plane_params       = vnorm*np.sign(vnorm[2])

        # estimate error
        err                = np.dot(xyz_matrix,plane_params)
        #z_est              = z + err + xyz_center[2]

        img_mean           = xyz_center[2] #z_est.mean()
        img_std            = err.std()
        self.plane_params  = plane_params[:3].flatten()
        self.plane_center  = xyz_center.flatten()

        #log.info(f'Plane : {self.plane_params}, error {img_std:.3f}, step {step_size}')
        
        return img_mean, img_std  
        
    def fit_plane_svd_weighted(self, img_roi):
        "estimates mean and std of the plane fit - fit is weighted assuming Sigma(z) = a*z"
        # roi converted to points with step size on the grid
        xyz_matrix          = self.convert_roi_to_points(img_roi, point_num = 500, step_size = 1)    

        # some problem with points
        if xyz_matrix.shape[0] < 2:
            log.warning('Not enough points in the ROI')
            return 0, 0 
        
        # # substract mean
        # xyz_center          = xyz_matrix[:,:3].mean(axis=0)
        # xyz_matrix          = xyz_matrix - xyz_center          

        # plane params - using only valid
        z                   = self.rect_z 
        uv1_matrix          = self.rect_dir  # (x - x0)/f, (y - y0)/f, 1
 
        # minimization function min |ua/f + vb/f + c + dz|^2/|z|^2
        f                   = self.cam_matrix[0,0]
        # uvf_matrix          = np.dot(uv1_matrix , np.diag([1/f,1/f,1]))

        mtrx_inv            = np.linalg.pinv(uv1_matrix)
        b                   = f/z
        plane_params        = np.dot(mtrx_inv,b)
        plane_params        = plane_params/np.linalg.norm(plane_params)

        # center
        xyz_matrix          = self.rect_xyz  #uv1_matrix[:,:3]*z  # keep 1 intact
        xyz_center          = xyz_matrix.mean(axis=0)                

        # estimate error
        err                = np.dot(uv1_matrix,plane_params) 
        z_est              = z - err

        img_mean           = z_est.mean()
        img_std            = err.std()
        self.plane_params  = plane_params[:3].flatten()
        self.plane_center  = xyz_center.flatten()

        log.info(f'Plane : {self.plane_params}, error {img_std:.3f}')
        
        return img_mean, img_std 

    def fit_plane_with_outliers(self, img_roi):
        "computes normal for the specifric roi and evaluates error. Do it twice to reject outliers"
        # roi converted to points with step size on the grid
        xyz_matrix          = self.convert_roi_to_points(img_roi, point_num = 1500, step_size = 1)    

        # some problem with points
        if xyz_matrix.shape[0] < 2:
            log.warning('Not enough points in the ROI')
            return 0, 0    
        
        # substract mean
        xyz_center_1 = xyz_matrix[:,:3].mean(axis=0)
        xyz_1        = xyz_matrix - xyz_center_1         

        # using svd to make the fit to a sub group     
        U, S, Vh    = np.linalg.svd(xyz_1, full_matrices=True)
        ii          = np.argmin(S)
        vnorm       = Vh[ii,:]
        #vnorm       = vnorm*np.sign(vnorm[2]) # keep orientation

        # keep orientation
        plane_params = vnorm*np.sign(vnorm[2])

        # estimate error
        err         = np.dot(xyz_1,plane_params)        
        err_std     = err.std()
        log.info('Fit error iteration 1: %s' %str(err_std))

        # filter only the matching points
        inlier_ind  = np.abs(err) < 3*err_std

        # substract mean when only inliers are kept
        xyz_center_2 = xyz_matrix[inlier_ind,:3].mean(axis=0)#  
        xyz_2        = xyz_matrix[inlier_ind,:] - xyz_center_2         

        # perform svd one more time 
        U, S, Vh    = np.linalg.svd(xyz_2, full_matrices=True)
        ii          = np.argmin(S)
        vnorm       = Vh[ii,:]

        # keep orientation
        plane_params = vnorm*np.sign(vnorm[2])

        # checking error
        err         = np.dot(xyz_2, plane_params)
        err_std     = err.std()
        log.info('Fit error iteration 2: %s' %str(err_std))    

        # # We can convert this flat index to row and column indices
        # row_index, col_index = np.unravel_index(inlier_ind, self.img_mask.shape)
        # self.img_mask[row_index, col_index] = 1    

        img_mean           = xyz_center_2[2] #z_est.mean()
        img_std            = err_std
        self.plane_params  = plane_params[:3].flatten()
        self.plane_center  = xyz_center_2.flatten()

        #log.info(f'Plane : {self.plane_params}, error {img_std:.3f}, step {step_size}')
        
        return img_mean, img_std   
    
    def fit_plane_ransac(self, img_roi):
        
        """
        Find the best equation for a plane.

        :param pts: 3D point cloud as a `np.array (N,3)`.
        :param thresh: Threshold distance from the plane which is considered inlier.
        :param maxIteration: Number of maximum iteration which RANSAC will loop over.
        :returns:
        - `self.equation`:  Parameters of the plane using Ax+By+Cy+D `np.array (1, 4)`
        - `self.inliers`: points from the dataset considered inliers

        """
        #log.info('Fit ransac: ...')  
        # roi converted to points with step size on the grid
        #xyz_matrix     = self.convert_roi_to_points_old(img_roi, point_num = 250, step_size = 1)
        xyz_matrix     = self.convert_roi_to_points(img_roi, point_num = 250, step_size = 1)
        if xyz_matrix is None:
            log.error('No points in the ROI')
            return 0, 0

        thresh         = 1.05
        maxIteration   = 100


        n_points        = xyz_matrix.shape[0]
        best_eq         = []
        best_inliers    = []

        for it in range(maxIteration):

            # Samples 3 random points
            if n_points < 3: break
            id_samples = random.sample(range(0, n_points), 3)
            pt_samples = xyz_matrix[id_samples,:]

            # We have to find the plane equation described by those 3 points
            # We find first 2 vectors that are part of this plane
            # A = pt2 - pt1
            # B = pt3 - pt1

            vecA        = pt_samples[1, :] - pt_samples[0, :]
            vecB        = pt_samples[2, :] - pt_samples[0, :]

            # Now we compute the cross product of vecA and vecB to get vecC which is normal to the plane
            vecC        = np.cross(vecA, vecB)
            vecC_norm   = np.linalg.norm(vecC)

            # protect from the close spaced points
            if vecC_norm < 10e-6:
                continue

            # make sure that Z direction is positive
            vecC        = vecC * np.sign(vecC[2])

            # The plane equation will be vecC[0]*x + vecC[1]*y + vecC[0]*z = -k
            # We have to use a point to find k
            vecC        = vecC / vecC_norm
            #k           = -np.sum(np.multiply(vecC, pt_samples[1, :]))
            k           = -np.dot(vecC, pt_samples[1, :])
            plane_eq    = [vecC[0], vecC[1], vecC[2], k]

            # Distance from a point to a plane
            # https://mathworld.wolfram.com/Point-PlaneDistance.html
            # pt_id_inliers = []  # list of inliers ids
            # dist_pt = (
            #     plane_eq[0] * xyz_matrix[:, 0] + plane_eq[1] * xyz_matrix[:, 1] + plane_eq[2] * xyz_matrix[:, 2] + plane_eq[3]
            # ) / np.sqrt(plane_eq[0] ** 2 + plane_eq[1] ** 2 + plane_eq[2] ** 2)

            dist_pt            = np.dot(xyz_matrix, vecC) + plane_eq[3]

            # Select indexes where distance is biggers than the threshold
            pt_id_inliers       = np.where(np.abs(dist_pt) <= thresh)[0]
            if len(pt_id_inliers) > len(best_inliers):
                best_eq         = plane_eq
                best_inliers    = pt_id_inliers
        
        #self.inliers = best_inliers
        #self.equation = best_eq

        # rtansform to pose output
        #tvec            = xyz_matrix[best_inliers,:].mean(axis=0)
        #pts_best        = xyz_matrix[best_inliers,:] - tvec
        tvec            = xyz_matrix.mean(axis=0)
        pts_best        = xyz_matrix - tvec        
        vnorm           = np.array(best_eq[:3])

        # checking error
        err             = np.dot(pts_best, vnorm)
        err_std         = err.std()
        log.info('Fit error ransac: %s' %str(err_std))  

        img_mean           = tvec[2] #z_est.mean()
        img_std            = err_std
        self.plane_params  = vnorm.flatten()
        self.plane_center  = tvec.flatten()

        #log.info(f'Plane : {self.plane_params}, error {img_std:.3f}, step {step_size}')
        
        return img_mean, img_std 
    
    def estimate_normals_from_depth_map(self,depth_map):
        """
        Estimates the surface normal vector for each pixel in a depth map
        using the image gradient (Sobel operator).

        Args:
            depth_map (np.ndarray): A single-channel depth image (e.g., CV_32F or CV_64F).
                                    Depth values must be in a consistent metric (e.g., meters).

        Returns:
            np.ndarray: A 3-channel image (H, W, 3) where each pixel contains the
                        (nx, ny, nz) unit normal vector, as CV_32F.
        """
        # 1. Convert to CV_32F for accurate gradient calculation
        if depth_map.dtype != np.float32:
            depth_map = depth_map.astype(np.float32)

        depth_map = cv.GaussianBlur(depth_map, (5, 5), 0)   

        # 2. Calculate Derivatives using Sobel Operator (Gradient)
        # The kernel size 'ksize=1' is often preferred for depth maps as it corresponds 
        # to a 3x1 or 1x3 kernel, providing a close approximation of the derivative.
        ksize = 1 
        
        # Calculate dz/du (gradient in X/horizontal direction)
        # dx=1, dy=0
        grad_x = cv.Sobel(depth_map, cv.CV_32F, 1, 0, ksize=ksize, borderType=cv.BORDER_DEFAULT)
        
        # Calculate dz/dv (gradient in Y/vertical direction)
        # dx=0, dy=1
        grad_y = cv.Sobel(depth_map, cv.CV_32F, 0, 1, ksize=ksize, borderType=cv.BORDER_DEFAULT)

        # 3. Construct the Normal Vector Components
        # The normal vector is proportional to n = (-dz/du, -dz/dv, 1)
        
        # Reshape the gradients to (H, W, 1) for stacking
        grad_x = grad_x[:, :, np.newaxis]
        grad_y = grad_y[:, :, np.newaxis]
        
        # Create the 'z' component of the direction vector, which is always 1
        # np.ones_like creates an array with the same shape and type as the gradient arrays
        z_component = np.ones_like(grad_x)

        # Stack the components to create the direction vector (H, W, 3)
        # The X and Y gradients are negated: -dz/du and -dz/dv
        direction_vectors = np.concatenate((-grad_x, -grad_y, z_component), axis=2)

        # 4. Normalize the Direction Vectors
        # Calculate the magnitude (Euclidean norm) of each (nx, ny, nz) vector
        # axis=2 computes the norm across the 3 channels
        magnitude = np.linalg.norm(direction_vectors, axis=2, keepdims=True)
        
        # Use np.divide and np.where to prevent division by zero for magnitude=0
        # Set normals to (0, 0, 0) or another placeholder where magnitude is zero (flat or invalid depth)
        normals = np.divide(direction_vectors, magnitude, out=np.zeros_like(direction_vectors), where=magnitude != 0)

        return normals
    
    def estimate_normals_using_box_filters(self,img_roi):
        """
        Estimates the surface normal vector for each pixel in a depth map
        using the image gradient (Sobel operator).

        Args:
            img_roi (np.ndarray): A single-channel depth image (e.g., CV_32F or CV_64F).
                                    Depth values must be in a consistent metric (e.g., meters).

        Returns:
            np.ndarray: A 3-channel image (H, W, 3) where each pixel contains the
                        (nx, ny, nz) unit normal vector, as CV_32F.
        """    

        # sum of image pixels using box filter
        img_roi                 = img_roi.astype(np.float32)
        roih, roiw              = img_roi.shape[:2]                 
        kernel_size             = 5
        # count good points
        img_roi_mask            = (img_roi > 0).astype(np.float32)

        img_roi_sum             = cv.boxFilter(img_roi, -1,      (kernel_size, kernel_size), normalize=False) 
        img_roi_count           = cv.boxFilter(img_roi_mask, -1, (kernel_size, kernel_size), normalize=False) 

        # protect from non valid
        img_roi_count[img_roi_count < 1] = 1
        img_roi_mean            = img_roi_sum/img_roi_count

        # compute gradients
        shift                   = kernel_size>>1
        shift2                  = shift<<1
        img_normal              = np.ones((roih,roiw,3))
        # cross product
        #a × b = (a₂b₃ - a₃b₂)i + (a₃b₁ - a₁b₃)j + (a₁b₂ - a₂b₁)k        
        # dzdx, dzdy
        img_normal[:,shift:-shift,0]    = img_roi_mean[:,shift2:]   - img_roi_mean[:,:-shift2]
        img_normal[shift:-shift,:,1]    = img_roi_mean[shift2:,:]   - img_roi_mean[:-shift2,:]

        # align directions
        img_normal[:,:,0]               = -img_normal[:,:,0] # dx
        img_normal[:,:,1]               = -img_normal[:,:,1] # dy

        # normalize each vector to unit length
        norm2              = np.sqrt(np.sum(img_normal**2, axis=2))
        img_normal         = img_normal / norm2[:,:,np.newaxis]
        return img_normal

    def fit_plane_using_gradients(self, img_full, roi_rect = None):
        "estimates normal to the plane fit using gradients"

        # roi converted to points with step size on the grid
        xyz_roi             = self.convert_roi_to_points(img_full, point_num = 500, step_size = 1, roi_rect = roi_rect)    
        if self.img_roi is None:
            log.error('No ROI in image')
            return 0, 0
        img_roi             = self.img_roi.astype(np.float32)

        # old code
        #img_normal          = self.estimate_normals_using_box_filters(img_roi)

        img_normal           = self.estimate_normals_from_depth_map(img_roi)

        # roih, roiw          = img_roi.shape[:2]        

        # # count good points
        # img_roi_mask        = (img_roi > 0).astype(np.float32)

        # # sum of image pixels using box filter
        # kernel_size        = 7
        # img_roi_sum        = cv.boxFilter(img_roi, -1,      (kernel_size, kernel_size), normalize=False) 
        # img_roi_count      = cv.boxFilter(img_roi_mask, -1, (kernel_size, kernel_size), normalize=False) 

        # # protect from non valid
        # img_roi_count[img_roi_count < 1] = 1
        # img_roi_mean       = img_roi_sum/img_roi_count

        # # compute gradients
        # shift                   = kernel_size>>1
        # shift2                  = shift<<1
        # img_normal              = np.ones((roih,roiw,3))
        # # cross product
        # #a × b = (a₂b₃ - a₃b₂)i + (a₃b₁ - a₁b₃)j + (a₁b₂ - a₂b₁)k        
        # # dzdx, dzdy
        # img_normal[:,shift:-shift,0]    = img_roi_mean[:,shift2:]   - img_roi_mean[:,:-shift2]
        # img_normal[shift:-shift,:,1]    = img_roi_mean[shift2:,:]   - img_roi_mean[:-shift2,:]

        # # align directions
        # img_normal[:,:,0]               = -img_normal[:,:,0] # dx
        # img_normal[:,:,1]               = -img_normal[:,:,1] # dy

        # # normalize each vector to unit length
        # norm2              = np.sqrt(np.sum(img_normal**2, axis=2))
        # img_normal         = img_normal / norm2[:,:,np.newaxis]

        # plane normal
        plane_params       = img_normal.mean(axis=(0,1))

        # some problem with points
        if xyz_roi.shape[0] < 2:
            log.warning('Not enough points in the ROI')
            return 0, 0

        # estimate error
        xyz_center         = xyz_roi[:,:3].mean(axis=0)
        xyz_matrix         = xyz_roi - xyz_center         
        err                = np.dot(xyz_matrix,plane_params)

        img_mean           = xyz_center[2] #z_est.mean()
        img_std            = err.std()
        self.plane_params  = plane_params[:3].flatten()
        self.plane_center  = xyz_center.flatten()
        #self.plane_confidence = 1/(1+img_std)

        self.img_roi_normal = img_normal # save for debug and display

        log.info(f'Plane : {self.plane_params}, error {img_std:.3f}')
        
        return img_mean, img_std  
    
    def fit_plane_ransac_and_grow(self, img_full):
        
        """
        Find the best equation for a plane of the predefined ROI and then grow the ROI
        """
        h,w                         = img_full.shape[:2]
        if len(img_full.shape) > 2:
            img_full        = img_full[:,:,2].astype(np.float32)

        # start from the original ROI
        if self.img_mask is None:
            isOk                    = self.init_image(img_full)

        #img_mean, img_std           = self.fit_plane_ransac(img_full) 

        # make sure that mask is not empty - initial rectangle
        x0, y0, x1, y1              = self.rect
        self.img_mask[y0:y1,x0:x1]  = 1

        # grow the mask
        y,x                         = np.where(self.img_mask > 0.7)
        y_min, y_max                = y.min(), y.max()
        x_min, x_max                = x.min(), x.max()
        y_min, y_max                = np.maximum(0,y_min-1), np.minimum(self.img_mask.shape[0],y_max+2)
        x_min, x_max                = np.maximum(0,x_min-1), np.minimum(self.img_mask.shape[1],x_max+2)

        # extract ROI
        roi_rect                    = [x_min, y_min, x_max, y_max]
        #img_roi                     = img_full[y_min:y_max,x_min:x_max].astype(np.float32)
        xyz_matrix                  = self.convert_roi_to_points(img_full, point_num = 5000, step_size = 1, roi_rect = roi_rect)

        # check against the plane : do not substract plane.center from all the points
        vecC                        = self.plane_params[:3]
        dist_offset                 = np.dot(self.plane_center, vecC) 
        dist_pt                     = np.dot(xyz_matrix, vecC) - dist_offset

        # Select indexes where distance is biggers than the threshold
        thresh                      = 3.5
        err                         = np.abs(dist_pt)
        i2                          = np.where( err <= thresh)[0]

        # transfer xi,yi coordinates to the original image index
        ii                          = self.roi_index[i2] # convert to 2D index

        # update mask according to the valid pixels
        self.img_mask               = 0.95*self.img_mask
        self.img_mask.flat[ii]      = self.img_mask.flat[ii] + 0.5*(1 - self.img_mask.flat[ii])


        # position in 2d array
        # unravel_index(a.argmax(), a.shape)   

        # output
        img_std                    = err.std()
        img_mean                   = xyz_matrix[i2].mean(axis=0)[2]


        return img_mean, img_std 
        
    def fit_and_split_roi_recursively(self, roi, level = 0):
        # splits ROI on 4 regions and recursevly call 
        x0,y0,x1,y1     = roi
        #roi3d           = self.img_xyz[y0:y1,x0:x1,:]   
        log.info('Processing level %d, region x = %d, y = %d' %(level,x0,y0))
        # check the current fit
        roi_params_f    = self.fit_plane(roi)
        roi_params_ret  = [roi_params_f]
        if roi_params_f['error'] < self.MIN_STD_ERROR:
            log.info('Fit is good enough x = %d, y = %d' %(x0,y0))
            return roi_params_ret

        # too small exit
        xs, ys          = int((x1 + x0)/2), int((y1 + y0)/2)
        if (xs - x0) < self.MIN_SPLIT_SIZE or (ys - y0) < self.MIN_SPLIT_SIZE:
            log.info('Min size is reached x = %d, y = %d' %(x0,y0))
            return roi_params_ret
        
        # 4 ROIs - accept the split if error of one of them is lower from the total
        roi_params_list = []
        roi_split   = [[x0,y0,xs,ys],[x0,ys,xs,y1],[xs,y0,x1,ys],[xs,ys,x1,y1]]
        for roi_s in roi_split:
            roi_params_prev = self.fit_and_split_roi_recursively(roi_s, level + 1)
            # save locally
            #roi_params_list.append(roi_params_prev)
            roi_params_list = roi_params_list + roi_params_prev
            
        # extract each of the below and check the error
        makeTheSplit = False
        for roi_params_s in roi_params_list:
            #roi_params_s       = roi_params_prev[-1]
            # accept the split if twice lower (if noise of 4 split should be 2)
            if roi_params_s['error'] < roi_params_f['error']/2:
                makeTheSplit = True
                break

        # decide what to return
        if makeTheSplit:
            roi_params_ret = roi_params_list
            log.info('Split at level %d, region x = %d, y = %d' %(level,x0,y0))
        else:
            log.info('No split level %d, region x = %d, y = %d' %(level,x0,y0))

        return roi_params_ret
    
    def fit_plane_svd_weighted_4d(self, img_roi):
        "estimates weighted plane fit using weight inversely proportional to the depth"

        # roi converted to points with step size on the grid
        xyz_matrix          = self.convert_roi_to_points(img_roi, point_num = 350, step_size = 1)    

        # some problem with points
        if xyz_matrix.shape[0] < 2:
            log.warning('Not enough points in the ROI')
            return 0, 0         

        # form matrix [u,v,f,zf] result of sigma(z) = alpha*z
        if self.roi_index is None:
            log.error('No ROI index found')
            return 0,0
        
        # form matrix for svd
        f                   = self.cam_matrix[0,0]          
        # uv1z_matrix         = np.hstack((self.full_dir[self.roi_index,:],xyz_matrix[:,2].reshape((-1,1))))
        # uv1z_matrix[:,2:3] *= f  # keep 1 intact

        # plane params - using only valid
        z                   = self.rect_z 
        uv1_matrix          = self.rect_dir     
        uv1z_matrix         = np.hstack((uv1_matrix,1/z))
        uv1z_matrix[:,2:3] *= f  # keep 1 intact           
  
        # using svd to make the fit
        U, S, Vh            = np.linalg.svd(uv1z_matrix, full_matrices=True)
        ii                  = np.argmin(S)
        vnorm               = Vh[ii,:]

        # keep orientation
        plane_params       = vnorm*np.sign(vnorm[2])

        # estimate error
        err                = np.dot(xyz_matrix,plane_params[:3])

        # patch mean
        xyz_center         = xyz_matrix[:,:3].mean(axis=0)        
        img_mean           = xyz_center[2] #z_est.mean()
        img_std            = err.std()
        self.plane_params  = plane_params[:3].flatten()
        self.plane_center  = xyz_center.flatten()

        log.info(f'Plane : {self.plane_params}, error {img_std:.3f}')
        
        return img_mean, img_std  
    
    def fit_plane_and_project_the_image(self, img_full):
        
        """
        Find the best equation for a plane of the predefined ROI and then projecvt the entire image on the plane
        """
        h,w                         = img_full.shape[:2]
        if len(img_full.shape) > 2:
            img_full        = img_full[:,:,2].astype(np.float32)

        # start from the original ROI
        if self.img_mask is None:
            isOk                    = self.init_image(img_full)

        # extract ROI
        roi_rect                    = [50, 50, w-50, h-50]
        xyz_matrix                  = self.convert_roi_to_points(img_full, point_num = 50000, step_size = 1, roi_rect = roi_rect)

        # check against the plane : do not substract plane.center from all the points
        vecC                        = self.plane_params[:3]
        dist_offset                 = np.dot(self.plane_center, vecC) 
        dist_pt                     = np.dot(xyz_matrix, vecC) - dist_offset

        # Select indexes where distance is biggers than the threshold
        thresh                      = 1.5
        err                         = np.abs(dist_pt)
        i2                          = np.where(err <= thresh)[0]

        # transfer xi,yi coordinates to the original image index
        ii                          = self.roi_index[i2] # convert to 2D index

        # update mask according to the valid pixels
        self.img_mask.flat[ii]      = 1
        # make sure that mask is not empty - initial rectangle
        # x0, y0, x1, y1              = self.rect
        # self.img_mask[y0:y1,x0:x1]  = 1

        # position in 2d array
        # unravel_index(a.argmax(), a.shape)   

        # output
        img_std                    = err.std()
        img_mean                   = xyz_matrix[i2].mean(axis=0)[2]


        return img_mean, img_std 
        
    def fit_plane_and_project_the_image_using_gradients(self, img_full):
        
        """
        Find the best equation for a plane of the predefined ROI and then projecvt the entire image on the plane
        """
        h,w                         = img_full.shape[:2]
        if len(img_full.shape) > 2:
            img_full        = img_full[:,:,2].astype(np.float32)

        # start from the original ROI
        if self.img_mask is None:
            isOk                    = self.init_image(img_full)

        # extract ROI
        #roi_rect                    = [50, 50, w-50, h-50]
        #xyz_matrix                  = self.convert_roi_to_points(img_full, point_num = 50000, step_size = 1, roi_rect = roi_rect)

        # estimate normals for the entire image
        img_normal                  = self.estimate_normals_from_depth_map(img_full)

        # check against the plane : do not substract plane.center from all the points
        vecC                        = self.plane_params[:3]
        dist_pt                     = np.dot(img_normal, vecC)

        # Select indexes where distance is biggers than the threshold
        thresh                      = 0.1
        dist_abs                    = np.abs(dist_pt)
        #ii                          = np.where(dist_abs > thresh)[0]
        ii                          = dist_abs > thresh
        # transfer xi,yi coordinates to the original image index
        #ii                          = self.roi_index[i2] # convert to 2D index

        # update mask according to the valid pixels
        #self.img_mask.flat[ii]      = 1
        # make sure that mask is not empty - initial rectangle
        x0, y0, x1, y1              = self.rect
        self.img_mask[y0:y1,x0:x1]  = 1

        # update mask according to the valid pixels
        self.img_mask               = 0.95*self.img_mask
        #self.img_mask.flat[ii]      = self.img_mask.flat[ii] + 0.5*(1 - self.img_mask.flat[ii]) 
        self.img_mask[ii]           = self.img_mask[ii] + 0.5*(1 - self.img_mask[ii]) 

        # output
        img_std                    = dist_abs.std()
        img_mean                   = 0 #xyz_matrix[i2].mean(axis=0)[2]


        return img_mean, img_std 
        


    def growingStep(self, depths:np.ndarray, step:int, direction:str, mean:float, std:float,
                    left:int, top:int, right:int, bottom:int, numConfInt:int=3, testRatio:float=0.95):
        """
        testing whether growing in a given direction and size is acceptable
        :param depths:  uncropped depth image
        :param step: step size to test
        :param direction: growing direction. should be either: 'left', 'right', 'top' or  'bottom'
        :param mean: the current mean depth of the segment
        :param std: the current standard deviation of the segment's depths
        :param left: the current left bound of the segment
        :param top: the current top bound of the segment
        :param right: the current right bound of the segment
        :param bottom: the current bottom bound of the segment
        :param numConfInt: number of confidence intervals threshold
        :param testRatio: ratio of elements in growing area to be passed the confidence interval threshold
        :return: true if growing is accepted, otherwise false
        """
        directions = ['left', 'right', 'bottom', 'top']
        if direction not in directions:
            raise ValueError('direction must be one of "left", "right", "bottom", "top"')

        if step < 0:
            raise ValueError('step must be non-negative')

        if numConfInt < 1:
            raise ValueError('numConfInt must be at least 1')

        if testRatio > 1 or testRatio < 0:
            raise ValueError('testRatio must be between 0 and 1')

        data = {
            'left': depths[bottom:top + 1, left - step:left] if step > 0 else None,
            'right': depths[bottom:top + 1, right + 1:right + step + 1] if step > 0 else None,
            'top': depths[top + 1:top + step + 1, left:right] if step > 0 else None,
            'bottom': depths[bottom - step:bottom, left:right] if step > 0 else None,
        }
        testData = data[direction]
        testRows, testCols = np.nonzero(testData)  # getting the indexes of all nonzero elements
        norTestData = np.abs(testData[testRows, testCols] - mean) / std  # normalizing test data to standard values
        test = np.nonzero(norTestData < numConfInt)[0]  # test if elements are below confidence interval threshold
        if test.size / testData.size > testRatio:
            return True     # growing is acceptable
        else:
            return False    # growing is denied

    def findMaxROI(self, depths:np.ndarray, initialRoi) -> tuple[int, int, int, int]:
        """
        Find maximum ROI in depth image using region growing from an initial ROI
        :param depths: entire depth image
        :param initialRoi: initial ROI to grow from as list/tuple in the order: left, top, right, bottom
        :return: final ROI found as a tuple in the order: left, top, right, bottom
        """
        height, width = depths.shape
        left, top, right, bottom = initialRoi
        stepLeft = stepRight = stepTop = stepBottom = 1     # initializing growing steps

        while stepLeft + stepRight + stepTop + stepBottom > 0:
            currentData = depths[bottom:top + 1, left:right + 1]
            currentRows, currentCols = np.nonzero(currentData)   # getting the indexes of all nonzero elements
            currentMean = currentData[currentRows, currentCols].mean()
            currentStd = currentData[currentRows, currentCols].std()

            if stepLeft > 0:    # attempting to grow to the left
                if left - stepLeft < 0:  # growing left with current step exceeded image dimensions
                    stepLeft = 1 if stepLeft > 1 else 0
                elif left == 0:          # growing reached and of image, no more growing available
                    stepLeft = 0
                else:
                    res = self.growingStep(depths, stepLeft, 'left', currentMean, currentStd, left, top, right, bottom)
                    if res:
                        left -= stepLeft
                        stepLeft *= 2   # increasing growing step for next iteration
                    else:
                        stepLeft = 1 if stepLeft > 1 else 0

            if stepRight > 0:   # attempting to grow to the right
                if right + stepRight > width:  # growing right with current step exceeded image dimensions
                    stepRight = 1 if stepRight > 1 else 0
                elif right == width - 1:       # growing reached and of image, no more growing available
                    stepRight = 0
                else:
                    res = self.growingStep(depths, stepRight, 'right', currentMean, currentStd, left, top, right, bottom)
                    if res:
                        right += stepRight
                        stepRight *= 2   # increasing growing step for next iteration
                    else:
                        stepRight = 1 if stepRight > 1 else 0

            if stepBottom > 0:    # attempting to grow down
                if bottom - stepBottom < 0:  # growing down with current step exceeded image dimensions
                    stepBottom = 1 if stepBottom > 1 else 0
                elif bottom == 0:          # growing reached and of image, no more growing available
                    stepBottom = 0
                else:
                    res = self.growingStep(depths, stepBottom, 'bottom', currentMean, currentStd, left, top, right, bottom)
                    if res:
                        bottom -= stepBottom
                        stepBottom *= 2   # increasing growing step for next iteration
                    else:
                        stepBottom = 1 if stepBottom > 1 else 0

            if stepTop > 0:   # attempting to grow up
                if top + stepTop > height:  # growing up with current step exceeded image dimensions
                    stepTop = 1 if stepTop > 1 else 0
                elif top == height - 1:       # growing reached and of image, no more growing available
                    stepTop = 0
                else:
                    res = self.growingStep(depths, stepTop, 'top', currentMean, currentStd, left, top, right, bottom)
                    if res:
                        top += stepTop
                        stepTop *= 2   # increasing growing step for next iteration
                    else:
                        stepTop = 1 if stepTop > 1 else 0

            # print(left, top, right, bottom, '|', stepLeft, stepTop, stepRight, stepBottom)

        return left, top, right, bottom    
    
    def find_planes(self, img):
        "finds planes using different algo"
        detect_type         = self.detect_type.upper()

        img_mean, img_std   = 0,0             
        if detect_type == 'P':
            img_roi             = self.preprocess(img)
            img_mean, img_std   = self.fit_plane_svd(img_roi)  

        elif detect_type == 'W':
            img_roi             = self.preprocess(img)
            img_mean, img_std   = self.fit_plane_svd_weighted(img_roi)   

        elif detect_type == 'O':
            img_roi             = self.preprocess(img)
            img_mean, img_std   = self.fit_plane_with_outliers(img_roi)  

        elif detect_type == 'T': # weighted by depth
            img_roi             = self.preprocess(img)
            img_mean, img_std   = self.fit_plane_svd_weighted_4d(img_roi) 

        elif detect_type == 'R':
            img_roi             = self.preprocess(img)
            img_mean, img_std   = self.fit_plane_ransac(img_roi) 

        elif detect_type == 'G': # gradients
            img_roi             = self.preprocess(img)
            img_mean, img_std   = self.fit_plane_using_gradients(img_roi) 
            
        elif detect_type == 'F':
            img_roi             = self.preprocess(img)
            img_mean, img_std   = self.fit_plane_svd(img_roi) #fit_plane_ransac(img_roi)   # initial ROI plane          
            img_mean, img_std   = self.fit_plane_ransac_and_grow(img)                

        elif detect_type == 'S': # project all the points on the plane and color them accordingly
            img_roi             = self.preprocess(img)
            img_mean, img_std   = self.fit_plane_svd(img_roi) #fit_plane_ransac(img_roi)   # initial ROI plane 
            #img_mean, img_std   = self.fit_plane_svd_weighted(img_roi)      
            # img_mean, img_std   = self.fit_plane_svd_weighted_4d(img_roi)       
            img_mean, img_std   = self.fit_plane_and_project_the_image(img)  

        elif detect_type == 'N': # project all the points on the plane and color them accordingly
            img_roi             = self.preprocess(img)
            img_mean, img_std   = self.fit_plane_svd(img_roi) #fit_plane_ransac(img_roi)   # initial ROI plane 
            img_mean, img_std   = self.fit_plane_and_project_the_image_using_gradients(img)             
                               
            
        #log.debug(f'camera noise           - roi mean : {img_mean}')
        self.img_mean       = img_mean        # final measurements per frame
        self.img_std        = img_std    
        return True 

    def process_frame(self, img):
        "process the entire image and find the planes"

        img_roi     = self.preprocess(img)
        img3d       = self.init_img3d(img_roi)
        imgXYZ      = self.compute_img3d(img_roi)
        roim,rois   = self.fit_plane_with_outliers(img_roi)
        pose        = self.convert_plane_params_to_pose()

        return pose


#%% Helpers
def draw_axis(img, rvec, tvec, cam_mtrx, cam_dist, len = 10):
    # unit is mm
    try:
        points          = np.float32([[len, 0, 0], [0, len, 0], [0, 0, len], [0, 0, 0]]).reshape(-1, 3)
        axisPoints, _   = cv.projectPoints(points, rvec.astype(np.float32), tvec.astype(np.float32), cam_mtrx, cam_dist)
        axisPoints      = axisPoints.squeeze().astype(np.int32)
        img = cv.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (0,0,255), 3)
        img = cv.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0,255,0), 3)
        img = cv.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (255,0,0), 3)
    except Exception as e:
        print(e)
        print(rvec, tvec, img.shape)
    return img

def draw_polygon(img, rvec, tvec, cam_mtrx, cam_dist, points3d):
    # unit is mm
    points              = np.float32(points3d).reshape(-1, 3)
    polygon_points, _   = cv.projectPoints(points, rvec, tvec, cam_mtrx, cam_dist)
    polygon_points      = polygon_points.squeeze().astype(np.int32)
    img                 = cv.polylines(img, [polygon_points], True, (0, 200, 200), 1)

    # To fill the polygon, use thickness=-1
    # cv2.fillPoly(img, [pts], color)

    return img

def draw_cube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img

#%% ROI selector from OpenCV
class RectSelector:
    def __init__(self, win, callback):
        self.win = win
        self.callback = callback
        cv.setMouseCallback(win, self.onmouse)
        self.drag_start = None
        self.drag_rect = None
    def onmouse(self, event, x, y, flags, param):
        x, y = np.int16([x, y]) # BUG
        if event == cv.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            return
        if self.drag_start:
            if flags & cv.EVENT_FLAG_LBUTTON:
                xo, yo = self.drag_start
                x0, y0 = np.minimum([xo, yo], [x, y])
                x1, y1 = np.maximum([xo, yo], [x, y])
                self.drag_rect = None
                if x1-x0 > 0 and y1-y0 > 0:
                    self.drag_rect = (x0, y0, x1, y1)
            else:
                rect = self.drag_rect
                self.drag_start = None
                self.drag_rect = None
                if rect:
                    self.callback(rect)
    def draw(self, vis):
        if not self.drag_rect:
            return False
        x0, y0, x1, y1 = self.drag_rect
        cv.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        return True
    @property
    def dragging(self):
        return self.drag_rect is not None

#%% Data Generator
class DataGen:
    def __init__(self, img_size = (640,480)):

        self.frame_size     = img_size
        self.img            = None
        self.rect           = None  # roi  


    def add_noise(self, img_gray, noise_percentage = 0.01):
        "salt and pepper noise"
        if noise_percentage < 0.001:
            return img_gray


        # Get the image size (number of pixels in the image).
        img_size = img_gray.size

        # Set the percentage of pixels that should contain noise
        #noise_percentage = 0.1  # Setting to 10%

        # Determine the size of the noise based on the noise precentage
        noise_size = int(noise_percentage*img_size)

        # Randomly select indices for adding noise.
        random_indices = np.random.choice(img_size, noise_size)

        # Create a copy of the original image that serves as a template for the noised image.
        img_noised = img_gray.copy()

        # Create a noise list with random placements of min and max values of the image pixels.
        #noise = np.random.choice([img_gray.min(), img_gray.max()], noise_size)
        noise = np.random.choice([-10, 10], noise_size)

        # Replace the values of the templated noised image at random indices with the noise, to obtain the final noised image.
        img_noised.flat[random_indices] += noise
        
        log.info('adding image noise')
        return img_noised

    def init_image(self, img_type = 1):
        # create some images for test
        w,h             = self.frame_size
        if img_type == 1: # /
            
            self.img        = np.tile(np.linspace(100, 300, w), (h,1))

        elif img_type == 2: # /|/

            self.img        = np.tile(np.linspace(100, 200, int(w/2)), (h,2))
         
        elif img_type == 3: # |_|

            self.img        = np.tile(np.linspace(100, 200, h).reshape((-1,1)), (1,w)) 
        
        elif img_type == 4: # /\

            self.img        = np.tile(np.hstack((np.linspace(300, 500, w>>1),np.linspace(500, 300, w>>1))), (h,1))        

        elif img_type == 5: # dome

            x,y             = np.meshgrid(np.arange(w),np.arange(h))   
            self.img        = (np.abs(x - w/2) + np.abs(y - h/2))/10 + 200 # less slope

        elif img_type == 6: # sphere

            x,y             = np.meshgrid(np.arange(w),np.arange(h))   
            self.img        = np.sqrt((x - w/2)**2 + (y - h/2)**2)/10 + 200 # less slope   

        elif img_type == 7: # stair

            x,y             = np.meshgrid(np.arange(w),np.arange(h))   
            self.img        = (np.sign(x - w/2) + np.sign(y - h/2))*5 + 200 # less slope     


        elif img_type == 8: # corner

            x,y             = np.meshgrid(np.arange(w),np.arange(h))   
            self.img        = np.ones((h,w))*250
            img_bool        = np.logical_and((x - w/2) < 0, (y - h/2) < 0)
            self.img[img_bool] = 230 # quarter                            

        elif img_type == 10: # flat

            self.img        = np.ones((h,w))*500             

        elif img_type == 11:
            "chess board"
            fname           = r"C:\Users\udubin\Documents\Code\opencv-4x\samples\data\left04.jpg"
            self.img        = cv.imread(fname)

        elif img_type == 12:
            self.img = cv.imread('image_scl_001.png', cv.IMREAD_GRAYSCALE)
            #self.img = cv.resize(self.img , dsize = self.frame_size) 
            
        elif img_type == 13:
            self.img = cv.imread(r"wrappers\python\applications\planes\data\image_ddd_000.png", cv.IMREAD_GRAYSCALE)
            #self.img = cv.resize(self.img , dsize = self.frame_size) 

        elif img_type == 21:
            self.img = cv.imread(r"C:\Data\Depth\Plane\image_scl_000.png", cv.IMREAD_GRAYSCALE)  
            #self.img = cv.resize(self.img , dsize = self.frame_size)                                     
            
        #self.img        = np.uint8(self.img) 

        self.img = self.add_noise(self.img, 0)
        self.frame_size = self.img.shape[:2]      
        return self.img
      
    def init_roi(self, test_type = 1):
        "load the test case"
        roi = [0,0,self.frame_size[0],self.frame_size[1]]
        if test_type == 1:
            roi = [310,230,330,250] # xlu, ylu, xrb, yrb
        elif test_type == 2:
            roi = [300,220,340,260] # xlu, ylu, xrb, yrb
        elif test_type == 3:
            roi = [280,200,360,280] # xlu, ylu, xrb, yrb            
        elif test_type == 4:
            roi = [220,140,420,340] # xlu, ylu, xrb, yrb      
        elif test_type == 4:
            roi = [200,120,440,360] # xlu, ylu, xrb, yrb            
        return roi    
    
    def test_image(self):
        "test single image depth"
        img  = self.init_image(1)
        roi  = self.init_roi(1)      

#%% Adds display functionality to the PlaneDetector
class PlaneDetectorDisplay(PlaneDetector):
    def __init__(self, detect_type='p',image_size = (1280,720)):
        super().__init__(detect_type, image_size=image_size)
        self.detect_type    = detect_type
        self.frame_size     = image_size
        self.img            = None
        self.show_dict     = {}       # show figures in 3D

    def show_image_data(self, frame, display_mode = 1):
        "converts image data to 3d color"

        if display_mode == 1: # left
            img = frame[:,:,0]
        elif display_mode == 2: # right
            img = frame[:,:,1]
        elif display_mode == 3:
            img  = cv.convertScaleAbs(frame[:,:,2], alpha=0.1)            
        else: # depth
            img  = cv.convertScaleAbs(frame[:,:,2], alpha=0.03)

        vis     = np.uint8(img)
        vis     = cv.cvtColor(vis, cv.COLOR_GRAY2BGR)
        return vis  

    def show_data(self, frame = None, ttl = 'Depth'):
        "draw relevant image data"
            
        if frame is None :
            log.info('No images found')
            return False
        
        if len(frame.shape) > 2 and frame.shape[2]==2: # extract 3 images
            img_show    = np.concatenate((frame[:,:,0], frame[:,:,1]), axis = 1)
            img_show    = np.uint8(img_show)

        elif len(frame.shape) == 2 and frame.dtype == 'uint16':
            img_show    = cv.convertScaleAbs(frame, alpha=0.03)
            img_show    = np.uint8(img_show)
        else:   
            img_show    = np.uint8(frame)

        while img_show.shape[1] > 2000:
            img_show    = cv.resize(img_show, (img_show.shape[1]>>1,img_show.shape[0]>>1), interpolation=cv.INTER_LINEAR)

        while img_show.shape[0] < 300:
            img_show    = cv.resize(img_show, (img_show.shape[1]<<1,img_show.shape[0]<<1), interpolation=cv.INTER_LINEAR)            

        cv.imshow(ttl + ' (q-Quit)', img_show)
        ch  = cv.waitKey(10)
        ret = ch != ord('q')
        return ret          

    def show_image_with_axis(self, img, poses = []):
        "draw results : axis on the image. poses are list of 6D vectors"
        axis_number = len(poses)
        if axis_number < 1:
            log.error('No poses found')
            
        # deal with black and white
        img_show = np.uint8(img) #.copy()
        if len(img.shape) < 3:
            img_show = cv.applyColorMap(img_show, cv.COLORMAP_JET)
         
        for k in range(axis_number):
            
            euler_angles    = poses[k][3:] # orientation in degrees
            rvec            = Rot.from_euler('xyz',euler_angles[:3], degrees=True).as_rotvec()
            tvec            = np.array(poses[k][:3]) #np.array(, dtype = np.float32).reshape(rvec.shape) # center of the patch
            img_show        = draw_axis(img_show, rvec, tvec, self.cam_matrix, self.cam_distort, len = 10)

        cv.imshow('Image & Axis', img_show)
        log.info('show done')
        ch = cv.waitKey()

    # def show_image_with_roi_normals(self, img = None):
    #     "draw results : show normals at each point of ROI"
    #     if img is None:
    #         log.error('No image found')
    #         return img
    #     if self.rect is None:  # roi
    #         log.error('No ROI found')
    #         return img
    #     if self.img_roi_normal is None:
    #         log.error('No normals found')
    #         return img

    #     # deal with black and white
    #     img_show = np.uint8(img) #.copy()
    #     if len(img.shape) < 3:
    #         #img_show = cv.applyColorMap(img_show, cv.COLORMAP_JET)
    #         img_show = np.tile(img_show[:,:,np.newaxis], (1,1,3))

    #     # scale normals to fir RGB color space
    #     normals         = (self.img_roi_normal + 1)*127.5
    #     x0,y0,x1,y1     = self.rect
    #     img_show[y0:y1,x0:x1, :] = np.uint8(normals)
         
    #     #cv.imshow('Image & Normals', img_show)
    #     #log.info('show done')
    #     #ch = cv.waitKey(0) 
    #     return img_show       

    def show_image_with_rois(self, img, roi_params_ret = []):
        "draw results by projecting ROIs on image"

        axis_number = len(roi_params_ret)
        if axis_number < 1:
            print('No poses found')
            
        # deal with black and white
        img_show = np.uint8(img) #.copy()
        if len(img.shape) < 3:
            img_show = cv.applyColorMap(img_show, cv.COLORMAP_JET)
         
        for roi_p in roi_params_ret:

            pose    = self.convert_roi_params_to_pose(roi_p)            
            
            avec    = pose[3:6] # orientation in degrees
            levl    = pose[6]   # level
            #R       = eulerAnglesToRotationMatrix(avec)
            R       = Rot.from_euler('zyx',avec, degrees = True).as_matrix()
            rvec, _ = cv.Rodrigues(R)
            tvec    = np.array(pose[:3], dtype = np.float32).reshape(rvec.shape) # center of the patch
            img_show= draw_axis(img_show, rvec, tvec, self.cam_matrix, self.cam_distort, len = levl)

        cv.imshow('Image & Axis', img_show)
        log.info('show done')
        ch = cv.waitKey()

    def show_points_3d_with_normal(self, img3d, pose = None):
        "display in 3D"
        fig = plt.figure()
        ax  = fig.add_subplot(projection='3d')

        #xs,ys,zs       = img3d[:,:,0].reshape((-1,1)), img3d[:,:,1].reshape((-1,1)), img3d[:,:,2].reshape((-1,1))
        
        xs,ys,zs       = img3d[:,0].reshape((-1,1)), img3d[:,1].reshape((-1,1)), img3d[:,2].reshape((-1,1))
        ax.scatter(xs, ys, zs, marker='.')
        
        if pose is not None:
            pose       = pose.flatten()
            vnorm      = pose[3:6].flatten()*10
            xa, ya, za = [pose[0], pose[0]+vnorm[0]], [pose[1], pose[1]+vnorm[1]], [pose[2], pose[2]+vnorm[2]]
            ax.plot(xa, ya, za, 'r', label='Normal')


        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        ax.set_aspect('equal', 'box')
        plt.show()

    def show_rois_3d_with_normals(self, roi_params_ret = [], roi_init = None):
        "display in 3D each ROI region with split"
        
        if len(roi_params_ret) < 1:
            log.info('roi_params_ret is empty')
            return

        # extract the initial ROI - to make the show more compact
        roi_init       = [0,0,self.frame_size[1], self.frame_size[0]] if roi_init is None else roi_init
        x0,y0,x1,y1    = roi_init

        if self.img_xyz is None:
            log.info('Need init')
            return      

        img3d          = self.img_xyz[y0:y1,x0:x1,:] 
        xs,ys,zs       = img3d[:,:,0].reshape((-1,1)), img3d[:,:,1].reshape((-1,1)), img3d[:,:,2].reshape((-1,1))

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(xs, ys, zs, marker='.')
        
        for roi_p in roi_params_ret:
            pose       = self.convert_roi_params_to_pose(roi_p)
            pose       = pose.flatten()
            # R          = Rot.from_euler('zyx',pose[3:6],degrees=True).as_matrix()
            # vnorm      = R[:,2]*pose[6]
            vnorm      = pose[3:6]*pose[6]
            #log.info(str(vnorm))
            xa, ya, za = [pose[0], pose[0]+vnorm[0]], [pose[1], pose[1]+vnorm[1]], [pose[2], pose[2]+vnorm[2]]
            ax.plot(xa, ya, za, 'r', label='Normal')


        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        ax.set_aspect('equal', 'box')
        plt.show() #block=False)  

    def show_3d_point_cloud(self):
        "displays point cloud in real time"
        if self.rect is None or self.rect_xyz is None:
            return 
        X = self.rect_xyz[:,:3]
        if not 'line' in self.show_dict : #len(self.show_dict) < 1:

            fig_num     = int(self.rect[0]+self.rect[1])
            fig         = plt.figure(fig_num)
            plt.clf() 
            #fig.canvas.set_window_title('3D Scene')
            try:
                ax = fig.gca(projection='3d')
            except:
                ax = fig.add_subplot(projection = '3d')
            fig.tight_layout()
            fig.suptitle(f'ROI : {self.rect[0]},{self.rect[1]}')
            ax.set_proj_type('ortho')
            #lineGray,      = ax.plot3D(X[:,0], X[:,1], X[:,2], color='k')
            lineGray    = ax.scatter(X[:,0], X[:,1], X[:,2])

            ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
            ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
            ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))             

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.invert_yaxis()
            ax.view_init(elev=-70, azim=-90)
         
            lims = 500
            ax.set_xlim(X[:,0].min()*0.8, X[:,0].max()*1.2)
            ax.set_ylim(X[:,1].min()*0.8, X[:,1].max()*1.2)
            ax.set_zlim(X[:,2].min()*0.9, X[:,2].max()*1.2)
            plt.ion()
            plt.show(block = False)
            self.show_dict = {'fig':fig, 'ax':ax, 'line':lineGray}
        else:
            #self.show_dict['line'].set_data(X[:,0], X[:,1])
            #self.show_dict['line'].set_3d_properties(X[:,2])
            self.show_dict['line']._offsets3d = (X[:,0], X[:,1], X[:,2])
            #self.show_dict['ax'].set_ylim(low_limit, high_limit)
        
        self.show_dict['fig'].canvas.draw_idle()
        self.show_dict['fig'].canvas.flush_events()
        return              

    def show_axis(self, vis):
        "draw axis after plane estimation"
        if self.plane_params is None:
            return vis
        
        #rvec = self.plane_params/np.sum(self.plane_params**2) # normalize
        rvec = self.convert_plane_params(self.plane_params)
        #rvec = self.convert_plane_to_rvec(self.plane_params)
        
        tvec = self.plane_center
        vis  = draw_axis(vis, rvec, tvec, self.cam_matrix, self.cam_distort, len = 50)
        return vis
    
    def show_text(self, vis):
        "draw text plane estimation"
        err_mean, err_std = self.img_mean, self.img_std
        if err_mean is None:
            return vis
        
        if self.rect is None:
            return vis
        
        x0, y0, x1, y1 = self.rect
        txt = f'{self.detect_type}:{err_mean:.2f}:{err_std:.3f}'
        #if self.detect_type == 'F':
        #    txt = f'{self.detect_type}:{self.img_fill:.2f} %'
        vis = draw_str(vis,(x0,y0-10),txt)

        return vis 

    def show_rect_and_text(self, vis):
        "draw axis after plane estimation"
        err_mean, err_std = self.img_mean, self.img_std
        if err_mean is None:
            return vis
        
        if self.rect is None:
            return vis
        
        x0, y0, x1, y1 = self.rect
        clr = (0, 0, 0) if vis[y0:y1,x0:x1].mean() > 128 else (240,240,240)
        vis = cv.rectangle(vis, (x0, y0), (x1, y1), clr, 2)
        txt = f'{self.detect_type}:{err_mean:.2f}-{err_std:.3f}'
        if self.detect_type == 'F':
            txt = f'{self.detect_type}:{self.img_fill:.2f} %'
        vis = draw_str(vis,(x0,y0-10),txt)

        return vis 

    def show_rect_and_axis_projected(self, vis):
        "projects rectangle on the plane"
        if self.rect is None:
            return vis
        if self.plane_params is None:
            return vis
        
        rvec = self.convert_plane_params(self.plane_params)
        tvec = self.plane_center

        vis  = draw_axis(vis, rvec, tvec, self.cam_matrix, self.cam_distort, len = 50)        
        vis  = draw_polygon(vis, rvec, tvec, self.cam_matrix, self.cam_distort, self.rect_3d)
    
        return vis 

    def show_mask(self, img):
        "draw image mask"

        # deal with black and white
        img_show = np.uint8(img) #.copy()
        if len(img.shape) < 3:
            img_show = cv.applyColorMap(img_show, cv.COLORMAP_JET)

        if not np.all(self.img_mask.shape[:2] == img_show.shape[:2]):
            log.error('mask and image size are not equal')
            return img_show
        
        img_show[self.img_mask > 0.75] = self.color_mask
        return img_show
    
    def show_polygon(self, img):
        "shows polygon on top of the image"
        # Define polygon vertices (e.g., a triangle)
        if self.polygon is None:
            return img
        
        polygon = np.array(self.polygon, np.int32)

        # Reshape for OpenCV (required shape: (n_points, 1, 2))
        polygon = polygon.reshape((-1, 1, 2))

        # Draw the polygon on the image
        img     = cv.polylines(img, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

        return img    
    
    def show_image_colored_by_normals(self, img = None):
        "draw results : show normals at each point of ROI"
        if img is None:
            log.error('No image found')
            return img
        if self.rect is None:  # roi
            log.error('No ROI found')
            return img
        if self.img_roi_normal is None:
            log.error('No normals found')
            return img

        # deal with black and white
        img_show    = np.uint8(img).copy()
        if len(img.shape) < 3:
            #img_show = cv.applyColorMap(img_show, cv.COLORMAP_JET)
            img_show = np.tile(img_show[:,:,np.newaxis], (1,1,3))

        # scale normals to fir RGB color space
        normals                  = (self.img_roi_normal + 1)*127.5
        x0,y0,x1,y1              = self.rect
        #img_show[y0:y1,x0:x1, :] = np.uint8(normals)
        img_show[y0:y1,x0:x1, :] = cv.addWeighted(img_show[y0:y1,x0:x1, :], 0.2, np.uint8(normals), 0.8, 0)

        # make black outside roi or non valid
        non_valid_pixels = img[:,:,2] < 1   
        img_show[non_valid_pixels,0] = 0     
        img_show[non_valid_pixels,1] = 0     
        img_show[non_valid_pixels,2] = 0     
         
        #cv.imshow('Image & Normals', img_show)
        #log.info('show done')
        #ch = cv.waitKey(0) 
        return img_show      

    def show_scene(self, vis):
        "draw ROI and Info"

        #vis = self.show_rect_and_text(vis)
        #vis = self.show_axis(vis)

        vis = self.show_mask(vis)

        #vis = self.show_image_colored_by_normals(vis)        
        vis = self.show_rect_and_axis_projected(vis)
        vis = self.show_text(vis)



        return vis

        

# ----------------------
#%% Tests
class TestPlaneDetector(unittest.TestCase):

    def test_image_show(self):
        "checking image show"
        d       = DataGen()
        img     = d.init_image(1)
        p       = PlaneDetectorDisplay()
        poses   = [[0,0,100,0,0,45,10]]
        p.show_image_with_axis(img,poses)
        self.assertFalse(d.img is None)    

    def test_init_img3d(self):
        "XYZ point cloud structure init"
        d       = DataGen()
        img     = d.init_image(1)
        p       = PlaneDetectorDisplay()
        isOk    = p.init_image(img)
        img3d   = p.init_img3d()
        self.assertFalse(img3d is None)    

    def test_compute_img3d(self):
        "XYZ point cloud structure init and compute"
        d       = DataGen()
        img     = d.init_image(1)        
        p       = PlaneDetectorDisplay()
        img3d   = p.init_img3d(img)
        imgXYZ  = p.compute_img3d(img)
        self.assertFalse(imgXYZ is None)     

    def test_show_img3d(self):
        "XYZ point cloud structure init and compute"
        d       = DataGen()
        img     = d.init_image(1)        
        p       = PlaneDetectorDisplay()
        img3d   = p.init_img3d(img)
        imgXYZ  = p.compute_img3d(img)
        roi     = p.init_roi(1)
        x0,y0,x1,y1 = roi
        roiXYZ    = imgXYZ[y0:y1,x0:x1,:]
        p.show_points_3d_with_normal(roiXYZ)
        self.assertFalse(imgXYZ is None)  

    def test_convert_roi_to_points(self):
        "computes 3d points of the ROI"
        im_size     = (640,480)
        d           = DataGen(img_size=im_size)
        img         = d.init_image(1)        
        p           = PlaneDetectorDisplay(image_size=im_size)
        roi         = p.init_roi(1)
        rect3d      = p.convert_roi_to_points(img, roi_rect=roi)
        self.assertTrue(rect3d.shape[0] > 4)
                     
    def test_fit_plane_svd(self):
        "computes normal to the ROI"
        d           = DataGen()
        img         = d.init_image(5)        
        p           = PlaneDetectorDisplay()
        roi         = p.init_roi(4)
        img_roi     = p.preprocess(img)
        roim,rois   = p.fit_plane_svd(img_roi)
        pose        = p.convert_plane_params_to_pose()
        p.show_image_with_axis(img, pose)
        p.show_points_3d_with_normal(p.matrix_xyz, pose)
        self.assertTrue(pose[0][2] > 0.01)         

    def test_fit_plane_depth_image(self):
        "computes normal to the ROI"
        d           = DataGen()
        img         = d.init_image(13)        
        p           = PlaneDetectorDisplay()
        roi         = p.init_roi(4)
        img_roi     = p.preprocess(img)
        roim,rois   = p.fit_plane_svd(img_roi)
        pose        = p.convert_plane_params_to_pose()
        p.show_image_with_axis(img, pose)
        p.show_points_3d_with_normal(p.rect_xyz, pose)
        self.assertTrue(pose[0][2] > 0.01)  

    def test_fit_plane_with_outliers(self):
        "computes normal to the ROI"
        d           = DataGen()
        img         = d.init_image(13)        
        p           = PlaneDetectorDisplay()
        roi         = p.init_roi(4)
        img_roi     = p.preprocess(img)
        roim,rois   = p.fit_plane_with_outliers(img_roi)
        pose        = p.convert_plane_params_to_pose()
        p.show_image_with_axis(img, pose)
        p.show_points_3d_with_normal(p.rect_xyz, pose)
        self.assertTrue(pose[0][2] > 0.01)  

    def test_fit_plane_ransac(self):
        "computes with ransac"
        d           = DataGen()
        img         = d.init_image(6)        
        p           = PlaneDetectorDisplay()
        roi         = p.init_roi(4)
        img_roi     = p.preprocess(img)
        roim,rois   = p.fit_plane_ransac(img_roi)
        pose        = p.convert_plane_params_to_pose()
        p.show_image_with_axis(img, pose)
        p.show_points_3d_with_normal(p.rect_xyz, pose)
        self.assertTrue(pose[0][2] > 0.01)  

    def test_fit_plane_using_gradients(self):
        "computes normal of the ROI using gradients and then cross product"
        im_size     = (640,480)
        d           = DataGen(img_size=im_size)
        img         = d.init_image(1)     # corner     
        p           = PlaneDetectorDisplay(image_size=im_size)        
        roi         = p.init_roi(13)      # image center
        roim,rois   = p.fit_plane_using_gradients(img, roi)
        pose        = p.convert_plane_params_to_pose()
        p.show_image_colored_by_normals(img)
        p.show_image_with_axis(img, pose)
        p.show_points_3d_with_normal(p.rect_xyz, pose)
        self.assertTrue(pose[0][2] > 0.01)         

    def test_split_roi(self):
        "computes ROIS and splits if needed"
        p       = PlaneDetector()
        p.MIN_STD_ERROR = 0.1
        img     = p.init_image(13)
        roi     = p.init_roi(4)
        img3d   = p.init_img3d(img)
        imgXYZ  = p.compute_img3d(img)
        roi_list= p.fit_and_split_roi_recursively(roi)
        p.show_rois_3d_with_normals(roi_list, roi)
        p.show_image_with_rois(p.img, roi_list)

        for roi_s in roi_list:
            self.assertFalse(roi_s['error'] > 0.01) 

    def test_plane_fit(self):
        "plane fit data"
        d               = DataSourceMovie()
        srcid           = 422            # 421,422, 423-ok
        ret             = d.init_video(srcid)
        p               = PlaneDetectorDisplay('P') #estimator_type=self.estim_type, estimator_id=estim_ind)
        roi             = p.init_roi(21)  # center image
        while ret:
            ret,img     = d.get_data() 
            if not ret: break        
            retp        = p.find_planes(img)
            vis         = p.show_image_data(img, display_mode=3)
            vis         = p.show_scene(vis)
            ret         = p.show_data(vis) & ret
        d.finish()
        self.assertFalse(ret)     

    def test_plane_fit_weighted(self):
        "plane fit data"
        d               = DataSourceMovie()
        srcid           = 422            # 421,422, 423-ok
        ret             = d.init_video(srcid)
        p               = PlaneDetectorDisplay('W') #estimator_type=self.estim_type, estimator_id=estim_ind)
        roi             = p.init_roi(21)  # center image
        while ret:
            ret,img     = d.get_data()  
            if not ret: break      
            retp        = p.find_planes(img)
            vis         = p.show_image_data(img, display_mode=1)
            vis         = p.show_scene(vis)
            ret         = p.show_data(vis) & ret
        d.finish()
        self.assertFalse(ret) 

    def test_plane_fit_outliers(self):
        "plane fit data"
        d               = DataSourceMovie()
        srcid           = 421            # 421,422, 423-ok
        ret             = d.init_video(srcid)
        p               = PlaneDetectorDisplay('O') #estimator_type=self.estim_type, estimator_id=estim_ind)
        roi             = p.init_roi(41)  # center image
        while ret:
            ret,img     = d.get_data()  
            if not ret: break      
            retp        = p.find_planes(img)
            vis         = p.show_image_data(img, display_mode=3)
            vis         = p.show_scene(vis)
            ret         = p.show_data(vis) & ret
        d.finish()
        self.assertFalse(ret) 

    def test_plane_fit_ransac(self):
        "plane fit data"
        d               = DataSourceMovie()
        srcid           = 422            # 421,422, 423-ok
        ret             = d.init_video(srcid)
        p               = PlaneDetectorDisplay('R') #estimator_type=self.estim_type, estimator_id=estim_ind)
        roi             = p.init_roi(21)  # center image
        while ret:
            ret,img     = d.get_data() 
            if not ret: break        
            retp        = p.find_planes(img)
            vis         = p.show_image_data(img, display_mode=3)
            vis         = p.show_scene(vis)
            ret         = p.show_data(vis) & ret
        d.finish()
        self.assertFalse(ret)   

    def test_plane_fit_weighted_4d(self):
        "plane fit data"
        d               = DataSourceMovie()
        srcid           = 422            # 421,422, 423-ok
        ret             = d.init_video(srcid)
        p               = PlaneDetectorDisplay('T') #estimator_type=self.estim_type, estimator_id=estim_ind)
        roi             = p.init_roi(53)  # center image
        while ret:
            ret,img     = d.get_data() 
            if not ret: break        
            retp        = p.find_planes(img)
            vis         = p.show_image_data(img, display_mode=3)
            vis         = p.show_scene(vis)
            ret         = p.show_data(vis) & ret
        d.finish()
        self.assertFalse(ret)   

    def test_plane_fit_using_gradients(self):
        "plane fit data"
        d               = DataSourceMovie()
        srcid           = 422            # 421,422, 423-ok
        ret             = d.init_video(srcid)
        p               = PlaneDetectorDisplay('G') #estimator_type=self.estim_type, estimator_id=estim_ind)
        roi             = p.init_roi(33)  # center image
        while ret:
            ret,img     = d.get_data() 
            if not ret: break        
            retp        = p.find_planes(img)
            vis         = p.show_image_data(img, display_mode=3)
            vis         = p.show_scene(vis)
            ret         = p.show_data(vis) & ret
        d.finish()
        self.assertFalse(ret)               

    def test_multi_plane_fit(self):
        "plane fit data multiple ROIs"
        d               = DataSourceMovie()
        srcid           = 422            # 421,422, 423-ok
        ret             = d.init_video(srcid)
        roi_types       = [21,22,23,31,32,33,52,53]
        pm              = []
        for rt in roi_types:
            p           = PlaneDetectorDisplay('T') #estimator_type=self.estim_type, estimator_id=estim_ind)
            roi         = p.init_roi(rt)  # center image
            pm.append(p)

        while ret:
            ret,img     = d.get_data()
            if not ret: break
            vis         = pm[0].show_image_data(img, display_mode=3)
            for p in pm:         
                retp        = p.find_planes(img)
                vis         = p.show_scene(vis)

            ret         = p.show_data(vis) 

        d.finish()
        self.assertFalse(ret)   

    def test_grid_plane_fit(self):
        "plane fit data multiple ROIs in grid"
        d               = DataSourceMovie()
        srcid           = 422            # 421,422, 423-ok
        ret             = d.init_video(srcid)
        
        "generate grid of trackers"
        nx, ny          = 10, 10
        w,h             = 1280, 720     
        dx, dy          = int(w/(nx+1)), int(h/(ny+1))
        wx,wy           = int(dx*0.4), int(dy*0.4)
        pm              = []
        for ix in range(nx):
            for iy in range(ny):
                x0, y0      = (ix+1)*dx, (iy+1)*dy
                #x1, y1      = min((ix+1)*dx, w-1), min((iy+1)*dy, h-1)
                rect        = (x0-wx,y0-wy,x0+wx,y0+wy)
                p           = PlaneDetectorDisplay('P') #estimator_type=self.estim_type, estimator_id=estim_ind)
                p.rect      = rect  # center image
                pm.append(p)                             

        while ret:
            ret,img     = d.get_data()
            if not ret: break
            vis         = pm[0].show_image_data(img, display_mode=3)
            for p in pm:         
                retp        = p.find_planes(img)
                vis         = p.show_scene(vis)

            ret         = p.show_data(vis) 

        d.finish()
        self.assertFalse(ret)        


    def test_plane_fit_with_show_3d(self):
        "plane fit data and show 3d data"
        d               = DataSourceMovie()
        srcid           = 421            # 421,422, 423-ok
        ret             = d.init_video(srcid)
        p               = PlaneDetectorDisplay('P') #estimator_type=self.estim_type, estimator_id=estim_ind)
        roi             = p.init_roi(41)  # center image
        while ret:
            ret,img     = d.get_data()
            if not ret: break           
            retp        = p.find_planes(img)
            vis         = p.show_image_data(img, display_mode=3)
            vis         = p.show_scene(vis)
            ret         = p.show_data(vis) & ret
            p.show_3d_point_cloud()
        d.finish()
        self.assertFalse(ret)      

    def test_plane_fit_colored_by_normals(self):
        "plane fit data and show 3d data"
        d               = DataSourceMovie()
        srcid           = 422           # 421,422, 423-ok
        ret             = d.init_video(srcid)
        p               = PlaneDetectorDisplay('G') #estimator_type=self.estim_type, estimator_id=estim_ind)
        roi             = p.init_roi(0)  # center image
        while ret:
            ret,img     = d.get_data()
            if not ret: break           
            retp        = p.find_planes(img)
            vis         = p.show_image_data(img, display_mode=1)
            vis         = p.show_image_colored_by_normals(vis)
            #vis         = p.show_scene(vis)
            ret         = p.show_data(vis) & ret
            #p.show_3d_point_cloud()
            
        d.finish()
        self.assertFalse(ret)      

    def test_plane_fit_with_grow(self):
        "plane fit data and show 3d data"
        d               = DataSourceMovie()
        srcid           = 422           # 421,422, 423-ok
        ret             = d.init_video(srcid)
        p               = PlaneDetectorDisplay('F') #estimator_type=self.estim_type, estimator_id=estim_ind)
        roi             = p.init_roi(21)  # center image
        while ret:
            ret,img     = d.get_data()
            if not ret: break           
            retp        = p.find_planes(img)
            vis         = p.show_image_data(img, display_mode=1)
            vis         = p.show_mask(vis)
            #vis         = p.show_scene(vis)
            ret         = p.show_data(vis) & ret
            #p.show_3d_point_cloud()
            
        d.finish()
        self.assertFalse(ret)       

    def test_plane_fit_with_project_on_image(self):
        "plane fit data and show 3d data"
        d               = DataSourceMovie()
        srcid           = 422           # 421,422, 423-ok
        ret             = d.init_video(srcid)
        p               = PlaneDetectorDisplay('S') #estimator_type=self.estim_type, estimator_id=estim_ind)
        roi             = p.init_roi(21)  # center image
        while ret:
            ret,img     = d.get_data()
            if not ret: break           
            retp        = p.find_planes(img)
            vis         = p.show_image_data(img, display_mode=1)
            vis         = p.show_mask(vis)
            #vis         = p.show_scene(vis)
            ret         = p.show_data(vis) & ret
            #p.show_3d_point_cloud()
            
        d.finish()
        self.assertFalse(ret) 

    def test_multi_plane_fit_with_project_on_image(self):
        "plane fit data multiple ROIs with image coloring per ROI mask"
        d               = DataSourceMovie()
        srcid           = 422            # 421,422, 423-ok
        ret             = d.init_video(srcid)
        roi_types       = [21,31,52,33] #22,23,31,32,33,52,53]
        pm              = []
        for rt in roi_types:
            p           = PlaneDetectorDisplay('S') #estimator_type=self.estim_type, estimator_id=estim_ind)
            roi         = p.init_roi(rt)  # center image
            pm.append(p)

        while ret:
            ret,img     = d.get_data()
            if not ret: break
            vis         = pm[0].show_image_data(img, display_mode=3)
            for p in pm:         
                retp        = p.find_planes(img)
                vis         = p.show_mask(vis)

            ret         = p.show_data(vis) 

        d.finish()
        self.assertFalse(ret)                                                            

# ----------------------
#%% Run Test
def RunTest():
    #unittest.main()
    #suite = unittest.TestSuite()
    suite = TestPlaneDetector()
    #suite.test_image_show() # ok
    #suite.test_init_img3d()  # ok
    #suite.test_compute_img3d() # ok
    #suite.test_show_img3d() # ok
    #suite.test_convert_roi_to_points()  # ok


    #suite.test_fit_plane_svd() # ok
    #suite.test_fit_plane_depth_image() #
    #suite.test_fit_plane_with_outliers() 
    #suite.test_fit_plane_ransac()  
    #suite.test_fit_plane_using_gradients()  # ok 
    #suite.test_split_roi() 

    #suite.test_plane_fit() # ok
    #suite.test_plane_fit_weighted() # ok
    #suite.test_plane_fit_outliers() # ok
    #suite.test_plane_fit_with_show_3d() # ok
    
    #suite.test_plane_fit_ransac() # ok
    #suite.test_plane_fit_weighted_4d() # ok
    #suite.test_multi_plane_fit() # ok
    #suite.test_plane_fit_using_gradients() # ok
    #suite.test_plane_fit_colored_by_normals() # ok
    #suite.test_plane_fit_with_grow() #
    #suite.test_grid_plane_fit() # ok
    #suite.test_plane_fit_with_project_on_image() # ok
    suite.test_multi_plane_fit_with_project_on_image() # ok
    

   
    # runner = unittest.TextTestRunner()
    # runner.run(suite)    

# ----------------------
#%% App
class PlaneApp:
    def __init__(self):
        self.cap            = RealSense() #
        self.cap.set_display_mode('d16')
        #self.cap.set_exposure(1000)
        self.frame          = None
        self.rect           = None
        self.paused         = False
        self.trackers       = []

        self.camera_bf      = self.cap.get_bf() # for depth conversion (should be in meter)
        self.args          = process_arguments()
        self.args.scale    = 0.5

        # init
        self.model        = foundation_stereo_algo_init(self.args)

        self.show_dict      = {} # hist show

        self.detect_type    = 'G'
        self.show_type      = 'depth' # left, depth
        self.win_name       = 'Plane Detector (q-quit, c-clear, a,r,p,o,g,f,t,s,n,w 0-depth,1-left,2-right)'

        cv.namedWindow(self.win_name )
        self.rect_sel       = RectSelector(self.win_name , self.on_rect)
        self.run()

    def on_rect(self, rect):
        "remember ROI defined by user"
        #self.define_roi(self.frame, rect)
        tracker             = PlaneDetectorDisplay() #estimator_type=self.estim_type, estimator_id=estim_ind)
        tracker.rect        = rect
        tracker.detect_type = self.detect_type
        self.trackers.append(tracker)        
        log.info(f'Adding plane estimator at  : {rect}') 

    def generate_grid_trackers(self):
        "generate grid of trackers"
        nx, ny          = 10, 10
        w,h             = self.cap.frame_size     
        dx, dy          = int(w/nx), int(h/ny)
        for ix in range(nx):
            for iy in range(ny):
                x0, y0 = ix*dx, iy*dy
                x1, y1 = min((ix+1)*dx, w-1), min((iy+1)*dy, h-1)
                rect    = (x0,y0,x1,y1)
                self.on_rect(rect)

    def process_image(self, img_depth):
        "makes measurements"
        for tracker in self.trackers:
            tracker.find_planes(img_depth) 

    def show_scene(self, frame, img_depth_fs):
        "draw ROI and Info"
        if self.show_type == 'left':
            vis     = frame[:,:,0].astype(np.uint8)
        elif self.show_type == 'right':
            vis     = frame[:,:,1].astype(np.uint8)   
        elif self.show_type == 'rs':
            vis     = cv.convertScaleAbs(frame[:,:,2], alpha=0.1).astype(np.uint8)                     
        else:
            vis     = cv.convertScaleAbs(img_depth_fs, alpha=0.1).astype(np.uint8)

        vis     = cv.cvtColor(vis, cv.COLOR_GRAY2BGR)
        self.rect_sel.draw(vis)

        for tracker in self.trackers:
            vis = tracker.show_scene(vis) 

        return vis 
    
    def show_histogram(self, img):
        "show roi histgram"
        if self.rect is None:
            #print('define ROI')
            return 0
        
        x0, y0, x1, y1 = self.rect
        img_roi = img[y0:y1,x0:x1].astype(np.float32)
        # Compute histogram
        hist, bins = np.histogram(img_roi.flatten(), bins=1024, range=[0, 2**15])

        if not 'fig' in self.show_dict : #len(self.show_dict) < 1:
            fig, ax = plt.subplots()
            fig.set_size_inches([24, 16])
            ax.set_title('Histogram (Depth)')
            ax.set_xlabel('Bin')
            ax.set_ylabel('Frequency')
            lineGray, = ax.plot(bins[:-1], hist, c='k', lw=3)
            ax.set_xlim(bins[0], bins[-1])
            ax.set_ylim(0, max(hist)+10)
            plt.ion()
            #plt.show()

            self.show_dict = {'fig':fig, 'ax':ax, 'line':lineGray}
        else:
            self.show_dict['line'].set_ydata(hist)
        
        self.show_dict['fig'].canvas.draw()
        return    

    def run(self):
        while True:
            playing = not self.paused and not self.rect_sel.dragging
            if playing or self.frame is None:
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.frame = frame.copy()

            # depth
            img_left, img_right, img_depth_rs = self.frame[:,:,0], self.frame[:,:,1], self.frame[:,:,2]    

            img_disparity   = foundation_stereo_algo(self.args, self.model, img_left, img_right)
            img_depth_fs    = convert_disparity_to_depth(self.camera_bf, img_disparity)

            
            #self.statistics(frame)
            self.process_image(img_depth_fs)

            vis     = self.show_scene(frame, img_depth_fs)
            cv.imshow(self.win_name , vis)
            ch = cv.waitKey(1)
            if ch == ord(' '):
                self.paused = not self.paused
            elif ch == ord('a'):
                self.detect_type = 'A' 
                log.info(f'Detect type : {self.detect_type}')
            elif ch == ord('r'):
                self.detect_type = 'R'  
                log.info(f'Detect Ransac : {self.detect_type}')
            elif ch == ord('p'):
                self.detect_type = 'P'  
                log.info(f'Detect svd : {self.detect_type}')
            elif ch == ord('o'):
                self.detect_type = 'O'  
                log.info(f'Detect with outliers : {self.detect_type}') 
            elif ch == ord('g'):
                self.detect_type = 'G'    
                log.info(f'Detect gradients : {self.detect_type}')   
            elif ch == ord('f'):
                self.detect_type = 'F'    
                log.info(f'Detect fit and grow : {self.detect_type}')     
            elif ch == ord('w'):
                self.detect_type = 'W'    
                log.info(f'Detect weighted svd : {self.detect_type}')     
            elif ch == ord('s'):
                self.detect_type = 'S'    
                log.info(f'Detect entire image : {self.detect_type}')    
            elif ch == ord('n'):
                self.detect_type = 'N'    
                log.info(f'Detect entire image using normals : {self.detect_type}')                                              
            elif ch == ord('t'):
                self.detect_type = 'T'    
                log.info(f'Detect weighted svd 4d : {self.detect_type}')                              
            elif ch == ord('0'):
                self.show_type = 'rs'      
                log.info(f'Show type : {self.show_type}')                               
            elif ch == ord('1'):
                self.show_type = 'left' 
                log.info(f'Show type : {self.show_type}')   
            elif ch == ord('2'):
                self.show_type = 'right'   
                log.info(f'Show type : {self.show_type}')   
            elif ch == ord('3'):
                self.show_type = 'fs'   
                log.info(f'Show type : {self.show_type}')                  
            elif ch == ord('m'):
                self.generate_grid_trackers()                                             
            elif ch == ord('c'):
                if len(self.trackers) > 0:
                    t = self.trackers.pop()
            elif ch == 27 or ch == ord('q'):
                break              


if __name__ == '__main__':
    #print(__doc__)

    #RunTest()
    PlaneApp()



