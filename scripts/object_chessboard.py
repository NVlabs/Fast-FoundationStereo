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
import matplotlib.pyplot as plt

import sys, os
current_dir = os.path.dirname(__file__)
parent_dir  = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

 # importing common Use modules 
#from src.logger                     import log
import logging as log


# --------------------------------
#%% ObjectManager - manages the entire experiment
class ObjectChessboard:

    def __init__(self, config = None):

        # params
        self.config             = config
        self.name               = 'chessboard'
        self.frame              = []
        self.resolution         = (1280,720)
        self.square_size         = 21.8                  # size in mm of the pattern square         
        self.debug_on            = True
        self.pattern_size       = (9,6)
        self._rt_plot            = None
        
        
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
    
    def get_grid_points(self, grid_size=1.0):
        # prepare grid points with step size of 1 mm, 
        # like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        scale_factor = self.square_size / grid_size
        a = int(self.pattern_size[0]*scale_factor)
        b = int(self.pattern_size[1]*scale_factor)
        s = grid_size # 21.8 # 21mm, but i want the units to be in meters
         
        grid_corners        = np.zeros((b*a,3), np.float32)
        grid_corners[:,:2]  = np.mgrid[0:a,0:b].T.reshape(-1,2)*s 
        return grid_corners    

    def get_image_points(self, img):
        "detect corners in the image and return their coordinates"

        # reduce size
        if len(img.shape) > 2: 
            gray  = cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_BGR2GRAY)
        else:
            gray = img.astype(np.uint8)

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

    def estimate_camera_pose(self, img, camera_matrix, dist_coeffs=None):
        """Detect chessboard and estimate camera pose with solvePnP.

        Args:
            img: BGR image.
            camera_matrix: Intrinsic matrix (3x3).
            dist_coeffs: Distortion coefficients (optional).

        Returns:
            dict: {
                success (bool), reason (str), image_points, object_points,
                rvec, tvec, rotation_matrix, camera_position
            }
            camera_position is in chessboard coordinates, same unit as square_size.
        """
        detection = self.detect(img)
        if not detection.get('success', False):
            return {
                'success': False,
                'reason': detection.get('reason', 'chessboard detection failed'),
                'image_points': detection.get('image_points', []),
                'object_points': detection.get('object_points', self.get_object_points()),
            }

        obj_points = detection['object_points'].astype(np.float32)
        img_points = np.asarray(detection['image_points'], dtype=np.float32)

        if img_points.ndim == 3 and img_points.shape[1] == 1:
            img_points = img_points.reshape(-1, 2)

        cam_mtx = np.asarray(camera_matrix, dtype=np.float32)
        dist = None if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float32)

        ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, cam_mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return {
                'success': False,
                'reason': 'solvePnP failed',
                'image_points': img_points,
                'object_points': obj_points,
            }

        rot_mtx, _ = cv2.Rodrigues(rvec)
        # Camera center in object/chessboard coordinates: C = -R^T * t
        camera_position = -rot_mtx.T @ tvec

        return {
            'success': True,
            'image_points': img_points,
            'object_points': obj_points,
            'rvec': rvec,
            'tvec': tvec,
            'rotation_matrix': rot_mtx,
            'camera_position': camera_position.reshape(3),
        }

    def estimate_board_pose_in_camera(self, img, camera_matrix, dist_coeffs=None):
        """Detect chessboard and estimate the board pose in the camera coordinate system.

        solvePnP returns rvec/tvec that transform points from object (board) space into
        camera space, so tvec is already the board origin expressed in the camera frame,
        and rot_mtx columns are the board X/Y/Z axes expressed in the camera frame.

        Args:
            img: grayscale or BGR image.
            camera_matrix: Intrinsic matrix (3x3).
            dist_coeffs: Distortion coefficients (optional).

        Returns:
            dict on success:
                success          : bool
                image_points     : (N, 2) detected corners in the image
                object_points    : (N, 3) 3-D corners in board frame
                rvec             : (3, 1) rotation vector (board → camera)
                tvec             : (3, 1) translation vector (board → camera)
                rotation_matrix  : (3, 3) rotation matrix R (board → camera)
                board_position   : (3,)  position of the board origin in camera frame (= tvec)
                board_center     : (3,)  position of the board geometric centre in camera frame
                board_x_axis     : (3,)  board X axis expressed in camera frame
                board_y_axis     : (3,)  board Y axis expressed in camera frame
                board_z_axis     : (3,)  board normal expressed in camera frame
            dict with success=False and reason on failure.
        """
        detection = self.detect(img)
        if not detection.get('success', False):
            return {
                'success': False,
                'reason': detection.get('reason', 'chessboard detection failed'),
                'image_points': detection.get('image_points', []),
                'object_points': detection.get('object_points', self.get_object_points()),
            }

        obj_points = detection['object_points'].astype(np.float32)
        img_points = np.asarray(detection['image_points'], dtype=np.float32)
        if img_points.ndim == 3 and img_points.shape[1] == 1:
            img_points = img_points.reshape(-1, 2)

        cam_mtx = np.asarray(camera_matrix, dtype=np.float32)
        dist = None if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float32)

        ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, cam_mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return {
                'success': False,
                'reason': 'solvePnP failed',
                'image_points': img_points,
                'object_points': obj_points,
            }

        rot_mtx, _ = cv2.Rodrigues(rvec)

        # Board origin and axes in camera frame.
        board_position = tvec.reshape(3)                  # origin corner [0,0] in camera frame
        board_x_axis   = rot_mtx[:, 0]                   # board +X in camera frame
        board_y_axis   = rot_mtx[:, 1]                   # board +Y in camera frame
        board_z_axis   = rot_mtx[:, 2]                   # board normal in camera frame

        # Geometric centre: average of all object points transformed into camera frame.
        obj_center_board  = obj_points.mean(axis=0).reshape(3, 1).astype(np.float32)
        board_center       = (rot_mtx @ obj_center_board + tvec).reshape(3)

        return {
            'success': True,
            'image_points': img_points,
            'object_points': obj_points,
            'rvec': rvec,
            'tvec': tvec,
            'rotation_matrix': rot_mtx,
            'board_position': board_position,
            'board_center': board_center,
            'board_x_axis': board_x_axis,
            'board_y_axis': board_y_axis,
            'board_z_axis': board_z_axis,
        }

    def get_grid_in_camera_coordinates(self, rvec, tvec, camera_matrix, dist_coeffs):
        """Project 3D grid points onto the camera image using the estimated pose.

        Args:
            grid_points: (N, 3) array of 3D points in board coordinates.
            rvec: (3, 1) rotation vector from solvePnP (board → camera).
            tvec: (3, 1) translation vector from solvePnP (board → camera).
            camera_matrix: Intrinsic matrix (3x3).
            dist_coeffs: Distortion coefficients (optional).

        Returns:
            projected_points: (N, 2) array of 2D points in image coordinates.
        """
        grid_points         = self.get_grid_points()
        cam_mtx             = np.asarray(camera_matrix, dtype=np.float32)
        dist                = np.asarray(dist_coeffs, dtype=np.float32)
        rot_mtx, _          = cv2.Rodrigues(rvec)

        # transform grid points from board frame to camera frame: P_cam = R * P_board + t
        #grid_transformed    = (rot_mtx @ grid_points + tvec).reshape(-1,3)  
        grid_transformed    = (grid_points @ rot_mtx.T + tvec.T)
        #Z                   = grid_transformed[:,2]  # depth of each point in camera frame      

        # points projected to the camera image plane (with distortion):  p_img = project(P_cam)
        projected_points, _ = cv2.projectPoints(grid_points.astype(np.float32), rvec, tvec, cam_mtx, dist)
        return grid_transformed, projected_points.reshape(-1, 2)

    def render_board_and_camera(self, object_points, camera_position, rvec=None, axis_length=None, show=True):
        """Render chessboard points and estimated camera position in 3D (matplotlib)."""
        a, b = self.pattern_size
        s = float(self.square_size)
        if axis_length is None:
            axis_length = max(2.0 * s, 1.0)

        pts = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
        cam = np.asarray(camera_position, dtype=np.float32).reshape(3)

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Draw board grid as wireframe in object coordinates.
        grid = pts.reshape(b, a, 3)
        ax.plot_wireframe(grid[:, :, 0], grid[:, :, 1], grid[:, :, 2], color='tab:blue', linewidth=1.0)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='tab:cyan', s=15, label='Chessboard corners')

        # Camera center.
        ax.scatter([cam[0]], [cam[1]], [cam[2]], c='tab:red', s=70, marker='^', label='Camera center')
        ax.text(cam[0], cam[1], cam[2], ' camera', color='tab:red')

        # Optional camera orientation axes (in object frame).
        if rvec is not None:
            rot_mtx, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float32))
            # camera frame unit axes expressed in object frame = columns of R^T
            cam_axes = rot_mtx.T
            colors = ['r', 'g', 'b']
            labels = ['Xc', 'Yc', 'Zc']
            for i in range(3):
                end = cam + cam_axes[:, i] * axis_length
                ax.plot([cam[0], end[0]], [cam[1], end[1]], [cam[2], end[2]], color=colors[i], linewidth=2)
                ax.text(end[0], end[1], end[2], labels[i], color=colors[i])

        # Keep axes visually balanced.
        x_vals = np.concatenate([pts[:, 0], np.array([cam[0]], dtype=np.float32)])
        y_vals = np.concatenate([pts[:, 1], np.array([cam[1]], dtype=np.float32)])
        z_vals = np.concatenate([pts[:, 2], np.array([cam[2]], dtype=np.float32)])
        max_range = max(np.ptp(x_vals), np.ptp(y_vals), np.ptp(z_vals), axis_length) * 0.6
        center = np.array([np.mean(x_vals), np.mean(y_vals), np.mean(z_vals)], dtype=np.float32)
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)

        ax.set_xlabel(f'X [{self.square_size} mm units]')
        ax.set_ylabel(f'Y [{self.square_size} mm units]')
        ax.set_zlabel(f'Z [{self.square_size} mm units]')
        ax.set_title('Chessboard and estimated camera pose')
        ax.legend(loc='best')
        plt.tight_layout()

        if show:
            plt.show(block=False)

        return fig, ax

    def render_board_and_camera_real_time(self, object_points, camera_position, rvec=None,
                                          axis_length=None, show=True, pause_sec=0.001):
        """Realtime version of render_board_and_camera.

        Reuses the same matplotlib figure/artists and updates their data in-place,
        so repeated calls can refresh visualization without creating a new plot.
        """
        a, b = self.pattern_size
        s = float(self.square_size)
        if axis_length is None:
            axis_length = max(2.0 * s, 1.0)

        pts = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
        cam = np.asarray(camera_position, dtype=np.float32).reshape(3)
        grid = pts.reshape(b, a, 3)

        # Create plot once, then update artists only.
        need_init = (
            self._rt_plot is None
            or self._rt_plot.get('fig', None) is None
            or not plt.fignum_exists(self._rt_plot['fig'].number)
        )

        if need_init:
            plt.ion()
            fig = plt.figure(figsize=(9, 7))
            ax = fig.add_subplot(111, projection='3d')

            # Board as persistent line artists (rows + cols), easier to update than wireframe collection.
            grid_lines = []
            for ri in range(b):
                line, = ax.plot(grid[ri, :, 0], grid[ri, :, 1], grid[ri, :, 2], color='tab:blue', linewidth=1.0)
                grid_lines.append(line)
            for ci in range(a):
                line, = ax.plot(grid[:, ci, 0], grid[:, ci, 1], grid[:, ci, 2], color='tab:blue', linewidth=1.0)
                grid_lines.append(line)

            board_scatter = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='tab:cyan', s=15, label='Chessboard corners')
            cam_scatter = ax.scatter([cam[0]], [cam[1]], [cam[2]], c='tab:red', s=70, marker='^', label='Camera center')
            cam_text = ax.text(cam[0], cam[1], cam[2], ' camera', color='tab:red')

            cam_axes_lines = []
            colors = ['r', 'g', 'b']
            labels = ['Xc', 'Yc', 'Zc']
            for i in range(3):
                line, = ax.plot([cam[0], cam[0]], [cam[1], cam[1]], [cam[2], cam[2]], color=colors[i], linewidth=2)
                txt = ax.text(cam[0], cam[1], cam[2], labels[i], color=colors[i])
                cam_axes_lines.append((line, txt))

            ax.set_xlabel(f'X [{self.square_size} mm units]')
            ax.set_ylabel(f'Y [{self.square_size} mm units]')
            ax.set_zlabel(f'Z [{self.square_size} mm units]')
            ax.set_title('Chessboard and estimated camera pose (real-time)')
            ax.legend(loc='best')
            plt.tight_layout()

            self._rt_plot = {
                'fig': fig,
                'ax': ax,
                'grid_lines': grid_lines,
                'board_scatter': board_scatter,
                'cam_scatter': cam_scatter,
                'cam_text': cam_text,
                'cam_axes_lines': cam_axes_lines,
            }
        else:
            fig = self._rt_plot['fig']
            ax = self._rt_plot['ax']

        # --- Update board lines ---
        line_idx = 0
        for ri in range(b):
            line = self._rt_plot['grid_lines'][line_idx]
            line.set_data_3d(grid[ri, :, 0], grid[ri, :, 1], grid[ri, :, 2])
            line_idx += 1
        for ci in range(a):
            line = self._rt_plot['grid_lines'][line_idx]
            line.set_data_3d(grid[:, ci, 0], grid[:, ci, 1], grid[:, ci, 2])
            line_idx += 1

        # --- Update scatters ---
        self._rt_plot['board_scatter']._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        self._rt_plot['cam_scatter']._offsets3d = (np.array([cam[0]]), np.array([cam[1]]), np.array([cam[2]]))

        # --- Update camera label text ---
        old_text = self._rt_plot.get('cam_text', None)
        if old_text is not None:
            old_text.remove()
        self._rt_plot['cam_text'] = ax.text(cam[0], cam[1], cam[2], ' camera', color='tab:red')

        # --- Update camera orientation axes ---
        if rvec is not None:
            rot_mtx, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float32))
            cam_axes = rot_mtx.T
            labels = ['Xc', 'Yc', 'Zc']
            for i in range(3):
                line, txt = self._rt_plot['cam_axes_lines'][i]
                end = cam + cam_axes[:, i] * axis_length
                line.set_data_3d([cam[0], end[0]], [cam[1], end[1]], [cam[2], end[2]])
                txt.remove()
                color = ['r', 'g', 'b'][i]
                self._rt_plot['cam_axes_lines'][i] = (line, ax.text(end[0], end[1], end[2], labels[i], color=color))

        # Keep axes balanced around board + camera.
        x_vals = np.concatenate([pts[:, 0], np.array([cam[0]], dtype=np.float32)])
        y_vals = np.concatenate([pts[:, 1], np.array([cam[1]], dtype=np.float32)])
        z_vals = np.concatenate([pts[:, 2], np.array([cam[2]], dtype=np.float32)])
        max_range = max(np.ptp(x_vals), np.ptp(y_vals), np.ptp(z_vals), axis_length) * 0.6
        center = np.array([np.mean(x_vals), np.mean(y_vals), np.mean(z_vals)], dtype=np.float32)
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)

        if show:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(pause_sec)

        return fig, ax

    def detect_estimate_and_render(self, img, camera_matrix, dist_coeffs=None, axis_length=None, show=True):
        """Detect chessboard, estimate camera 3D pose, and render board + camera.

        Returns:
            dict pose result from estimate_camera_pose(), plus optional 'figure' and 'axes'.
        """
        pose = self.estimate_camera_pose(img, camera_matrix, dist_coeffs)
        if not pose.get('success', False):
            log.info(f"Pose estimation failed: {pose.get('reason', 'unknown reason')}")
            return pose

        fig, ax = self.render_board_and_camera(
            object_points=pose['object_points'],
            camera_position=pose['camera_position'],
            rvec=pose['rvec'],
            axis_length=axis_length,
            show=show,
        )
        pose['figure'] = fig
        pose['axes'] = ax
        return pose

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

    def test_render_board_and_camera(self):
        """Test chessboard pose estimation and 3D rendering on a calibration image."""
        file_path = r"C:\Work\Code\robot_vision\pose6d\data\camera_calibration\calib_robot_0001.jpg"
        img = cv2.imread(file_path)
        self.assertTrue(img is not None)

        h, w = img.shape[:2]
        # Approximate intrinsics for test robustness; replace with calibrated values when available.
        fx = 600.0
        fy = 600.0
        cx = w / 2.0
        cy = h / 2.0
        camera_matrix = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        pose = self.s.estimate_camera_pose(img, camera_matrix, dist_coeffs)
        self.assertTrue(pose.get('success', False))

        fig, ax = self.s.render_board_and_camera(
            object_points=pose['object_points'],
            camera_position=pose['camera_position'],
            rvec=pose.get('rvec', None),
            show=True,
        )

        self.assertTrue(fig is not None)
        self.assertTrue(ax is not None)
        plt.close(fig)

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

    def test_rs_camera_connection(self):
        """
        Function that connects to RS camera and shows live stream,
        chessboard detection and real-time 3D pose rendering.
        """
        import importlib.util
        #from opencv_realsense_camera import RealSense

        cam_module_path = r"C:\Work\Code\Fast-FoundationStereo\scripts\opencv_realsense_camera.py"
        self.assertTrue(os.path.isfile(cam_module_path))

        spec = importlib.util.spec_from_file_location("opencv_realsense_camera", cam_module_path)
        rs_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rs_mod)
        RealSense = rs_mod.RealSense

        cap = RealSense(frame_size=(1280, 720), use_ir=False, mode = 'd16')
        self.assertTrue(cap is not None)

        # intr = cap.config.resolve(rs_mod.rs.pipeline_wrapper(cap.pipeline)) \
        #     .get_stream(rs_mod.rs.infrared, 1).as_video_stream_profile().get_intrinsics()
        intr = cap.intr #get_camera_intrinsics(1)
        camera_matrix = np.array([
            [intr.fx, 0.0, intr.ppx],
            [0.0, intr.fy, intr.ppy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        dist_coeffs = np.array(intr.coeffs, dtype=np.float32).reshape(-1, 1)

        # Use a finite loop for test-style behavior; press 'q' in OpenCV window to exit early.
        try:
            for _ in range(300):
                ret, _ = cap.read()
                self.assertTrue(ret)

                # Use left IR image for chessboard detection.
                ir_left = cap.img_l
                pose = self.s.estimate_camera_pose(ir_left, camera_matrix, dist_coeffs)
                if pose.get('success', False):
                    self.s.render_board_and_camera_real_time(
                        object_points=pose['object_points'],
                        camera_position=pose['camera_position'],
                        rvec=pose.get('rvec', None),
                        show=True,
                        pause_sec=0.001,
                    )

                # Keep OpenCV feed visible and allow keyboard control ('q' to break).
                should_exit = cap.show_image(cap.img_l)
                if should_exit:
                    break
        finally:
            cap.close()
            cv2.destroyAllWindows()
           
    def test_get_grid_in_camera_coordinates(self):
        """Test projecting a 3D grid onto the camera image using the estimated pose."""
        file_path = r"C:\Work\Code\robot_vision\pose6d\data\camera_calibration\calib_robot_0001.jpg"
        img = cv2.imread(file_path)
        self.assertTrue(img is not None)

        h, w = img.shape[:2]
        fx = 600.0
        fy = 600.0
        cx = w / 2.0
        cy = h / 2.0
        camera_matrix = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        pose = self.s.estimate_camera_pose(img, camera_matrix, dist_coeffs)
        self.assertTrue(pose.get('success', False))

        XYZ, projected_points = self.s.get_grid_in_camera_coordinates(
            rvec=pose['rvec'],
            tvec=pose['tvec'],
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )

        # Draw projected grid points on the image.
        for pt in projected_points:
            cv2.circle(img, (int(pt[0]), int(pt[1])), radius=1, color=(0, 255, 0), thickness=-1)

        cv2.imshow('Projected Grid', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# ----------------------------------------------------
#%% Run Test
def RunTest():
    "Run all tests in the MainApp class"
    tst = TestObjectChessboard()

    # tst.test_object_detect_single_image()  # interactive (waits for key press)
    #tst.test_render_board_and_camera()
    tst.test_rs_camera_connection()
    #tst.test_get_grid_in_camera_coordinates()

    

#%% Run ALL
if __name__ == '__main__':
    #print(__doc__)
    RunTest()
