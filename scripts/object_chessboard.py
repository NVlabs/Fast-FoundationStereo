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
        self.square_size         = 10                  # size in mm of the pattern square         
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
        # prepare dense grid points with selected step size, in board coordinates
        scale_factor = self.square_size / grid_size
        a = int(self.pattern_size[0] * scale_factor)
        b = int(self.pattern_size[1] * scale_factor)
        s = grid_size

        grid_corners = np.zeros((b * a, 3), np.float32)
        grid_corners[:, :2] = np.mgrid[0:a, 0:b].T.reshape(-1, 2) * s
        return grid_corners

    def get_image_points(self, img):
        "detect corners in the image and return their coordinates"

        # reduce size
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

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
        """Detect chessboard and estimate camera pose with solvePnP."""
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
        """Detect chessboard and estimate board pose in camera coordinate system."""
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

        board_position = tvec.reshape(3)
        board_x_axis = rot_mtx[:, 0]
        board_y_axis = rot_mtx[:, 1]
        board_z_axis = rot_mtx[:, 2]

        obj_center_board = obj_points.mean(axis=0).reshape(3, 1).astype(np.float32)
        board_center = (rot_mtx @ obj_center_board + tvec).reshape(3)

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
        """Project 3D board grid points to the image using estimated pose."""
        grid_points = self.get_grid_points()
        cam_mtx = np.asarray(camera_matrix, dtype=np.float32)
        dist = np.asarray(dist_coeffs, dtype=np.float32)
        rot_mtx, _ = cv2.Rodrigues(rvec)

        grid_transformed = (grid_points @ rot_mtx.T + tvec.T)
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

        grid = pts.reshape(b, a, 3)
        ax.plot_wireframe(grid[:, :, 0], grid[:, :, 1], grid[:, :, 2], color='tab:blue', linewidth=1.0)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='tab:cyan', s=15, label='Chessboard corners')

        ax.scatter([cam[0]], [cam[1]], [cam[2]], c='tab:red', s=70, marker='^', label='Camera center')
        ax.text(cam[0], cam[1], cam[2], ' camera', color='tab:red')

        if rvec is not None:
            rot_mtx, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float32))
            cam_axes = rot_mtx.T
            colors = ['r', 'g', 'b']
            labels = ['Xc', 'Yc', 'Zc']
            for i in range(3):
                end = cam + cam_axes[:, i] * axis_length
                ax.plot([cam[0], end[0]], [cam[1], end[1]], [cam[2], end[2]], color=colors[i], linewidth=2)
                ax.text(end[0], end[1], end[2], labels[i], color=colors[i])

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
        """Realtime version of render_board_and_camera with persistent artists."""
        a, b = self.pattern_size
        s = float(self.square_size)
        if axis_length is None:
            axis_length = max(2.0 * s, 1.0)

        pts = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
        cam = np.asarray(camera_position, dtype=np.float32).reshape(3)
        grid = pts.reshape(b, a, 3)

        need_init = (
            self._rt_plot is None
            or self._rt_plot.get('fig', None) is None
            or not plt.fignum_exists(self._rt_plot['fig'].number)
        )

        if need_init:
            plt.ion()
            fig = plt.figure(figsize=(9, 7))
            ax = fig.add_subplot(111, projection='3d')

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

        line_idx = 0
        for ri in range(b):
            line = self._rt_plot['grid_lines'][line_idx]
            line.set_data_3d(grid[ri, :, 0], grid[ri, :, 1], grid[ri, :, 2])
            line_idx += 1
        for ci in range(a):
            line = self._rt_plot['grid_lines'][line_idx]
            line.set_data_3d(grid[:, ci, 0], grid[:, ci, 1], grid[:, ci, 2])
            line_idx += 1

        self._rt_plot['board_scatter']._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        self._rt_plot['cam_scatter']._offsets3d = (np.array([cam[0]]), np.array([cam[1]]), np.array([cam[2]]))

        old_text = self._rt_plot.get('cam_text', None)
        if old_text is not None:
            old_text.remove()
        self._rt_plot['cam_text'] = ax.text(cam[0], cam[1], cam[2], ' camera', color='tab:red')

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
        """Detect chessboard, estimate camera pose and render board + camera."""
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

    def test_render_board_and_camera_real_time(self):
        """Test real-time renderer initialization and in-place update path."""
        object_points = self.s.get_object_points()
        camera_position = np.array([50.0, 80.0, 250.0], dtype=np.float32)
        rvec = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)

        # First call should initialize persistent plot state.
        fig1, ax1 = self.s.render_board_and_camera_real_time(
            object_points=object_points,
            camera_position=camera_position,
            rvec=rvec,
            show=False,
        )
        self.assertTrue(fig1 is not None)
        self.assertTrue(ax1 is not None)
        self.assertTrue(self.s._rt_plot is not None)

        # Second call should reuse existing plot objects (real-time update path).
        camera_position_2 = np.array([60.0, 70.0, 240.0], dtype=np.float32)
        fig2, ax2 = self.s.render_board_and_camera_real_time(
            object_points=object_points,
            camera_position=camera_position_2,
            rvec=rvec,
            show=False,
        )
        self.assertTrue(fig2 is fig1)
        self.assertTrue(ax2 is ax1)

        # Cleanup to avoid leaking figure state across tests.
        plt.close(fig1)
        self.s._rt_plot = None

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
        Function that connects to RealSense camera and reads one frame
        """
        import importlib.util

        cam_module_path = os.path.join(current_dir, 'opencv_realsense_camera.py')
        self.assertTrue(os.path.isfile(cam_module_path))

        spec = importlib.util.spec_from_file_location("opencv_realsense_camera", cam_module_path)
        rs_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rs_mod)
        RealSense = rs_mod.RealSense

        cap = None
        try:
            cap = RealSense(frame_size=self.s.resolution, use_ir=True, mode='d16')
            self.assertTrue(cap is not None)

            ret, frame = cap.read()
            self.assertTrue(ret)
            self.assertTrue(frame is not None)
        finally:
            if cap is not None:
                cap.close()
            cv2.destroyAllWindows()
           

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
