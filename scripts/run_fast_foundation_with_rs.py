
'''
python scripts/run_fast_foundation_with_rs.py - github

Environment:
    ffs - GPU laptop

Installation:
    python -m venv ./envs/ffs
    source ./envs/ffs/bin/activate
    pip install torch==2.6.0 torchvision==0.21.0 xformers --index-url https://download.pytorch.org/whl/cu124
    cd Fast-FoundationStereo
    pip install -r requirements.txt
    pip install pyrealsense2
    download weights : https://drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap

    python scripts/run_demo.py --model_dir weights/20-30-48/model_best_bp2_serialize.pth --left_file demo_data/left.png --right_file demo_data/right.png --intrinsic_file assets/K.txt --out_dir output/ --remove_invisible 0 --denoise_cloud 1  --scale 1 --get_pc 1 --valid_iters 8 --max_disp 192 --zfar 100


'''




import numpy as np
from argparse import ArgumentParser

import os,sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
import argparse, torch, logging, yaml
import numpy as np
from Utils import (
    AMP_DTYPE, set_logging_format, set_seed, vis_disparity,
    depth2xyzmap, toOpen3dCloud, o3d,
)
import cv2
#from faro_data_manager import DataSource
from scripts.data_manager_faro_laptop import DataSource


def process_arguments():
    parser = ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default=f'{code_dir}/../weights/20-30-48/model_best_bp2_serialize.pth', type=str)
    parser.add_argument('--left_file', default=f'{code_dir}/../demo_data/left.png', type=str)
    parser.add_argument('--right_file', default=f'{code_dir}/../demo_data/right.png', type=str)
    parser.add_argument('--intrinsic_file', default=f'{code_dir}/../demo_data/K.txt', type=str, help='camera intrinsic matrix and baseline file')
    parser.add_argument('--out_dir', default='/home/bowen/debug/stereo_output', type=str)
    parser.add_argument('--remove_invisible', default=1, type=int)
    parser.add_argument('--denoise_cloud', default=0, type=int)
    parser.add_argument('--denoise_nb_points', type=int, default=30, help='number of points to consider for radius outlier removal')
    parser.add_argument('--denoise_radius', type=float, default=0.03, help='radius to use for outlier removal')
    parser.add_argument('--scale', default=1, type=float)
    parser.add_argument('--hiera', default=0, type=int)
    parser.add_argument('--get_pc', type=int, default=1, help='save point cloud output')
    parser.add_argument('--valid_iters', type=int, default=8, help='number of flow-field updates during forward pass')
    parser.add_argument('--max_disp', type=int, default=192, help='maximum disparity')
    parser.add_argument('--zfar', type=float, default=100, help="max depth to include in point cloud")

    return parser.parse_args()

# Original -------------------------------------------

# Original Split -------------------------------------------
from opencv_realsense_camera import RealSense
import time

def foundation_stereo_algo_init(args):
    "initialize the algorithm"

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    os.system(f'rm -rf {args.out_dir} && mkdir -p {args.out_dir}')

    with open(f'{os.path.dirname(args.model_dir)}/cfg.yaml', 'r') as ff:
        cfg:dict = yaml.safe_load(ff)
    for k in args.__dict__:
        if args.__dict__[k] is not None:
            cfg[k] = args.__dict__[k]
    args = OmegaConf.create(cfg)
    logging.info(f"args:\n{args}")
    model = torch.load(args.model_dir, map_location='cpu', weights_only=False)
    model.args.valid_iters = args.valid_iters
    model.args.max_disp = args.max_disp

    model.cuda().eval()
    return model

def foundation_stereo_algo(args, model, np_left, np_right):
    "stereo algo"
    scale = args.scale

    img0 = np_left #imageio.imread(args.left_file)
    img1 = np_right #imageio.imread(args.right_file)
    if len(img0.shape)==2:
        img0 = np.tile(img0[...,None], (1,1,3))
        img1 = np.tile(img1[...,None], (1,1,3))

    img0    = img0[...,:3]
    img1    = img1[...,:3]
    Ho,Wo   = img0.shape[:2]

    img0    = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
    img1    = cv2.resize(img1, dsize=(img0.shape[1], img0.shape[0]))

    H,W     = img0.shape[:2]
    #img0_ori = img0.copy()
    #img1_ori = img1.copy()
    #logging.info(f"img0: {img0.shape}")
    #imageio.imwrite(f'{args.out_dir}/left.png', img0)
    #imageio.imwrite(f'{args.out_dir}/right.png', img1)

    img0    = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
    img1    = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
    padder  = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)

    #logging.info(f"Start forward, 1st time run can be slow due to compilation")
    with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
        if not args.hiera:
            disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True, optimize_build_volume='pytorch1')
        else:
            disp = model.run_hierachical(img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5)

    #logging.info("forward done")
    
    scale_factor   = 1/scale
    disp    = padder.unpad(disp.float())
    #disp    = disp.data.cpu().numpy().reshape(H,W).clip(0, None)
    disp    = disp.data.cpu().numpy().reshape(H,W)*scale_factor
    disp    = disp.clip(0, None)

    # recover original size
    #orig_shape     = np_left.shape[::-1]
    orig_shape     = (np_left.shape[1],np_left.shape[0])
    #print(orig_shape, disp.shape)
    #scale_factor   = orig_shape[0]/process_shape[0]
    disp_out = cv2.resize(disp, orig_shape, interpolation=cv2.INTER_NEAREST) # A: changed    

    # cmap    = None
    # min_val = None
    # max_val = None
    # vis     = vis_disparity(disp, min_val=min_val, max_val=max_val, cmap=cmap, color_map=cv2.COLORMAP_TURBO)
    # vis     = np.concatenate([img0_ori, img1_ori, vis], axis=1)
    # imageio.imwrite(f'{args.out_dir}/disp_vis.png', vis)
    # s = 1280/vis.shape[1]
    # resized_vis = cv2.resize(vis, (int(vis.shape[1]*s), int(vis.shape[0]*s)))
    # cv2.imshow('disp', resized_vis[:,:,::-1])
    # cv2.waitKey(0)

    # if args.remove_invisible:
    #     yy,xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
    #     us_right = xx-disp
    #     invalid = us_right<0
    #     disp[invalid] = np.inf

    # debug and show
    #show_point_cloud(args, disp, img0_ori)
    #disp = disp * 1000 # must be in mm
    return disp_out

def convert_disparity_to_depth(BF, disparity):
    "from GIL"
    
    disparity           = disparity.astype(np.float32) 
    depth               = np.zeros_like(disparity) 
    disparity_valid     = disparity > 0.1
    depth[disparity_valid]   = BF / disparity[disparity_valid]
    #depth[disparity_valid]   += 0.5  # LUT in the simulator
    return depth.astype(np.uint16)

def depth_opencv_rs_merge(depth_rs, depth_cv):
    "computing disparity by merging depth from real sense and opencv"
    depth_merged    = np.copy(depth_rs)
    # lesss than 400 mm use opencv depth (520 min Z in HD D455)
    mask               = (depth_cv < 550) & (depth_cv > 80)
    depth_merged[mask] = depth_cv[mask]

    # A: changed
    # # if the real sense depth is invalid use opencv depth
    # mask            = (depth_rs < 1) & (depth_cv > 0)
    # depth_merged[mask] = depth_cv[mask]

    return depth_merged

def depth_fs_rs_error(depth_rs, depth_fs):
    "computing depth error between real sense and fs"
    depth_rs, depth_fs  = depth_rs.astype(np.float32), depth_fs.astype(np.float32) 
    depth_error         = np.abs(depth_rs - depth_fs)

    # lesss than 400 mm use opencv depth (520 min Z in HD D455)
    mask               = depth_rs < 2
    depth_error[mask]  = 0

    return depth_error

def preprocess(frame, scale_factor = 0.5):
    "convert and downscale"
    frame           = frame.astype(np.float32)

    # assign
    imgL            = frame[:,:,0]
    imgR            = frame[:,:,1]
    imgD            = frame[:,:,2] 
        
    # if scale_factor < 0.9:

    #     imgL           = cv2.resize(imgL, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    #     imgR           = cv2.resize(imgR, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    #     imgD           = cv2.resize(imgD, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)

    return imgL, imgR, imgD

def show_images_depth(imgD = None, imgC = None,  fig_num = 1, fig_name = 'Depth Image', vmax = 1500):
    "draw results of depth estimation"
          

    if (imgD is None) and (imgC is None):
        print('No images found')
        return False
    
    elif imgD is None: # no data acquired
        img_show = imgC

    elif imgC is None: # no data is processed
        img_show = imgD      
        #img_show = cv.applyColorMap(self.imgD, cv.COLORMAP_TURBO)     

    elif np.all(imgD.shape == imgC.shape):
        img_show = np.concatenate((imgD, imgC ), axis = 1)

    # deal with 16 uint    
    if img_show.dtype == 'uint16' or img_show.dtype == 'float32':
        img_show    = cv2.convertScaleAbs(img_show, alpha=0.1)
        img_show    = cv2.applyColorMap(img_show, cv2.COLORMAP_TURBO) #   
        pass
    else:
        #self.imgD = np.repeat(self.imgD[:,:,np.newaxis], 3, axis = 2)
        #img_show = np.concatenate((self.imgD, self.imgC ), axis = 1)
        #img_show = cv.applyColorMap(img_show.astype(np.uint8), cv.COLORMAP_TURBO) 
        #img_show = self.imgC #np.concatenate((self.imgD, self.imgC ), axis = 1)
        pass

    # deal with black and white
    if img_show.shape[1] > 2400:
        img_show = cv2.pyrDown(img_show)
            
    cv2.imshow(f'{fig_name} (q-exit)', img_show)
    ch = cv2.waitKey(5)
    ret = ch == ord('q')

    # plt.figure()
    # plt.imshow(imgD, vmin = 100, vmax=vmax)
    # plt.title(fig_name)
    # plt.show(block=False)

    return ret

def show_point_cloud(args, disp, img0_ori):
    "from fs"
    if not args.get_pc:
      return
      
    scale = args.scale
    with open(args.intrinsic_file, 'r') as f:
        lines = f.readlines()
        K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
        baseline = float(lines[1])
        K[:2] *= scale
        depth = K[0,0]*baseline/disp
        np.save(f'{args.out_dir}/depth_meter.npy', depth)
        xyz_map = depth2xyzmap(depth, K)
        pcd = toOpen3dCloud(xyz_map.reshape(-1,3), img0_ori.reshape(-1,3))
        keep_mask = (np.asarray(pcd.points)[:,2]>0) & (np.asarray(pcd.points)[:,2]<=args.zfar)
        keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
        pcd = pcd.select_by_index(keep_ids)
        #o3d.io.write_point_cloud(f'{args.out_dir}/cloud.ply', pcd)
        #logging.info(f"PCL saved to {args.out_dir}")

    if args.denoise_cloud:
        logging.info("[Optional step] denoise point cloud...")
        cl, ind = pcd.remove_radius_outlier(nb_points=args.denoise_nb_points, radius=args.denoise_radius)
        inlier_cloud = pcd.select_by_index(ind)
        #o3d.io.write_point_cloud(f'{args.out_dir}/cloud_denoise.ply', inlier_cloud)
        pcd = inlier_cloud

    logging.info("Visualizing point cloud. Press ESC to exit.")
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    id = np.asarray(pcd.points)[:,2].argmin()
    ctr.set_lookat(np.asarray(pcd.points)[id])
    ctr.set_up([0, -1, 0])
    vis.run()
    vis.destroy_window()

def depth2xyz(depth:np.ndarray, K, us=None, vs=None, zmin=0.1):
  #invalid_mask = (depth<zmin)
  # one time init
  if us is None:
    H,W = depth.shape[:2]
    vs,us = np.meshgrid(np.arange(0,H),np.arange(0,W), sparse=False, indexing='ij')
    vs = vs.reshape(-1)
    us = us.reshape(-1)

  zs = depth[vs,us]
  zs[zs<zmin] = zmin
  xs = (us-K[0,2])*zs/K[0,0]
  ys = (vs-K[1,2])*zs/K[1,1]
  pts = np.stack((xs.reshape(-1),ys.reshape(-1),zs.reshape(-1)), 1)  #(N,3)

  return pts,us,vs    

# ----------------------------------------
import open3d as o3d
import numpy as np
import time

def test_point_cloud_rt():
    "show point cloud update in RT"
    # Source - https://stackoverflow.com/a/74669788
    # Posted by Javier TG, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-02-10, License - CC BY-SA 4.0

    # create visualizer and window.
    vis = o3d.visualization.Visualizer()
    vis.create_window(height=480, width=640)

    # initialize pointcloud instance.
    pcd = o3d.geometry.PointCloud()
    # *optionally* add initial points
    points = np.random.rand(10, 3)
    pcd.points = o3d.utility.Vector3dVector(points)

    # include it in the visualizer before non-blocking visualization.
    vis.add_geometry(pcd)

    # to add new points each dt secs.
    dt = 0.01
    # number of points that will be added
    n_new = 10

    previous_t = time.time()

    # run non-blocking visualization. 
    # To exit, press 'q' or click the 'x' of the window.
    keep_running = True
    while keep_running:
        
        if time.time() - previous_t > dt:
            # Options (uncomment each to try them out):
            # 1) extend with ndarrays.
            pcd.points.extend(np.random.rand(n_new, 3))
            
            # 2) extend with Vector3dVector instances.
            # pcd.points.extend(
            #     o3d.utility.Vector3dVector(np.random.rand(n_new, 3)))
            
            # 3) other iterables, e.g
            # pcd.points.extend(np.random.rand(n_new, 3).tolist())
            
            vis.update_geometry(pcd)
            previous_t = time.time()

        keep_running = vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()

def test_video_stream_rs_and_foundation():
    "streaming rs to foundation"
    d           = RealSense(mode = 'd16', use_ir = True, frame_size = (1280,720))
    #d.load_preset_from_file('vpi/preset_rsm.json')
    #self.cap.set_exposure(1000) # 10ms
    d.switch_projector(True)
    #self.cap.set_laser_power(100) # max power
    camera_bf   = d.get_bf() # for depth conversion (should be in meter)
    args        = process_arguments()

    # init
    model       = foundation_stereo_algo_init(args)
    
    ret  = False
    while not ret:
        # frame is I1,I2, D data
        retf, frame = d.read()
        if retf is False:
            print('is your camera open/connected?')
            break
    
        # extract
        img_left, img_right, img_depth_rs = preprocess(frame, scale_factor=1.0)

        t_start         = time.time()
        img_disparity   = foundation_stereo_algo(args, model, img_left, img_right)
        print(f'Disp : {img_disparity.min()} - {img_disparity.max()}')
        img_depth_fs    = convert_disparity_to_depth(camera_bf, img_disparity)
        print(f'Depth : {img_depth_fs.min()} - {img_depth_fs.max()}')
        depth_merged    = depth_opencv_rs_merge(img_depth_rs, img_depth_fs)
        print(f'Proces time : {time.time()-t_start} sec')
        
        # massage
        #cmap, min_val, max_val = None, None, None
        #img_depth_fs = vis_disparity(img_disparity, min_val=min_val, max_val=max_val, cmap=cmap, color_map=cv2.COLORMAP_TURBO)

        # show
        ret1 = show_images_depth(img_depth_rs, None, fig_name='Depth RS')
        ret2 = show_images_depth(img_depth_fs, None, fig_name='Depth FS')
        ret3 = show_images_depth(depth_merged, None, fig_name='Depth Merged')
        ret4 = show_images_depth(img_left.astype(np.uint8), img_right.astype(np.uint8), fig_name='Images L-R')
        ret = ret1 or ret2 or ret3 or ret4
        
    d.release()

def test_video_error_rs_versus_foundation():
    "streaming rs to foundation - checks error between them"
    d           = RealSense(mode = 'd16', use_ir = True, frame_size = (1280,720))
    #d.load_preset_from_file('vpi/preset_rsm.json')
    #d.set_exposure(100000) # 10ms
    #d.switch_projector(False)
    #self.cap.set_laser_power(100) # max power
    camera_bf   = d.get_bf() # for depth conversion (should be in meter)
    args        = process_arguments()

    # init
    model       = foundation_stereo_algo_init(args)
    
    ret  = False
    while not ret:
        # frame is I1,I2, D data
        retf, frame = d.read()
        if retf is False:
            print('is your camera open/connected?')
            break
    
        # extract
        img_left, img_right, img_depth_rs = preprocess(frame, scale_factor=1.0)

        t_start         = time.time()
        img_disparity   = foundation_stereo_algo(args, model, img_left, img_right)
        #print(f'Disp : {img_disparity.min()} - {img_disparity.max()}')
        img_depth_fs    = convert_disparity_to_depth(camera_bf, img_disparity)
        #print(f'Depth : {img_depth_fs.min()} - {img_depth_fs.max()}')
        depth_error    = depth_fs_rs_error(img_depth_rs, img_depth_fs)
        print(f'Proces time : {time.time()-t_start} sec')
        
        # massage
        #cmap, min_val, max_val = None, None, None
        #img_depth_fs = vis_disparity(img_disparity, min_val=min_val, max_val=max_val, cmap=cmap, color_map=cv2.COLORMAP_TURBO)

        # show
        ret1 = show_images_depth(img_depth_rs, None, fig_name='Depth RS')
        ret2 = show_images_depth(img_depth_fs, None, fig_name='Depth FS')
        ret3 = show_images_depth(depth_error, None, fig_name='Depth Error', vmax = 100)
        ret4 = show_images_depth(img_left.astype(np.uint8), img_right.astype(np.uint8), fig_name='Images L-R')
        ret = ret1 or ret2 or ret3 or ret4

        # d.save_image(img_depth_rs,fname='depth_rs')
        # d.save_image(img_depth_fs,fname='depth_fs')
        # d.save_image(img_left,fname='img_left')
        # d.save_image(img_right,fname='img_right')

        # plt.show()
        
    d.release()

def test_point_cloud_rs_versus_foundation():
    "streaming rs to foundation - checks error between them"
    d           = RealSense(mode = 'd16', use_ir = True, frame_size = (1280,720))
    #d.load_preset_from_file('vpi/preset_rsm.json')
    #self.cap.set_exposure(1000) # 10ms
    #d.switch_projector(True)
    #self.cap.set_laser_power(100) # max power
    camera_bf   = d.get_bf() # for depth conversion (should be in meter)
    #camera_k    = d.get_camera_intrinsics()
    args        = process_arguments()
    K           = np.array([[637,0,640],[0,637,360],[0,0,1]])

    # init
    model       = foundation_stereo_algo_init(args)

    # create visualizer and window.
    vis         = o3d.visualization.Visualizer()
    vis.create_window(height=720, width=1280)

    # initialize pointcloud instance.
    pcd         = o3d.geometry.PointCloud()
    # *optionally* add initial points
    points      = np.random.rand(10, 3)*1000
    pcd.points  = o3d.utility.Vector3dVector(points)
    us, vs      = None, None

    # include it in the visualizer before non-blocking visualization.
    vis.add_geometry(pcd)    
    
    ret  = False
    while not ret:
        # frame is I1,I2, D data
        retf, frame = d.read()
        if retf is False:
            print('is your camera open/connected?')
            break
    
        # extract
        img_left, img_right, img_depth_rs = preprocess(frame, scale_factor=1.0)

        t_start         = time.time()
        img_disparity   = foundation_stereo_algo(args, model, img_left, img_right)
        print(f'Disp : {img_disparity.min()} - {img_disparity.max()}')
        img_depth_fs    = convert_disparity_to_depth(camera_bf, img_disparity)
        print(f'Depth : {img_depth_fs.min()} - {img_depth_fs.max()}')
        depth_error    = depth_fs_rs_error(img_depth_rs, img_depth_fs)
        print(f'Proces time : {time.time()-t_start} sec')
        
        # massage
        #cmap, min_val, max_val = None, None, None
        #img_depth_fs = vis_disparity(img_disparity, min_val=min_val, max_val=max_val, cmap=cmap, color_map=cv2.COLORMAP_TURBO)
        #xyz_map, pts     = depth2xyzmap(img_depth_fs, K)
        pts, us, vs     = depth2xyz(img_depth_fs, K, us, vs, zmin=100)
        clr             = np.zeros_like(pts)
        clr[:,0] = clr[:,1] = clr[:,2] = img_left.flatten()

        # show point cloud
        #pcd.points.extend(pts)
        
        #vis.remove_geometry(pcd)
        #pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        #pcd.colors = o3d.utility.Vector3dVector(clr)
        downpcd = pcd.voxel_down_sample(voxel_size=0.1)
        #vis.add_geometry(pcd)
        
        # 3) other iterables, e.g
        # pcd.points.extend(np.random.rand(n_new, 3).tolist())
        
        vis.update_geometry(downpcd)

        keep_running = vis.poll_events()
        if not keep_running: break
        vis.update_renderer()        

        # show
        ret1 = show_images_depth(img_depth_rs, None, fig_name='Depth RS')
        ret2 = show_images_depth(img_depth_fs, None, fig_name='Depth FS')
        ret3 = show_images_depth(depth_error, None, fig_name='Depth Error')
        ret4 = show_images_depth(img_left.astype(np.uint8), img_right.astype(np.uint8), fig_name='Images L-R')
        ret = ret1 or ret2 or ret3 or ret4
        
    d.release()
    vis.destroy_window()

def test_video_stream_rs_versus_foundation_x2():
    "streaming rs to foundation - checks speed"
    d           = RealSense(mode = 'd16', use_ir = True, frame_size = (1280,720))
    #d.load_preset_from_file('vpi/preset_rsm.json')
    #d.set_exposure(100000) # 10ms
    #d.switch_projector(False)
    #self.cap.set_laser_power(100) # max power
    camera_bf   = d.get_bf() # for depth conversion (should be in meter)
    args        = process_arguments()
    args.scale  = 0.5

    # init
    model       = foundation_stereo_algo_init(args)
    
    ret  = False
    while not ret:
        # frame is I1,I2, D data
        retf, frame = d.read()
        if retf is False:
            print('is your camera open/connected?')
            break
    
        # extract
        img_left, img_right, img_depth_rs = preprocess(frame, scale_factor=1.0)

        t_start         = time.time()
        img_disparity   = foundation_stereo_algo(args, model, img_left, img_right)
        img_depth_fs    = convert_disparity_to_depth(camera_bf, img_disparity)
        depth_error    = depth_fs_rs_error(img_depth_rs, img_depth_fs)
        print(f'Proces time : {time.time()-t_start} sec')
        
        # massage
        #cmap, min_val, max_val = None, None, None
        #img_depth_fs = vis_disparity(img_disparity, min_val=min_val, max_val=max_val, cmap=cmap, color_map=cv2.COLORMAP_TURBO)

        # show
        ret1 = show_images_depth(img_depth_rs, None, fig_name='Depth RS')
        ret2 = show_images_depth(img_depth_fs, None, fig_name='Depth FS')
        ret3 = show_images_depth(depth_error, None, fig_name='Depth Error', vmax = 100)
        ret4 = show_images_depth(img_left.astype(np.uint8), img_right.astype(np.uint8), fig_name='Images L-R')
        ret = ret1 or ret2 or ret3 or ret4

        # d.save_image(img_depth_rs,fname='depth_rs')
        # d.save_image(img_depth_fs,fname='depth_fs')
        # d.save_image(img_left,fname='img_left')
        # d.save_image(img_right,fname='img_right')
        # plt.show()
        
    d.release()

def merge_fs_rs(depth_rs, depth_fs):
    "trying to merge information and deal with non valid regions"
    nr, nc       = depth_rs.shape
    depth_rs_out = depth_rs.copy()
    valid_rs     = depth_rs > 1
    start_rs     = np.zeros_like(valid_rs)
    start_rs[:,:-1] = np.logical_and(valid_rs[:,:-1] , ~valid_rs[:,1:]) # if pixel k is valid and k + 1 is not
    stop_rs      = np.zeros_like(valid_rs)
    stop_rs[:,1:] = np.logical_and(~valid_rs[:,:-1] , valid_rs[:,1:]) # if pixel k-1 is not valid and k  is valid

    for r in range(nr):
        start_ind = np.where(start_rs[r,:])[0]
        stop_ind  = np.where(stop_rs[r,:])[0]
        if len(start_ind) < 1 or len(stop_ind) < 1:
            continue
        
        for s in start_ind:
            ii = np.where(s < stop_ind)[0]
            if len(ii) < 1: continue
            f = stop_ind[0]
            if np.abs(depth_fs[r,s] - depth_fs[r,f]) < 0.1*depth_fs[r,s]:
                depth_rs_out[r,s:f] = depth_fs[r,s:f]
                print('.')

    return depth_rs_out
      
def test_video_stream_rs_fs_merge():
    "streaming rs to foundation - checks speed"
    d           = RealSense(mode = 'd16', use_ir = True, frame_size = (1280,720))
    #d.load_preset_from_file('vpi/preset_rsm.json')
    #d.set_exposure(100000) # 10ms
    #d.switch_projector(False)
    #self.cap.set_laser_power(100) # max power
    camera_bf   = d.get_bf() # for depth conversion (should be in meter)
    args        = process_arguments()
    args.scale  = 0.5

    # init
    model       = foundation_stereo_algo_init(args)
    
    ret  = False
    while not ret:
        # frame is I1,I2, D data
        retf, frame = d.read()
        if retf is False:
            print('is your camera open/connected?')
            break
    
        # extract
        img_left, img_right, img_depth_rs = preprocess(frame, scale_factor=1.0)

        t_start         = time.time()
        img_disparity   = foundation_stereo_algo(args, model, img_left, img_right)
        img_depth_fs    = convert_disparity_to_depth(camera_bf, img_disparity)
        img_depth_merge = merge_fs_rs(img_depth_rs, img_depth_fs)
        print(f'Proces time : {time.time()-t_start} sec')
        
        # massage
        #cmap, min_val, max_val = None, None, None
        #img_depth_fs = vis_disparity(img_disparity, min_val=min_val, max_val=max_val, cmap=cmap, color_map=cv2.COLORMAP_TURBO)

        # show
        ret1 = show_images_depth(img_depth_rs, None, fig_name='Depth RS')
        ret2 = show_images_depth(img_depth_fs, None, fig_name='Depth FS')
        ret3 = show_images_depth(img_depth_merge, None, fig_name='Depth Merge', vmax = 100)
        ret4 = show_images_depth(img_left.astype(np.uint8), img_right.astype(np.uint8), fig_name='Images L-R')
        ret = ret1 or ret2 or ret3 or ret4

        d.save_image(img_depth_rs,fname='depth_rs')
        d.save_image(img_depth_fs,fname='depth_fs')
        d.save_image(img_left,fname='img_left')
        d.save_image(img_right,fname='img_right')
        # plt.show()
        
    d.release()

def test_faro_rs_fs_error():
    "reading data files from FARO and comparing the results"

    #d           = RealSense(mode = 'd16', use_ir = True, frame_size = (1280,720))
    d           = DataSource()
    img_num     = d.init_directory()    
    camera_bf   = d.get_bf() # for depth conversion (should be in meter)

    args        = process_arguments()
    args.scale  = 0.5
    args.model_dir = '/home/administrato/dev/Fast-FoundationStereo/weights/20-30-48/model_finetuned_faro.pth'
    args.model_dir = '/home/administrato/dev/Fast-FoundationStereo/weights/20-30-48/model_finetuned_faro.pth'

    # init
    model       = foundation_stereo_algo_init(args)


    img_index   = np.random.randint(0,img_num,8)
    for k in img_index:

        # frame is I1,I2, D data
        out_data       = d.get_item(k, debug = True)
        #out_data       = d.load_specific_files(debug = True)
        #print(out_data)
    
        # extract
        img_left, img_right, img_depth_rs, img_depth_faro = out_data["img_left"], out_data["img_right"], out_data["img_depth_rs"], out_data["img_depth_faro"]
        print(img_left.shape, img_right.shape)

        # process
        t_start         = time.time()
        img_disparity   = foundation_stereo_algo(args, model, img_left, img_right)
        img_depth_fs    = convert_disparity_to_depth(camera_bf, img_disparity)
        img_error_fs    = depth_fs_rs_error(img_depth_rs, img_depth_fs)
        img_error_faro  = depth_fs_rs_error(img_depth_faro, img_depth_fs)
        print(f'Proces time : {time.time()-t_start} sec')

        # show
        ret1 = show_images_depth(img_depth_rs, None, fig_name='Depth RS')
        ret2 = show_images_depth(img_depth_fs, None, fig_name='Depth FS', vmax = 400)
        ret3 = show_images_depth(img_depth_faro, None, fig_name='Depth Faro')
        ret4 = show_images_depth(img_error_fs.astype(np.uint8),     None, fig_name='Error RS-FS', vmax = 100)
        ret5 = show_images_depth(img_error_faro.astype(np.uint8), None, fig_name='Error Faro-FS', vmax = 100)
        ret6 = show_images_depth(img_left.astype(np.uint8), None, fig_name='Images L')
        ret7 = show_images_depth(img_right.astype(np.uint8), None, fig_name='Images R')
        ret = ret1 or ret2 or ret3 or ret4 or ret5 or ret6 or ret7
        if ret: break

        # d.save_image(img_depth_rs,fname='depth_rs')
        #d.save_image(img_depth_fs,fname='depth_fs')
        # d.save_image(img_left,fname='img_left')
        # d.save_image(img_right,fname='img_right')
        # plt.show()
        
    #d.ckose()


if __name__ == '__main__':
    #main()
    #test_video_stream_rs_and_foundation() # ok
    #test_video_error_rs_versus_foundation() # ok
    # test_point_cloud_rt() # ok
    #test_point_cloud_rs_versus_foundation()
    #test_video_stream_rs_versus_foundation_x2() # ok
    test_video_stream_rs_fs_merge()
    #test_faro_rs_fs_error()








    