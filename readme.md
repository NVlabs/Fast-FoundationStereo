# Fast-FoundationStereo: Real-Time Zero-Shot Stereo Matching

This is the official implementation of our paper accepted to CVPR 2026

[[Website]](https://nvlabs.github.io/Fast-FoundationStereo/) [[Paper]](https://arxiv.org/abs/2512.11130) [[Video]](https://www.youtube.com/watch?v=2BUYZojCzXE)

Authors: [Bowen Wen](https://wenbowen123.github.io/), [Shaurya Dewan](https://www.linkedin.com/in/shaurya-dewan-1b07231a2), [Stan Birchfield](https://research.nvidia.com/person/stan-birchfield)


# Abstract
Stereo foundation models achieve strong zero-shot generalization but remain computationally prohibitive for real-time applications. Efficient stereo architectures, on the other hand, sacrifice robustness for speed and require costly per-domain fine-tuning. To bridge this gap, we present Fast-FoundationStereo, a family of architectures that achieve, for the first time, strong zero-shot generalization at real-time frame rate. We employ a divide-and-conquer acceleration strategy with three components: (1) knowledge distillation to compress the hybrid backbone into a single efficient student; (2) blockwise neural architecture search for automatically discovering optimal cost filtering designs under latency budgets, reducing search complexity exponentially; and (3) structured pruning for eliminating redundancy in the iterative refinement module. Furthermore, we introduce an automatic pseudo-labeling pipeline used to curate 1.4M in-the-wild stereo pairs to supplement synthetic training data and facilitate knowledge distillation. The resulting model can run over 10× faster than FoundationStereo while closely matching its zero-shot accuracy, thus establishing a new state-of-the-art among real-time methods.

 [NOTE] This model is designed for real-time applications. For offline computation for the best accuracy, please checkout our earlier work [FoundationStereo](https://github.com/NVlabs/FoundationStereo).



<p align="center">
  <img src="assets/intro.jpg" width="100%"/>
</p>

<td align="center">
  <img src="assets/intro_c.webp" width="60%"/>
</td>
<td align="center">
  <img src="assets/bp2_vs_runtime.jpg" width="60%"/>
</td>


# Environment setup
- Option 1: Docker
```bash
docker build --network host -t ffs -f docker/dockerfile .
bash docker/run_container.sh
```

- Option 2: uv
```bash
uv venv --python=3.12
source .venv/bin/activate
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
uv pip install -r requirements.txt
```


# Weights and Trade-off
download from [here](https://drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap?usp=drive_link) and put under the folder `weights/` (e.g. `./weights/23-36-37`). Below table compares the differences among some representative models of varying sizes from our trained family. They are sorted from slowest to fastest, with accuracy descending.

To trade-off speed and accuracy, there are two options:
1) Try with different checkpoints.
2) Tune the config flags (see explanations in the "Run demo" section below).

| Checkpoint     |
|---------------|
| `23-36-37`    |
| `20-26-39`    |
| `00-04-56`    |
| `20-30-48`    |

# Run demo
```
python scripts/run_demo.py --model_dir weights/23-36-37/model_best_bp2_serialize.pth --left_file demo_data/left.png --right_file demo_data/right.png --intrinsic_file demo_data/K.txt --out_dir output/ --remove_invisible 0 --denoise_cloud 1  --scale 1 --get_pc 1 --valid_iters 8 --max_disp 192 --zfar 100
```
| Flag                        | Meaning                                                                |
|-----------------------------|------------------------------------------------------------------------|
| `--model_dir`               | Path to the trained weights/model file                                 |
| `--left_file`               | Path to the left image file                                            |
| `--right_file`              | Path to the right image file                                           |
| `--intrinsic_file`          | Path to the camera intrinsic matrix and baseline file                  |
| `--out_dir`                 | Output directory for saving results                                    |
| `--remove_invisible`        | Whether to ignore non-overlapping region's depth (0: no, 1: yes)      |
| `--denoise_cloud`           | Whether to apply denoising to the point cloud (0: no, 1: yes)          |
| `--scale`                   | Image scaling factor                                                   |
| `--get_pc`                  | Obtain point cloud output (0: no, 1: yes)                              |
| `--valid_iters`             | Number of refinement updates during forward pass                       |
| `--max_disp`                | Maximum disparity for volume encoding, 192 should be enough, unless you need to sense very near objects (e.g. <0.1m). Increasing it runs slower and uses more memory. |
| `--zfar`                    | Maximum depth to include in point cloud                                |

Refer to `scripts/run_demo.py` for comprehensive list of flags.

**Tips:**
- The input left and right images should be rectified and undistorted, which means there should not be fisheye kind of lens distortion and the epipolar lines are horizontal between the left/right images. If you obtain images from stereo cameras such as Zed, they usually have handled this for you.
- Do not swap left and right image. The left image should really be obtained from the left-side camera (objects will appear righter in the image).
- We recommend to use PNG files with no lossy compression
- Our method works best on stereo RGB images. However, we have also tested it on monochrome or IR stereo images (e.g. from RealSense D4XX series) and it works well too.
- To get point cloud for your own data, you need to specify the intrinsics. In the intrinsic file in args, 1st line is the flattened 1x9 intrinsic matrix, 2nd line is the baseline (distance) between the left and right camera, unit in meters.
- The model performs better for image width size <1000. You can run with smaller scale, e.g. `--scale 0.5` to downsize input image, then upsize the output depth to your need with nearest neighbor interpolation.
- For faster inference, you can reduce the input image resolution by e.g. `--scale 0.5`, and reduce refine iterations by e.g. `--valid_iters 4`.
- Note that the 1st time running is slower due to compilation, use a while loop after warm up for live running.

Expect to see results like below:
- Disparity/Depth:
  <p align="center">
    <img src="assets/disp_vis.png" alt="Disparity Visualization" width="100%">
  </p>

- Point cloud:
  <p align="center">
    <img src="assets/pcl_vis.png" alt="Point Cloud Visualization" width="100%">
  </p>


# ONNX/TRT
For TRT, we recommend first setup env in docker.

```
python scripts/make_onnx.py --model_dir weights/23-36-37/model_best_bp2_serialize.pth --save_path output/ --height 448 --width 640 --valid_iters 8 --max_disp 192
```

| Flag              | Meaning                                                                  |
|-------------------|--------------------------------------------------------------------------|
| `--model_dir`     | Path to the trained weights/model file                                   |
| `--save_path`     | Directory to save ONNX outputs and zip file                             |
| `--height`        | Input image height, better to be divisible by 32. Reduce image size can increase speed.                                          |
| `--width`         | Input image width,  better to be divisible by 32. Reduce image size can increase speed.                                     |
| `--valid_iters`   | Number of updates during forward pass, reduce it for faster speed, but may drop quality                                    |
| `--max_disp`      | Maximum disparity for volume encoding, 192 should be enough, unless you need to sense very near objects (e.g. <0.1m). Increasing it runs slower and uses more memory.                                    |

Refer to `scripts/make_onnx.py` for a comprehensive list of available flags.  Since some intermediate operation is not supported by TRT conversion. We split around it into 2 onnx files.

Then convert from ONNX to TRT as below.
```
trtexec --onnx=output/feature_runner.onnx --saveEngine=output/feature_runner.engine --fp16  --useCudaGraph
trtexec --onnx=output/post_runner.onnx --saveEngine=output/post_runner.engine --fp16  --useCudaGraph
```

To use TRT for inference:
```
python scripts/run_demo_tensorrt.py --onnx_dir output/ --left_file demo_data/left.png --right_file demo_data/right.png --intrinsic_file demo_data/K.txt --out_dir output/ --remove_invisible 0 --denoise_cloud 1  --get_pc 1 --zfar 100
```

# Internet-Scale Pseudo-Labeling
Real-world data offers greater diversity and realism than synthetic data. However, obtaining real stereo images with ground-truth metric depth annotation is notoriously difficult. To address this challenge, we propose an automatic data curation pipeline to generate pseudo-labels on internet-scale stereo images from [Stereo4D](https://stereo4d.github.io/) dataset. **Top:** Pseudo-labeling pipeline on in-the-wild internet stereo data. **Bottom:** Visualization of our generated pseudo-labels.

<p align="center">
  <img src="assets/stereo4d.jpg" width="50%">
</p>

Below are visualizations of the intermediate results in our pseudo-labeling process. In the rightmost column, green checkmark or red cross denotes whether samples are kept for training or not, based on the percentage of positive pixels in the consistency mask. Our data curation process can automatically discover failures on noisy internet data such as images containing subtitle (bottom), mosaic (2nd last row) and overly challenging samples that are unsuitable for training (top). The final pseudo-labels can also correct erroneous predictions from FoundationStereo on sky regions (5th row).

<p align="center">
  <img src="assets/stereo4d_labeling.jpg" width="100%">
</p>


The dataset is available at HuggingFace: https://huggingface.co/datasets/nvidia/ffs_stereo4d

# Citation
```bibtex
@article{wen2026fastfoundationstereo,
  title={{Fast-FoundationStereo}: Real-Time Zero-Shot Stereo Matching},
  author={Bowen Wen and Shaurya Dewan and Stan Birchfield},
  journal={CVPR},
  year={2026}
}
```

# Contact
Please contact [Bowen Wen](https://wenbowen123.github.io/) (bowenw@nvidia.com) for questions and commercial inquiries.

# Acknowledgement
We would like to thank Xutong Ren, Karsten Patzwaldt, Yonggan Fu, Saurav Muralidharan, Han Cai, Pavlo Molchanov, Yu Wang, Varun Praveen, Joseph Aribido and Jun Gao for their insightful early discussions for this project. We would also like to thank NVIDIA Isaac and TAO teams for their engineering support and valuable discussions. Thanks to the authors of [FoundationStereo](https://github.com/NVlabs/FoundationStereo), [Selective-IGEV](https://github.com/Windsrain/Selective-Stereo), [Stereo4D](https://github.com/Stereo4d/stereo4d-code) and [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) for their code release. Finally, thanks to CVPR reviewers and AC for their appreciation of this work and constructive feedback.