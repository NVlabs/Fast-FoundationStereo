"""Launch Fast-FoundationStereo stereo depth inference against a ZED camera.

The ZED topics are derived from the `camera_name` argument and follow the
zed-ros2-wrapper layout `/<camera_name>/zed_node/<side>/...`. Override the
individual topic parameters (or pass a custom params_file) for other cameras.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    camera_name = LaunchConfiguration('camera_name')

    def zed_topic(side, leaf):
        return ParameterValue(
            ['/', camera_name, '/zed_node/', side, '/', leaf], value_type=str)

    return LaunchDescription([
        DeclareLaunchArgument(
            'engine_dir', default_value='/engines',
            description='Directory with the TensorRT engine(s) and onnx.yaml'),
        DeclareLaunchArgument(
            'camera_name', default_value='zed',
            description='ZED camera name used by zed-ros2-wrapper topic namespacing'),
        DeclareLaunchArgument(
            'publish_disparity', default_value='false',
            description='Also publish raw disparity as 32FC1 image'),
        DeclareLaunchArgument(
            'params_file',
            default_value=PathJoinSubstitution(
                [FindPackageShare('ffs_ros'), 'config', 'ffs_stereo.yaml']),
            description='YAML file with additional node parameters'),
        Node(
            package='ffs_ros',
            executable='ffs_stereo_node',
            name='ffs_stereo',
            output='screen',
            parameters=[
                LaunchConfiguration('params_file'),
                {
                    'engine_dir': ParameterValue(
                        LaunchConfiguration('engine_dir'), value_type=str),
                    'left_image_topic': zed_topic('left', 'image_rect_color'),
                    'right_image_topic': zed_topic('right', 'image_rect_color'),
                    'left_camera_info_topic': zed_topic('left', 'camera_info'),
                    'right_camera_info_topic': zed_topic('right', 'camera_info'),
                    'publish_disparity': ParameterValue(
                        LaunchConfiguration('publish_disparity'), value_type=bool),
                },
            ],
        ),
    ])
