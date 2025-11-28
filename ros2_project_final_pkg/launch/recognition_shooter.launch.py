from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 弹丸击打节点
        Node(
            package='ros2_project_final_pkg',
            executable='ShooterNode',
            name='shooter_node',
            output='screen'
        ),
        # 视觉识别节点
        Node(
            package='ros2_project_final_pkg',
            executable='RecognitionNode',
            name='recognition_node',
            output='screen'
        )
    ])
