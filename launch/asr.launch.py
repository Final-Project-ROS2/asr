from launch import LaunchDescription
from launch_ros.actions import Node

import os

def generate_launch_description():
    ld = LaunchDescription()

    asr_node = Node(
        package='asr',
        executable='asr_service',
        name='asr_node',
        output='screen',
        emulate_tty=True,
    )
    ld.add_action(asr_node)

    asr_connector_node = Node(
        package='asr',
        executable='asr_connector',
        name='asr_connector_node',
        output='screen',
        emulate_tty=True,
    )
    ld.add_action(asr_connector_node)

    return ld
