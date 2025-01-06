"""Launch the ball hitting demo

   ros2 launch flyswatter hitball.launch.py

"""

import os

from ament_index_python.packages import get_package_share_directory as pkgdir

from launch                      import LaunchDescription
from launch.actions              import Shutdown
from launch_ros.actions          import Node


#
# Generate the Launch Description
#
def generate_launch_description():

    ######################################################################
    # LOCATE FILES

    # Locate the RVIZ configuration file.
    rvizcfg = os.path.join(pkgdir('flyswatter'), 'rviz/flyswatter.rviz')

    # Locate the URDF file.
    urdf = os.path.join(pkgdir('flyswatter'), 'urdf/ur10_robot.urdf')

    # Load the robot's URDF file (XML).
    with open(urdf, 'r') as file:
        robot_description = file.read()


    ######################################################################
    # PREPARE THE LAUNCH ELEMENTS

    # Configure a node for the point_publisher.
    node_demo = Node(
        name       = 'ball',
        package    = 'flyswatter',
        executable = 'ball',
        output     = 'screen',
        on_exit    = Shutdown())

    # Configure a node for the robot_state_publisher.
    node_robot_state_publisher = Node(
        name       = 'robot_state_publisher',
        package    = 'robot_state_publisher',
        executable = 'robot_state_publisher',
        output     = 'screen',
        parameters = [{'robot_description': robot_description}])

    # Configure a node for joint state publisher gui
    node_joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen')

    # Configure a node for the joint trajectory
    node_trajectory = Node(
        name       = 'trajectory',
        package    = 'flyswatter',
        executable = 'robot',
        output     = 'screen')

    # Configure a node for RVIZ
    node_rviz = Node(
        name       = 'rviz',
        package    = 'rviz2',
        executable = 'rviz2',
        output     = 'screen',
        arguments  = ['-d', rvizcfg],
        on_exit    = Shutdown())


    ######################################################################
    # RETURN THE ELEMENTS IN ONE LIST

    return LaunchDescription([

        # Start the demo and RVIZ
        node_demo,
        node_rviz,
        node_robot_state_publisher,
        node_trajectory,
        # node_joint_state_publisher_gui
    ])
