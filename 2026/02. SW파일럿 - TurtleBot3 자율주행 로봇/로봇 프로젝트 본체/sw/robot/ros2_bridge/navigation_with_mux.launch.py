"""TurtleBot3 Nav2/AMCL과 이 프로젝트 전용 RViz를 실행한다.

Nav2의 속도 메시지를 ``geometry_msgs/Twist``로 고정한다. TurtleBot3 OpenCR은
Twist를 받으며, cmd_vel_mux가 이를 /cmd_vel_muxed로 안전하게 전달한다.
"""
import os
import math

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


# A/B Nav2 구간도 C/D 수동 구간과 같은 "목표점 근처면 도착" 기준을 쓴다.
# 좌표 오차가 10cm 이내면 도착으로 보고, waypoint 통과에 불필요한 제자리
# 방향 맞춤은 하지 않는다. XY 반경을 한 번 통과한 사실이 이후에도 고정되지
# 않도록 goal checker와 DWB의 stateful 모드를 끈다.
WAYPOINT_XY_GOAL_TOLERANCE_M = 0.10
WAYPOINT_YAW_GOAL_TOLERANCE_RAD = 2.0 * math.pi
# 이 지도는 44x43(2.20x2.15m)로 작고 B가 우측/상단 경계에서 각각 약
# 0.43m, 0.33m밖에 떨어져 있지 않다. TurtleBot3 기본 NavFn tolerance 0.5m는
# 목표 후보를 지도 밖까지 확장하고, 막힌 목표를 30cm 이상 떨어진 지점의
# 성공으로 반환할 수 있다. 실제 도착 검증과 같은 반경만 허용한다.
NAVFN_GOAL_TOLERANCE_M = WAYPOINT_XY_GOAL_TOLERANCE_M

# 시연 지도에는 벽과 고정 장애물이 이미 들어 있다. 전역 코스트맵에 /scan을
# obstacle+voxel 두 레이어로 다시 표시하면 AMCL/scan 한 셀 오차만으로도 A-B의
# 좁은 통로가 닫힌다. 전역 계획은 static+inflation으로 고정하고, 실시간 장애물은
# 로컬 voxel 레이어와 별도 Safety supervisor가 처리한다.
GLOBAL_LIVE_OBSTACLE_LAYERS_ENABLED = False
LOCAL_DUPLICATE_OBSTACLE_LAYER_ENABLED = False
# Safety supervisor가 갑작스러운 장애물의 정지·취소·후진을 전담한다. Nav2의
# 기본 recovery 트리까지 동시에 켜면 잘못된 초기 위치나 Safety hold 상태에서
# spin/backup이 경쟁하므로, 경로 생성·추종만 수행하는 트리를 사용한다.
SAFE_NAV_TO_POSE_BT_XML = os.path.join(
    get_package_share_directory("nav2_bt_navigator"),
    "behavior_trees", "navigate_w_replanning_only_if_path_becomes_invalid.xml")


def generate_launch_description():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    navigation_share = get_package_share_directory("turtlebot3_navigation2")
    nav2_bringup_launch = os.path.join(
        get_package_share_directory("nav2_bringup"), "launch", "bringup_launch.py")
    model = os.environ.get("TURTLEBOT3_MODEL", "burger")
    # 이 PC는 ROS 2 Humble이다. Humble 파라미터에는 Humble 바이너리가 제공하는
    # ``nav2_navfn_planner/NavfnPlanner`` 플러그인 이름이 들어 있다. 루트 param
    # 파일은 신형 Nav2의 ``::`` 이름을 써서 Humble에서 planner가 기동하지 않는다.
    humble_params_file = os.path.join(
        navigation_share, "param", "humble", f"{model}.yaml")
    fallback_params_file = os.path.join(navigation_share, "param", f"{model}.yaml")
    params_file = (humble_params_file if os.path.isfile(humble_params_file)
                   else fallback_params_file)
    rviz_config = os.path.join(root_dir, "configs", "turtlebot3_navigation.rviz")

    # Humble Nav2와 TurtleBot3 OpenCR는 geometry_msgs/Twist를 사용한다. 혹시
    # 설치본에 Stamped Twist 옵션이 있더라도 false로 덮어 타입을 하나로 고정한다.
    configured_params = RewrittenYaml(
        source_file=params_file,
        param_rewrites={
            "use_sim_time": "false",
            "enable_stamped_cmd_vel": "false",
            "bt_navigator.ros__parameters.default_bt_xml_filename":
                SAFE_NAV_TO_POSE_BT_XML,
            # Humble 파라미터 파일(general_goal_checker)과 구형 fallback 파일
            # (goal_checker)을 모두 지원한다. 존재하는 경로만 RewrittenYaml이 바꾼다.
            "controller_server.ros__parameters.general_goal_checker.xy_goal_tolerance":
                str(WAYPOINT_XY_GOAL_TOLERANCE_M),
            "controller_server.ros__parameters.general_goal_checker.yaw_goal_tolerance":
                str(WAYPOINT_YAW_GOAL_TOLERANCE_RAD),
            "controller_server.ros__parameters.general_goal_checker.stateful":
                "false",
            "controller_server.ros__parameters.goal_checker.xy_goal_tolerance":
                str(WAYPOINT_XY_GOAL_TOLERANCE_M),
            "controller_server.ros__parameters.goal_checker.yaw_goal_tolerance":
                str(WAYPOINT_YAW_GOAL_TOLERANCE_RAD),
            "controller_server.ros__parameters.goal_checker.stateful":
                "false",
            # DWB의 goal-distance 평가도 같은 반경을 기준으로 맞춘다.
            "controller_server.ros__parameters.FollowPath.xy_goal_tolerance":
                str(WAYPOINT_XY_GOAL_TOLERANCE_M),
            "controller_server.ros__parameters.FollowPath.stateful": "false",
            # 작은 지도 경계를 넘는 0.5m 후보 검색과 원거리 false-success를
            # 막고, 미션의 실제 도착 검증 반경과 동일하게 유지한다.
            "planner_server.ros__parameters.GridBased.tolerance":
                str(NAVFN_GOAL_TOLERANCE_M),
            # 전역 지도에는 정적 장애물이 이미 있으므로 live scan을 중복
            # 적층하지 않는다. 로컬 voxel과 Safety는 그대로 활성 상태다.
            "global_costmap.global_costmap.ros__parameters.obstacle_layer.enabled":
                str(GLOBAL_LIVE_OBSTACLE_LAYERS_ENABLED).lower(),
            "global_costmap.global_costmap.ros__parameters.voxel_layer.enabled":
                str(GLOBAL_LIVE_OBSTACLE_LAYERS_ENABLED).lower(),
            # 로컬 costmap도 같은 scan을 obstacle/voxel 양쪽에 넣지 않고 3D
            # voxel 레이어 하나만 사용해 장애물 표시와 clearing을 담당한다.
            "local_costmap.local_costmap.ros__parameters.obstacle_layer.enabled":
                str(LOCAL_DUPLICATE_OBSTACLE_LAYER_ENABLED).lower(),
        },
        convert_types=True,
    )

    return LaunchDescription([
        DeclareLaunchArgument("map", description="Absolute path to the Nav2 map yaml"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav2_bringup_launch),
            launch_arguments={
                "map": LaunchConfiguration("map"),
                "use_sim_time": "false",
                "params_file": configured_params,
                "autostart": "true",
                # nav2_bringup 내부가 PythonExpression으로 평가하므로 Python의
                # bool 표기(True)를 써야 한다. 소문자 true는 NameError가 난다.
                "use_composition": "True",
            }.items(),
        ),
        # 기본 TurtleBot3 설정의 폐기된 RViz 패널(Selector/Docking)을 쓰지 않는다.
        # 이 설정은 map 프레임과 /map 표시를 고정하고 실제 지도 크기에 맞춘다.
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", rviz_config],
            parameters=[{"use_sim_time": False}],
            output="screen",
        ),
    ])
