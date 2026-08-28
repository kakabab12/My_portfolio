from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

FOLLOWER_PORT = "/dev/ttyACM1"
LEADER_PORT = "/dev/ttyACM2"

follower_cfg = SO101FollowerConfig(
    port=FOLLOWER_PORT,
    id="my_follower",
)

leader_cfg = SO101LeaderConfig(
    port=LEADER_PORT,
    id="my_leader",
)

follower = SO101Follower(follower_cfg)
leader = SO101Leader(leader_cfg)

try:
    print("Connecting leader...")
    leader.connect()

    print("Connecting follower...")
    follower.connect()

    print("START teleoperation")
    print("Ctrl + C 로 종료")

    while True:
        action = leader.get_action()
        follower.send_action(action)

except KeyboardInterrupt:
    print("\nStopped by user")

finally:
    try:
        leader.disconnect()
    except Exception:
        pass

    try:
        follower.disconnect()
    except Exception:
        pass

    print("Disconnected")
