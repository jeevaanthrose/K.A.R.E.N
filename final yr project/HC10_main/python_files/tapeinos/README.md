# Tapeinos Setup

Create a root-level `bash.source` file before running the app. This file is used by the dashboard and camera tools to source your ROS 2 workspaces.

Example:

```bash
source /opt/ros/humble/setup.bash
source $HOME/colcon_ws/install/local_setup.bash
source $HOME/micro_ros_agent_ws/install/local_setup.bash
source /home/blessinggrace/hc10dtp_motoXY/install/setup.bash
```

The file must be located in this project directory:

```bash
/home/blessinggrace/tapeinos-main/bash.source
```

If you use camera tracking, also create or refresh `sensors/camera/clean_env.json` from a working terminal session:

```bash
python3 -c "import os, json; print(json.dumps(dict(os.environ)))" > /home/blessinggrace/tapeinos-main/sensors/camera/clean_env.json
```

Run the app from the project root:

```bash
python3 app.py
```

On first startup, the app will create:

```bash
/home/blessinggrace/tapeinos-main/resources
```

## HC10 Defaults

This checkout is configured for a Yaskawa HC10 / MotoROS2 graph by default.
The jog runner expects:

```bash
/yaskawa_hc10/joint_states
/yaskawa_hc10/ctrl_groups/r1/joint_states
/yaskawa_hc10/robot_status
/yaskawa_hc10/write_single_io
/move_action
/execute_trajectory
/compute_cartesian_path
```

The default HC10 joint order is:

```bash
joint_1_s joint_2_l joint_3_u joint_4_r joint_5_b joint_6_t
```

If your launch files use different names, override them before starting Flask:

```bash
export TAPEINOS_ROBOT_NS=/yaskawa_hc10
export TAPEINOS_JOINT_NAMES=joint_1_s,joint_2_l,joint_3_u,joint_4_r,joint_5_b,joint_6_t
export TAPEINOS_JOINT_STATE_TOPICS=/yaskawa_hc10/joint_states,/yaskawa_hc10/ctrl_groups/r1/joint_states,/joint_states
export TAPEINOS_MOVE_GROUP=manipulator
export TAPEINOS_BASE_FRAME=base_link
export TAPEINOS_EEF_FRAME=tool0
```
