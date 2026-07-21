from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class M1RoughCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 2048
        num_one_step_observations = 3 + 3 + 3 + 16 + 16 + 16
        num_observations = num_one_step_observations * 6
        num_one_step_privileged_obs = num_one_step_observations + 3 + 3 + 11 * 17 + 12
        num_privileged_obs = num_one_step_privileged_obs
        num_actions = 16

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        # mesh_type = "trimesh"
        static_friction = 1.0
        dynamic_friction = 1.0
        # terrain_proportions = [0.1, 0.2, 0.3, 0.3, 0.1]
        terrain_proportions = [0.1, 0.1, 0.4, 0.4, 0.0]


    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.2
        num_commands = 4
        resampling_time = 10.0
        heading_command = True
        y_only_env_ratio = 0.1
        yaw_only_env_ratio = 0.1

        class ranges:
            lin_vel_x = [-1.5, 1.5]
            lin_vel_y = [-0.6, 0.6]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]

        class stair_ranges:
            lin_vel_x = [-0.8, 0.8]
            lin_vel_y = [-0.2, 0.2]
            ang_vel_yaw = [-0.2, 0.2]
            heading = [-0.2, 0.2]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.70]
        default_joint_angles = {
            "LF_HAA": 0.0,
            "LF_HFE": 0.8,
            "LF_KNEE": -1.28,
            "LF_WHEEL": 0.0,
            "RF_HAA": 0.0,
            "RF_HFE": 0.8,
            "RF_KNEE": -1.28,
            "RF_WHEEL": 0.0,
            "LH_HAA": 0.0,
            "LH_HFE": -0.8,
            "LH_KNEE": 1.28,
            "LH_WHEEL": 0.0,
            "RH_HAA": 0.0,
            "RH_HFE": -0.8,
            "RH_KNEE": 1.28,
            "RH_WHEEL": 0.0,
        }

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {"HAA": 40.0, "HFE": 40.0, "KNEE": 40.0, "WHEEL": 0.0}
        damping = {"HAA": 2.0, "HFE": 2.0, "KNEE": 2.0, "WHEEL": 1.0}
        action_scale = 0.25
        vel_scale = 5.0
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/m1/urdf/m1.urdf"
        name = "m1"
        foot_name = "FOOT"
        wheel_name = ["WHEEL"]
        penalize_contacts_on = ["abad", "thigh", "shank", "trunk"]
        terminate_after_contacts_on = ["trunk"]
        collapse_fixed_joints = True
        fix_base_link = False
        default_dof_drive_mode = 3
        self_collisions = 0
        replace_cylinder_with_capsule = False
        flip_visual_attachments = False

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_payload_mass = True
        payload_mass_range = [-3.0, 5.0]

        randomize_com_displacement = True
        com_displacement_range = [-0.05, 0.05]

        randomize_link_mass = True
        link_mass_range = [0.8, 1.2]

        randomize_friction = True
        friction_range = [0.2, 1.25]

        randomize_restitution = True
        restitution_range = [0.0, 1.0]

        randomize_motor_strength = True
        motor_strength_range = [0.85, 1.15]

        randomize_kp = True
        kp_range = [0.85, 1.15]

        randomize_kd = True
        kd_range = [0.85, 1.15]

        randomize_initial_joint_pos = False

        disturbance = True
        disturbance_range = [-30.0, 30.0]
        disturbance_interval = 10

        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 1.0

        delay = True

    class rewards(LeggedRobotCfg.rewards):
        class scales:
            tracking_lin_vel = 2.0
            tracking_ang_vel = 1.0
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -0.5
            base_height = -5.0
            hip_default = -0.5
            stand_still = -1.0 #0.5
            collision = -1.0
            feet_stumble = -0.1
            action_rate = -0.01
            torques = -5.0e-6
            dof_vel = -1e-5
            dof_acc = -1e-7
            run_still = -0.05
            feet_distance_y_exp = 0.5
            feet_air_time = 1.5
            feet_height_body = 0.0
            gait = 0.6 
            wheel_vel_penalty = -1.0

        only_positive_rewards = True
        tracking_sigma = 0.20
        soft_dof_pos_limit = 1.0
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        base_height_target = 0.54
        max_contact_force = 100.0
        feet_distance_y_target = 0.56
        feet_distance_y_sigma = 0.08
        feet_air_time_target = 0.7
        feet_height_body_target = 0.1
        feet_height_body_sigma = 0.05
        gait_std = 0.2
        gait_max_err = 0.5 #clipped
        gait_lateral_cmd_threshold = 0.2 # y command threshold for gait reward
        gait_yaw_cmd_threshold = 0.2 # yaw command threshold for gait reward
        gait_synced_feet_pair_labels = [["LF", "RH"], ["LH", "RF"]]
        wheel_vel_penalty_velocity_threshold = 0.15
        wheel_vel_penalty_command_threshold = 0.15


class M1RoughCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.005

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "HIMActorCritic"
        save_interval = 100
        num_steps_per_env = 48
        max_iterations = 50000
        experiment_name = "M1_HIM"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None


class M1RoughCfgPPONoCmd(M1RoughCfgPPO):
    class runner(M1RoughCfgPPO.runner):
        policy_class_name = "HIMActorCriticNoCmd"
        experiment_name = "M1_HIM_NoCmd"
