import argparse
import multiprocessing as mp
import os
import re
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import yaml
from pynput import keyboard

from joystick_interface import JoystickInterface
from deploy_mujoco_m1_torque_debug import (
    DEBUG_RECORD_ROOT,
    MATPLOTLIB_IMPORT_ERROR,
    PIL_IMPORT_ERROR,
    Image,
    TorqueMonitor,
    make_timestamped_record_dir,
    overlay_timestamp_on_frame,
    vel_torque_plot_worker,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MUJOCO_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = MUJOCO_DIR / "configs" / "m1_no_cmd.yaml"


class KeyboardState:
    def __init__(self):
        self.forward = False
        self.backward = False
        self.left = False
        self.right = False
        self.strafe_left = False
        self.strafe_right = False
        self.pending_reset = False
        self.clear_command = False


def quat_rotate_inverse(q, v):
    q_w = q[0]
    q_vec = q[1:4]
    a = v * (2.0 * q_w * q_w - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def resolve_repo_path(raw_path):
    root_str = str(PROJECT_ROOT)
    return str(Path(raw_path.replace("{LEGGED_GYM_ROOT_DIR}", root_str)).resolve())


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["config_path"] = str(Path(config_path).resolve())
    cfg["policy_path"] = resolve_repo_path(cfg["policy_path"])
    cfg["xml_path"] = resolve_repo_path(cfg["xml_path"])
    return cfg


def build_index_map(source_names, target_names, label):
    indices = []
    missing = []
    for name in target_names:
        if name in source_names:
            indices.append(source_names.index(name))
        else:
            missing.append(name)

    if missing:
        fallback = list(range(min(len(source_names), len(target_names))))
        print(f"[Warning] {label} missing names: {missing}")
        print(f"[Warning] Falling back to sequential mapping for {label}: {fallback}")
        return fallback
    return indices


class DeployM1NoCmd:
    def __init__(
        self,
        cfg,
        onnx_override=None,
        history_seconds=10.0,
        plot_refresh_interval=0.1,
        enable_monitor=False,
        fixed_cmd=None,
    ):
        self.cfg = cfg
        self.enable_monitor = bool(enable_monitor)
        self.fixed_cmd = None if fixed_cmd is None else np.asarray(fixed_cmd, dtype=np.float32)
        self.last_raw_tau = np.zeros(0, dtype=np.float32)
        self.recording_active = False
        self.recorded_dof_vel = []
        self.recorded_raw_tau = []
        self.recorded_clipped_tau = []
        self.reference_gear_ratios = np.zeros(0, dtype=np.float32)
        self.plot_processes = []
        self.recording_timestamp = None
        self.recording_output_dir = None
        self.recording_start_sim_time = 0.0
        self.video_fps = 30.0
        self.next_video_capture_time = 0.0
        self.video_frames = []
        self.video_frame_wall_times = []
        self.video_renderer = None
        self.torque_monitor = None
        self.keyboard_state = KeyboardState()
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self.listener.start()

        policy_path = onnx_override if onnx_override is not None else cfg["policy_path"]
        self.policy_path = str(Path(policy_path).resolve())
        self.xml_path = cfg["xml_path"]

        if not os.path.exists(self.xml_path):
            raise FileNotFoundError(f"MuJoCo scene not found: {self.xml_path}")
        if not os.path.exists(self.policy_path):
            raise FileNotFoundError(f"ONNX policy not found: {self.policy_path}")

        self.simulation_duration = float(cfg["simulation_duration"])
        self.simulation_dt = float(cfg["simulation_dt"])
        self.control_decimation = int(cfg["control_decimation"])
        self.num_actions = int(cfg["num_actions"])
        self.num_obs = int(cfg["num_obs"])
        self.history_length = int(cfg.get("history_length", 6))
        self.cmd_slice = cfg.get("cmd_slice", [6, 9])
        self.cmd_start = int(self.cmd_slice[0])
        self.cmd_end = int(self.cmd_slice[1])
        self.num_obs_no_cmd = self.num_obs - (self.cmd_end - self.cmd_start)
        self.expected_obs_history_no_cmd_dim = int(
            cfg.get(
                "obs_history_no_cmd_dim",
                self.history_length * self.num_obs_no_cmd,
            )
        )
        self.log_interval_steps = int(cfg["log_interval_steps"])
        self.torque_warning_cooldown_steps = int(cfg["torque_warning_cooldown_steps"])

        self.kps = np.array(cfg["kps"], dtype=np.float32)
        self.kds = np.array(cfg["kds"], dtype=np.float32)
        self.default_angles = np.array(cfg["default_angles"], dtype=np.float32)
        self.torque_limits = np.array(cfg["torque_limits"], dtype=np.float32)
        self.train_dof_names = list(cfg["train_dof_names"])
        self.wheel_policy_indices = np.array(cfg["wheel_policy_indices"], dtype=np.int64)
        self.cmd_scale = np.array(cfg["cmd_scale"], dtype=np.float32)
        self.cmd = np.array(cfg["cmd_init"], dtype=np.float32)

        self.ang_vel_scale = float(cfg["ang_vel_scale"])
        self.dof_pos_scale = float(cfg["dof_pos_scale"])
        self.dof_vel_scale = float(cfg["dof_vel_scale"])
        self.action_scale = float(cfg["action_scale"])
        self.vel_scale = float(cfg["vel_scale"])

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.simulation_dt

        self.ort_session = ort.InferenceSession(
            self.policy_path, providers=["CPUExecutionProvider"]
        )
        self.input_infos = self.ort_session.get_inputs()
        self.output_name = self.ort_session.get_outputs()[0].name
        self.output_dim = self._infer_last_dim(self.ort_session.get_outputs()[0].shape)

        if len(self.input_infos) != 2:
            raise RuntimeError(
                f"NoCmd ONNX expects 2 inputs (obs_history_no_cmd, obs_curr), "
                f"got {len(self.input_infos)}: {[i.name for i in self.input_infos]}"
            )

        self.history_input_name = self.input_infos[0].name
        self.curr_input_name = self.input_infos[1].name
        self.history_input_dim = self._infer_last_dim(self.input_infos[0].shape)
        self.curr_input_dim = self._infer_last_dim(self.input_infos[1].shape)

        # Prefer names if present
        for info in self.input_infos:
            dim = self._infer_last_dim(info.shape)
            if info.name == "obs_history_no_cmd" or dim == self.expected_obs_history_no_cmd_dim:
                self.history_input_name = info.name
                self.history_input_dim = dim
            if info.name == "obs_curr" or dim == self.num_obs:
                self.curr_input_name = info.name
                self.curr_input_dim = dim

        if self.history_input_dim != self.expected_obs_history_no_cmd_dim:
            print(
                f"[Warning] ONNX history input dim is {self.history_input_dim}, expected "
                f"{self.expected_obs_history_no_cmd_dim}."
            )
        if self.curr_input_dim != self.num_obs:
            print(
                f"[Warning] ONNX current-obs input dim is {self.curr_input_dim}, expected "
                f"{self.num_obs}."
            )
        if self.output_dim != self.num_actions:
            print(
                f"[Warning] ONNX output dim is {self.output_dim}, expected {self.num_actions}. "
                "The deploy script will pad or truncate actions."
            )

        self.joystick = JoystickInterface(
            device_path=cfg["joystick_device"],
            max_v_x=float(cfg["joystick_max_v_x"]),
            max_v_y=float(cfg["joystick_max_v_y"]),
            max_omega=float(cfg["joystick_max_omega"]),
        )

        self.qpos_policy = np.zeros(self.num_actions, dtype=np.float32)
        self.qvel_policy = np.zeros(self.num_actions, dtype=np.float32)
        self.base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.base_lin_vel = np.zeros(3, dtype=np.float32)
        self.projected_gravity = np.zeros(3, dtype=np.float32)
        self.action_policy = np.zeros(self.num_actions, dtype=np.float32)
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        # Direct no-cmd history roll buffer for ONNX estimator input (no full 342 buffer).
        self.obs_history_no_cmd = np.zeros(
            self.expected_obs_history_no_cmd_dim, dtype=np.float32
        )
        self.ctrl_policy = np.zeros(self.num_actions, dtype=np.float32)
        self.last_warn_step = {}
        self.control_step = 0
        self.missing_sensor_warnings = set()

        self._setup_mappings()
        self._print_startup_summary()
        self.reset(full_reset=True)

        self.last_raw_tau = np.zeros(self.num_actions, dtype=np.float32)
        self.reference_gear_ratios = np.array([1.0, 1.0, 2.5, 1.0] * 4, dtype=np.float32)
        if self.enable_monitor:
            wheel_names = [self.train_dof_names[idx] for idx in self.wheel_policy_indices.tolist()]
            self.torque_monitor = TorqueMonitor(
                joint_names=self.train_dof_names,
                torque_limits=self.torque_limits,
                default_selected=wheel_names,
                history_seconds=history_seconds,
                redraw_interval=plot_refresh_interval,
                sample_dt=self.simulation_dt,
            )
        print("[M1 NoCmd Deploy] Debug hotkeys: O start record, P stop record and save plots.")
        print(f"[M1 NoCmd Deploy] Debug save root: {DEBUG_RECORD_ROOT}")
        if self.fixed_cmd is not None:
            print(
                "[M1 NoCmd Deploy] Fixed command mode: "
                f"vx={self.fixed_cmd[0]:+.3f}, vy={self.fixed_cmd[1]:+.3f}, yaw={self.fixed_cmd[2]:+.3f}"
            )
        if self.torque_monitor is None:
            print("[M1 NoCmd Deploy] Torque monitor: disabled by default. Use --enable-monitor to turn it on.")
        else:
            print(
                "[M1 NoCmd Deploy] Torque monitor: "
                f"raw_tau only, {self.torque_monitor.history_seconds:.1f}s rolling window, "
                "wheel joints selected by default."
            )

    def _infer_last_dim(self, shape):
        if isinstance(shape[-1], int):
            return int(shape[-1])
        return -1

    def _setup_mappings(self):
        self.mujoco_joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            for joint_id in range(self.model.njnt)
            if self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
        ]

        self.actuator_joint_names = []
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            self.actuator_joint_names.append(
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            )

        self.policy_to_qpos = build_index_map(
            self.mujoco_joint_names, self.train_dof_names, "qpos-order mapping"
        )
        self.policy_to_actuator = build_index_map(
            self.actuator_joint_names, self.train_dof_names, "actuator-order mapping"
        )

        if len(self.mujoco_joint_names) != self.num_actions:
            print(
                f"[Warning] MuJoCo joint count is {len(self.mujoco_joint_names)}, "
                f"while config expects {self.num_actions} actions."
            )
        if self.model.nu != self.num_actions:
            print(
                f"[Warning] MuJoCo actuator count is {self.model.nu}, "
                f"while config expects {self.num_actions} actions."
            )

        self.actuator_to_policy = [-1] * len(self.actuator_joint_names)
        for policy_idx, actuator_idx in enumerate(self.policy_to_actuator):
            if actuator_idx < len(self.actuator_to_policy):
                self.actuator_to_policy[actuator_idx] = policy_idx

    def _print_startup_summary(self):
        print(f"[M1 NoCmd Deploy] Config: {self.cfg['config_path']}")
        print(f"[M1 NoCmd Deploy] Scene : {self.xml_path}")
        print(f"[M1 NoCmd Deploy] ONNX  : {self.policy_path}")
        print(
            f"[M1 NoCmd Deploy] ONNX inputs: "
            f"{self.history_input_name}={self.history_input_dim}, "
            f"{self.curr_input_name}={self.curr_input_dim}"
        )
        print(f"[M1 NoCmd Deploy] ONNX output dim: {self.output_dim}")
        print(f"[M1 NoCmd Deploy] Train DOF order : {self.train_dof_names}")
        print(f"[M1 NoCmd Deploy] MuJoCo qpos order: {self.mujoco_joint_names}")
        print(f"[M1 NoCmd Deploy] MuJoCo actuator order: {self.actuator_joint_names}")
        print(f"[M1 NoCmd Deploy] policy_to_qpos    : {self.policy_to_qpos}")
        print(f"[M1 NoCmd Deploy] policy_to_actuator: {self.policy_to_actuator}")
        print(
            "[M1 NoCmd Deploy] Direct no-cmd history: "
            f"obs_curr={self.num_obs} (with cmd), "
            f"history_no_cmd={self.expected_obs_history_no_cmd_dim} "
            f"({self.history_length}x{self.num_obs_no_cmd}); "
            f"cmd slice=[{self.cmd_start}:{self.cmd_end}]"
        )
        print("[M1 NoCmd Deploy] Keyboard fallback: W/S vx, Q/E vy, A/D yaw, Space zero cmd, R reset")

    def _on_press(self, key):
        key_char = getattr(key, "char", None)
        if key_char is None:
            if key == keyboard.Key.space:
                self.keyboard_state.clear_command = True
            return

        key_char = key_char.lower()
        if key_char == "w":
            self.keyboard_state.forward = True
        elif key_char == "s":
            self.keyboard_state.backward = True
        elif key_char == "a":
            self.keyboard_state.left = True
        elif key_char == "d":
            self.keyboard_state.right = True
        elif key_char == "q":
            self.keyboard_state.strafe_left = True
        elif key_char == "e":
            self.keyboard_state.strafe_right = True
        elif key_char == "r":
            self.keyboard_state.pending_reset = True
        elif key_char == "o":
            self.keyboard_state.start_recording = True
        elif key_char == "p":
            self.keyboard_state.stop_recording = True

    def _on_release(self, key):
        key_char = getattr(key, "char", None)
        if key_char is None:
            return

        key_char = key_char.lower()
        if key_char == "w":
            self.keyboard_state.forward = False
        elif key_char == "s":
            self.keyboard_state.backward = False
        elif key_char == "a":
            self.keyboard_state.left = False
        elif key_char == "d":
            self.keyboard_state.right = False
        elif key_char == "q":
            self.keyboard_state.strafe_left = False
        elif key_char == "e":
            self.keyboard_state.strafe_right = False

    def _sync_command(self):
        if self.keyboard_state.clear_command:
            self.cmd[:] = 0.0
            self.keyboard_state.clear_command = False
            print("[M1 NoCmd Deploy] Cleared command.")
            return

        if self.fixed_cmd is not None:
            self.cmd[:] = self.fixed_cmd
            return

        if self.joystick.available:
            cmd_x, cmd_y, cmd_yaw = self.joystick.get_command()
            self.cmd[0] = cmd_x
            self.cmd[1] = cmd_y
            self.cmd[2] = cmd_yaw
            return

        max_vx = float(self.cfg["joystick_max_v_x"])
        max_vy = float(self.cfg["joystick_max_v_y"])
        max_yaw = float(self.cfg["joystick_max_omega"])

        self.cmd[0] = (
            max_vx if self.keyboard_state.forward else 0.0
        ) + (-max_vx if self.keyboard_state.backward else 0.0)
        self.cmd[1] = (
            max_vy if self.keyboard_state.strafe_left else 0.0
        ) + (-max_vy if self.keyboard_state.strafe_right else 0.0)
        self.cmd[2] = (
            max_yaw if self.keyboard_state.left else 0.0
        ) + (-max_yaw if self.keyboard_state.right else 0.0)

    def reset(self, full_reset=False):
        key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "default_pos"
        )
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            print("[Warning] Keyframe 'default_pos' not found. Falling back to mj_resetData.")
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[2] = 0.54
        mujoco.mj_forward(self.model, self.data)

        self.cmd[:] = 0.0
        self.action_policy[:] = 0.0
        self.ctrl_policy[:] = 0.0
        self.obs[:] = 0.0
        self.obs_history_no_cmd[:] = 0.0
        self.keyboard_state.pending_reset = False
        self.keyboard_state.clear_command = False
        self.last_warn_step.clear()
        if hasattr(self.keyboard_state, "start_recording"):
            self.keyboard_state.start_recording = False
            self.keyboard_state.stop_recording = False
        if hasattr(self, "last_raw_tau") and self.last_raw_tau.size == self.num_actions:
            self.last_raw_tau[:] = 0.0
        if hasattr(self, "next_video_capture_time"):
            self.next_video_capture_time = float(self.data.time)
        if getattr(self, "torque_monitor", None) is not None:
            self.torque_monitor.clear()
        if full_reset:
            self.control_step = 0
        print("[M1 NoCmd Deploy] Reset to default_pos keyframe.")

    def _get_sensor(self, sensor_name, fallback):
        try:
            return self.data.sensor(sensor_name).data.copy()
        except KeyError:
            if sensor_name not in self.missing_sensor_warnings:
                self.missing_sensor_warnings.add(sensor_name)
                print(f"[Warning] Sensor '{sensor_name}' is missing. Using fallback state.")
            return fallback

    def get_robot_state(self):
        qpos_mujoco = self.data.qpos[7:].astype(np.float32)
        qvel_mujoco = self.data.qvel[6:].astype(np.float32)
        self.qpos_policy = qpos_mujoco[self.policy_to_qpos]
        self.qvel_policy = qvel_mujoco[self.policy_to_qpos]

        self.base_quat = self._get_sensor(
            "body_quat", self.data.qpos[3:7].astype(np.float32)
        ).astype(np.float32)
        self.base_ang_vel = self._get_sensor(
            "body_gyro", self.data.qvel[3:6].astype(np.float32)
        ).astype(np.float32)

        world_lin_vel = self.data.qvel[:3].astype(np.float32)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, world_lin_vel).astype(
            np.float32
        )

        gravity_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.projected_gravity = quat_rotate_inverse(
            self.base_quat, gravity_vec
        ).astype(np.float32)

    def _strip_commands_one_step(self, obs):
        return np.concatenate(
            (obs[: self.cmd_start], obs[self.cmd_end :]), axis=-1
        ).astype(np.float32)

    def compute_observation(self):
        dof_err = self.qpos_policy - self.default_angles
        dof_err[self.wheel_policy_indices] = 0.0

        self.obs[0:3] = self.base_ang_vel * self.ang_vel_scale
        self.obs[3:6] = self.projected_gravity
        self.obs[6:9] = self.cmd * self.cmd_scale
        self.obs[9:25] = dof_err * self.dof_pos_scale
        self.obs[25:41] = self.qvel_policy * self.dof_vel_scale
        self.obs[41:57] = self.action_policy

        # Directly roll 54-dim no-cmd frames into history_no_cmd (324).
        obs_no_cmd = self._strip_commands_one_step(self.obs)
        self.obs_history_no_cmd[self.num_obs_no_cmd :] = self.obs_history_no_cmd[
            : -self.num_obs_no_cmd
        ].copy()
        self.obs_history_no_cmd[: self.num_obs_no_cmd] = obs_no_cmd

    def _prepare_policy_inputs(self):
        history = self.obs_history_no_cmd
        if self.history_input_dim != self.expected_obs_history_no_cmd_dim:
            if self.history_input_dim > self.expected_obs_history_no_cmd_dim:
                padded = np.zeros(self.history_input_dim, dtype=np.float32)
                padded[: self.expected_obs_history_no_cmd_dim] = history
                history = padded
            else:
                history = history[: self.history_input_dim]

        curr = self.obs
        if self.curr_input_dim != self.num_obs:
            if self.curr_input_dim > self.num_obs:
                padded = np.zeros(self.curr_input_dim, dtype=np.float32)
                padded[: self.num_obs] = curr
                curr = padded
            else:
                curr = curr[: self.curr_input_dim]

        return history.reshape(1, -1), curr.reshape(1, -1)

    def run_policy(self):
        history_input, curr_input = self._prepare_policy_inputs()
        action = self.ort_session.run(
            [self.output_name],
            {
                self.history_input_name: history_input.astype(np.float32),
                self.curr_input_name: curr_input.astype(np.float32),
            },
        )[0][0]

        if action.shape[0] != self.num_actions:
            fixed_action = np.zeros(self.num_actions, dtype=np.float32)
            copy_dim = min(action.shape[0], self.num_actions)
            fixed_action[:copy_dim] = action[:copy_dim]
            action = fixed_action

        self.action_policy = action.astype(np.float32)

    def compute_desired_targets(self):
        """SDK-style position/velocity PD targets (policy order: LF/LH/RF/RH).

        Legs:  q_des = default + action * action_scale,  qd_des = 0
        Wheels: q_des unused (kp=0),                    qd_des = action * vel_scale
        """
        q_des = self.default_angles.copy()
        qd_des = np.zeros(self.num_actions, dtype=np.float32)

        # leg position targets
        q_des = self.default_angles + self.action_policy * self.action_scale
        # wheels do not use position targets
        q_des[self.wheel_policy_indices] = self.default_angles[self.wheel_policy_indices]
        # wheel velocity targets
        qd_des[self.wheel_policy_indices] = (
            self.action_policy[self.wheel_policy_indices] * self.vel_scale
        )
        return q_des, qd_des

    def compute_torques(self):
        # Match SDK leg controller form:
        #   tau = kp * (q_des - q) + kd * (qd_des - qd)
        # then write force to MuJoCo <motor> actuators (same as SDK sim bridge).
        q_des, qd_des = self.compute_desired_targets()
        raw_tau = self.kps * (q_des - self.qpos_policy) + self.kds * (
            qd_des - self.qvel_policy
        )
        self.last_raw_tau = raw_tau.astype(np.float32)
        self._warn_if_torque_exceeds_limit(raw_tau)
        clipped_tau = np.clip(raw_tau, -self.torque_limits, self.torque_limits)
        self.ctrl_policy = clipped_tau.astype(np.float32)

        actuator_ctrl = np.zeros(self.model.nu, dtype=np.float32)
        for policy_idx, actuator_idx in enumerate(self.policy_to_actuator):
            if actuator_idx < len(actuator_ctrl):
                actuator_ctrl[actuator_idx] = clipped_tau[policy_idx]
        return actuator_ctrl

    def _warn_if_torque_exceeds_limit(self, raw_tau):
        over_limit = np.where(np.abs(raw_tau) > self.torque_limits)[0]
        for joint_idx in over_limit:
            last_warn = self.last_warn_step.get(joint_idx, -10**9)
            if self.control_step - last_warn < self.torque_warning_cooldown_steps:
                continue
            self.last_warn_step[joint_idx] = self.control_step
            print(
                f"[Warning] Torque limit exceeded on {self.train_dof_names[joint_idx]}: "
                f"raw={raw_tau[joint_idx]:+.3f}, limit={self.torque_limits[joint_idx]:.3f}"
            )

    def log_command_and_velocity(self):
        print(
            "[M1 NoCmd Deploy] "
            f"cmd(vx, vy, yaw)=({self.cmd[0]:+.3f}, {self.cmd[1]:+.3f}, {self.cmd[2]:+.3f}) | "
            f"actual(vx, vy, yaw)=({self.base_lin_vel[0]:+.3f}, {self.base_lin_vel[1]:+.3f}, {self.base_ang_vel[2]:+.3f})"
        )

    def _get_recording_kind(self):
        return "no_cmd_torque_debug"

    def _get_plot_filename(self):
        return "torque_scatter.png"

    def _begin_recording_artifacts(self):
        if PIL_IMPORT_ERROR is not None:
            raise ModuleNotFoundError(
                "Pillow is required for saving timestamped MuJoCo recordings."
            ) from PIL_IMPORT_ERROR
        self.recording_timestamp, self.recording_output_dir = make_timestamped_record_dir(
            self._get_recording_kind()
        )
        self.recording_start_sim_time = float(self.data.time)
        self.next_video_capture_time = float(self.data.time)
        self.video_frames = []
        self.video_frame_wall_times = []
        if self.video_renderer is None:
            self.video_renderer = self._create_video_renderer()
        print(f"[M1 NoCmd Deploy] Recording output dir: {self.recording_output_dir}")

    def _create_video_renderer(self):
        target_width = 1280
        target_height = 720
        try:
            return mujoco.Renderer(self.model, height=target_height, width=target_width)
        except ValueError as exc:
            message = str(exc)
            width_match = re.search(r"framebuffer width (\d+)", message)
            height_match = re.search(r"framebuffer height (\d+)", message)
            fallback_width = int(width_match.group(1)) if width_match else 640
            fallback_height = int(height_match.group(1)) if height_match else 480
            fallback_width = max(64, fallback_width)
            fallback_height = max(64, fallback_height)
            print(
                "[M1 NoCmd Deploy] Renderer size fallback: "
                f"requested {target_width}x{target_height}, "
                f"using {fallback_width}x{fallback_height} due to offscreen framebuffer limits."
            )
            return mujoco.Renderer(self.model, height=fallback_height, width=fallback_width)

    def _capture_video_frame_if_needed(self, viewer=None):
        if not self.recording_active or self.video_renderer is None:
            return
        current_time = float(self.data.time)
        if current_time + 1e-9 < self.next_video_capture_time:
            return

        camera = viewer.cam if viewer is not None else -1
        self.video_renderer.update_scene(self.data, camera=camera)
        frame_rgb = self.video_renderer.render()
        rel_time = current_time - self.recording_start_sim_time
        overlay = f"{self.recording_timestamp} | no_cmd | t={rel_time:06.3f}s"
        self.video_frames.append(overlay_timestamp_on_frame(frame_rgb, overlay))
        self.video_frame_wall_times.append(time.perf_counter())
        self.next_video_capture_time += 1.0 / self.video_fps

    def _finalize_recording_video(self):
        if PIL_IMPORT_ERROR is not None:
            raise ModuleNotFoundError(
                "Pillow is required for saving timestamped MuJoCo recordings."
            ) from PIL_IMPORT_ERROR
        if self.recording_output_dir is None:
            return None
        if not self.video_frames:
            print("[M1 NoCmd Deploy] No video frames captured. Skip video save.")
            return None

        output_path = self.recording_output_dir / f"{self.recording_timestamp}_mujoco_recording.gif"
        frames = [Image.fromarray(frame) for frame in self.video_frames]
        if len(self.video_frame_wall_times) >= 2:
            frame_duration_ms = [
                max(
                    1,
                    int(
                        round(
                            (self.video_frame_wall_times[i + 1] - self.video_frame_wall_times[i])
                            * 1000.0
                        )
                    ),
                )
                for i in range(len(self.video_frame_wall_times) - 1)
            ]
            frame_duration_ms.append(frame_duration_ms[-1])
        else:
            frame_duration_ms = max(1, int(round(1000.0 / self.video_fps)))
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration_ms,
            loop=0,
        )
        print(f"[M1 NoCmd Deploy] Saved recording video: {output_path}")
        return output_path

    def _start_recording_session(self):
        self.recording_active = True
        self._begin_recording_artifacts()
        self.recorded_dof_vel = []
        self.recorded_raw_tau = []
        self.recorded_clipped_tau = []
        print("[M1 NoCmd Deploy] Torque scatter recording started.")

    def _stop_recording_session(self):
        self.recording_active = False
        if not self.recorded_dof_vel:
            print("[M1 NoCmd Deploy] No recorded samples. Skip plotting.")
            return
        self._finalize_recording_video()
        print(
            "[M1 NoCmd Deploy] Torque scatter recording stopped. "
            f"Plotting {len(self.recorded_dof_vel)} samples."
        )
        self._launch_vel_torque_plot_process()

    def _record_current_sample(self):
        if not self.recording_active:
            return
        self.recorded_dof_vel.append(self.qvel_policy.copy())
        self.recorded_raw_tau.append(self.last_raw_tau.copy())
        self.recorded_clipped_tau.append(self.ctrl_policy.copy())

    def _launch_vel_torque_plot_process(self):
        if MATPLOTLIB_IMPORT_ERROR is not None:
            raise ModuleNotFoundError(
                "matplotlib is required for plotting torque debug figures."
            ) from MATPLOTLIB_IMPORT_ERROR

        ctx = mp.get_context("spawn")
        output_image_path = str(
            self.recording_output_dir / f"{self.recording_timestamp}_{self._get_plot_filename()}"
        )
        plot_process = ctx.Process(
            target=vel_torque_plot_worker,
            args=(
                list(self.train_dof_names),
                np.asarray(self.torque_limits, dtype=np.float32),
                np.asarray(self.reference_gear_ratios, dtype=np.float32),
                np.asarray(self.recorded_dof_vel, dtype=np.float32),
                np.asarray(self.recorded_raw_tau, dtype=np.float32),
                np.asarray(self.recorded_clipped_tau, dtype=np.float32),
                self.recording_timestamp,
                output_image_path,
            ),
            daemon=False,
        )
        plot_process.start()
        self.plot_processes = [process for process in self.plot_processes if process.is_alive()]
        self.plot_processes.append(plot_process)

    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            start = time.time()
            sim_step = 0

            while viewer.is_running() and time.time() - start < self.simulation_duration:
                step_start = time.time()
                if self.keyboard_state.pending_reset:
                    self.reset()
                if getattr(self.keyboard_state, "start_recording", False):
                    self.keyboard_state.start_recording = False
                    self._start_recording_session()
                if getattr(self.keyboard_state, "stop_recording", False):
                    self.keyboard_state.stop_recording = False
                    self._stop_recording_session()

                self._sync_command()
                self.get_robot_state()
                actuator_ctrl = self.compute_torques()
                self._record_current_sample()
                self.data.ctrl[:] = actuator_ctrl
                mujoco.mj_step(self.model, self.data)
                self._capture_video_frame_if_needed(viewer)

                sim_step += 1
                if sim_step % self.control_decimation == 0:
                    self.control_step += 1
                    if self.torque_monitor is not None:
                        self.torque_monitor.push_sample(self.data.time, self.last_raw_tau)
                    self.get_robot_state()
                    self.compute_observation()
                    self.run_policy()

                    if self.control_step % self.log_interval_steps == 0:
                        self.log_command_and_velocity()

                    if self.torque_monitor is not None:
                        self.torque_monitor.maybe_redraw()

                viewer.sync()
                remaining = self.model.opt.timestep - (time.time() - step_start)
                if remaining > 0:
                    time.sleep(remaining)

    def shutdown(self):
        self.joystick.stop()
        self.listener.stop()
        if self.torque_monitor is not None:
            self.torque_monitor.close()
        if self.video_renderer is not None:
            self.video_renderer.close()
            self.video_renderer = None
        for plot_process in self.plot_processes:
            if plot_process.is_alive():
                plot_process.join(timeout=0.1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deploy HIMLoco M1 NoCmd policy in MuJoCo (deploy-side strip: ONNX inputs are history_no_cmd + obs_curr)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to the deploy yaml.",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default=None,
        help="Optional ONNX override.",
    )
    parser.add_argument(
        "--enable-monitor",
        action="store_true",
        help="Enable real-time raw torque monitor window.",
    )
    parser.add_argument(
        "--history-seconds",
        type=float,
        default=10.0,
        help="Raw torque monitor history window in seconds.",
    )
    parser.add_argument(
        "--plot-refresh-interval",
        type=float,
        default=0.1,
        help="Raw torque monitor redraw interval in seconds.",
    )
    parser.add_argument(
        "--cmd-x",
        type=float,
        default=None,
        help="Optional fixed forward velocity command in m/s.",
    )
    parser.add_argument(
        "--cmd-y",
        type=float,
        default=None,
        help="Optional fixed lateral velocity command in m/s.",
    )
    parser.add_argument(
        "--cmd-yaw",
        type=float,
        default=None,
        help="Optional fixed yaw-rate command in rad/s.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    fixed_cmd = None
    if args.cmd_x is not None or args.cmd_y is not None or args.cmd_yaw is not None:
        fixed_cmd = np.array(
            [
                0.0 if args.cmd_x is None else args.cmd_x,
                0.0 if args.cmd_y is None else args.cmd_y,
                0.0 if args.cmd_yaw is None else args.cmd_yaw,
            ],
            dtype=np.float32,
        )

    deploy = DeployM1NoCmd(
        cfg,
        onnx_override=args.onnx,
        history_seconds=args.history_seconds,
        plot_refresh_interval=args.plot_refresh_interval,
        enable_monitor=args.enable_monitor,
        fixed_cmd=fixed_cmd,
    )
    try:
        deploy.run()
    finally:
        deploy.shutdown()


if __name__ == "__main__":
    main()
