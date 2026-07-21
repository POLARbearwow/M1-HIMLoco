import argparse
import os
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import yaml
from pynput import keyboard

from joystick_interface import JoystickInterface


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
    def __init__(self, cfg, onnx_override=None):
        self.cfg = cfg
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
        self.expected_obs_history_dim = int(cfg["obs_history_dim"])
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
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name
        self.input_dim = self._infer_last_dim(self.ort_session.get_inputs()[0].shape)
        self.output_dim = self._infer_last_dim(self.ort_session.get_outputs()[0].shape)

        self.obs_history_dim = self.expected_obs_history_dim
        if self.input_dim != self.expected_obs_history_dim:
            print(
                f"[Warning] ONNX input dim is {self.input_dim}, expected "
                f"{self.expected_obs_history_dim}. The deploy script will adapt by padding or truncating."
            )
            self.obs_history_dim = max(self.input_dim, self.expected_obs_history_dim)

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
        self.obs_history = np.zeros(self.obs_history_dim, dtype=np.float32)
        self.ctrl_policy = np.zeros(self.num_actions, dtype=np.float32)
        self.last_warn_step = {}
        self.control_step = 0
        self.missing_sensor_warnings = set()

        self._setup_mappings()
        self._print_startup_summary()
        self.reset(full_reset=True)

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
        print(f"[M1 NoCmd Deploy] ONNX input dim : {self.input_dim}")
        print(f"[M1 NoCmd Deploy] ONNX output dim: {self.output_dim}")
        print(f"[M1 NoCmd Deploy] Train DOF order : {self.train_dof_names}")
        print(f"[M1 NoCmd Deploy] MuJoCo qpos order: {self.mujoco_joint_names}")
        print(f"[M1 NoCmd Deploy] MuJoCo actuator order: {self.actuator_joint_names}")
        print(f"[M1 NoCmd Deploy] policy_to_qpos    : {self.policy_to_qpos}")
        print(f"[M1 NoCmd Deploy] policy_to_actuator: {self.policy_to_actuator}")
        print("[M1 NoCmd Deploy] Deploy obs (full history for ONNX): ang_vel(3)+gravity(3)+cmd(3)+dof_err(16)+dof_vel(16)+last_action(16)=57; estimator strips cmd inside network")
        print(f"[M1 NoCmd Deploy] Obs history dim: {self.expected_obs_history_dim}")
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
        self.obs_history[:] = 0.0
        self.keyboard_state.pending_reset = False
        self.keyboard_state.clear_command = False
        self.last_warn_step.clear()
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

    def compute_observation(self):
        dof_err = self.qpos_policy - self.default_angles
        dof_err[self.wheel_policy_indices] = 0.0

        self.obs[0:3] = self.base_ang_vel * self.ang_vel_scale
        self.obs[3:6] = self.projected_gravity
        self.obs[6:9] = self.cmd * self.cmd_scale
        self.obs[9:25] = dof_err * self.dof_pos_scale
        self.obs[25:41] = self.qvel_policy * self.dof_vel_scale
        self.obs[41:57] = self.action_policy

        self.obs_history[self.num_obs:] = self.obs_history[:-self.num_obs].copy()
        self.obs_history[: self.num_obs] = self.obs

    def _prepare_policy_input(self):
        if self.input_dim == self.expected_obs_history_dim:
            return self.obs_history.reshape(1, -1)

        if self.input_dim > self.expected_obs_history_dim:
            policy_input = np.zeros((1, self.input_dim), dtype=np.float32)
            policy_input[0, : self.expected_obs_history_dim] = self.obs_history[
                : self.expected_obs_history_dim
            ]
            return policy_input

        return self.obs_history[: self.input_dim].reshape(1, -1)

    def run_policy(self):
        policy_input = self._prepare_policy_input()
        action = self.ort_session.run(
            [self.output_name], {self.input_name: policy_input.astype(np.float32)}
        )[0][0]

        if action.shape[0] != self.num_actions:
            fixed_action = np.zeros(self.num_actions, dtype=np.float32)
            copy_dim = min(action.shape[0], self.num_actions)
            fixed_action[:copy_dim] = action[:copy_dim]
            action = fixed_action

        self.action_policy = action.astype(np.float32)

    def compute_torques(self):
        dof_err = self.default_angles - self.qpos_policy
        dof_err[self.wheel_policy_indices] = 0.0

        actions_scaled = self.action_policy * self.action_scale
        actions_scaled[self.wheel_policy_indices] = 0.0

        vel_ref = np.zeros_like(self.action_policy)
        vel_ref[self.wheel_policy_indices] = (
            self.action_policy[self.wheel_policy_indices] * self.vel_scale
        )

        raw_tau = self.kps * (actions_scaled + dof_err) + self.kds * (
            vel_ref - self.qvel_policy
        )
        self._warn_if_torque_exceeds_limit(raw_tau)
        clipped_tau = np.clip(raw_tau, -self.torque_limits, self.torque_limits)
        self.ctrl_policy = clipped_tau

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

    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            start = time.time()
            sim_step = 0

            while viewer.is_running() and time.time() - start < self.simulation_duration:
                step_start = time.time()
                if self.keyboard_state.pending_reset:
                    self.reset()

                self._sync_command()
                self.get_robot_state()
                actuator_ctrl = self.compute_torques()
                self.data.ctrl[:] = actuator_ctrl
                mujoco.mj_step(self.model, self.data)

                sim_step += 1
                if sim_step % self.control_decimation == 0:
                    self.control_step += 1
                    self.get_robot_state()
                    self.compute_observation()
                    self.run_policy()

                    if self.control_step % self.log_interval_steps == 0:
                        self.log_command_and_velocity()

                viewer.sync()
                remaining = self.model.opt.timestep - (time.time() - step_start)
                if remaining > 0:
                    time.sleep(remaining)

    def shutdown(self):
        self.joystick.stop()
        self.listener.stop()


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy HIMLoco M1 NoCmd policy in MuJoCo (estimator strips history commands internally).")
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
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    deploy = DeployM1NoCmd(cfg, onnx_override=args.onnx)
    try:
        deploy.run()
    finally:
        deploy.shutdown()


if __name__ == "__main__":
    main()
