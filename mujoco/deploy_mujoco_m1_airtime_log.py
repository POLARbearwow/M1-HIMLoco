"""Deploy M1 policy and log feet air time (swing duration on first contact).

Usage:
  python mujoco/deploy_mujoco_m1_airtime_log.py
  python mujoco/deploy_mujoco_m1_airtime_log.py --onnx /path/to/policy.onnx
  python mujoco/deploy_mujoco_m1_airtime_log.py --config mujoco/configs/m1.yaml --csv airtime.csv

Contact is estimated from MuJoCo body external force on FOOT links
(cfrc_ext force magnitude > threshold), analogous to training's Fz > 1.
"""

import argparse
import csv
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from deploy_mujoco_m1 import DeployM1, DEFAULT_CONFIG, load_config


FOOT_BODY_NAMES = [
    "LF_FOOT_link",
    "LH_FOOT_link",
    "RF_FOOT_link",
    "RH_FOOT_link",
]


class DeployM1AirTimeLog(DeployM1):
    def __init__(
        self,
        cfg,
        onnx_override=None,
        contact_force_threshold=1.0,
        airtime_log_interval=1,
        csv_path=None,
    ):
        super().__init__(cfg, onnx_override=onnx_override)
        self.contact_force_threshold = float(contact_force_threshold)
        self.airtime_log_interval = max(int(airtime_log_interval), 1)
        self.csv_path = csv_path
        self.control_dt = self.simulation_dt * self.control_decimation

        self.foot_body_ids = self._resolve_foot_body_ids()
        self.num_feet = len(self.foot_body_ids)
        self.feet_air_time = np.zeros(self.num_feet, dtype=np.float32)
        self.last_contacts = np.zeros(self.num_feet, dtype=bool)
        self.foot_contact_force = np.zeros(self.num_feet, dtype=np.float32)

        self.touchdown_air_times = []  # list of (step, foot_idx, air_time)
        self.csv_file = None
        self.csv_writer = None
        if self.csv_path is not None:
            Path(self.csv_path).parent.mkdir(parents=True, exist_ok=True)
            self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(
                ["control_step", "time_s", "foot", "air_time_s", "cmd_vx", "cmd_vy", "cmd_yaw"]
            )

        print(
            f"[M1 AirTime] Foot bodies: "
            f"{[mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(i)) for i in self.foot_body_ids]}"
        )
        print(
            f"[M1 AirTime] contact_force_threshold={self.contact_force_threshold}, "
            f"control_dt={self.control_dt:.4f}s, csv={self.csv_path}"
        )

    def _resolve_foot_body_ids(self):
        ids = []
        for name in FOOT_BODY_NAMES:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id < 0:
                raise RuntimeError(f"Foot body '{name}' not found in MuJoCo model.")
            ids.append(body_id)
        return np.array(ids, dtype=np.int32)

    def _get_foot_contacts(self):
        # cfrc_ext: [torque(3), force(3)] external wrench on body, world frame.
        forces = np.zeros(self.num_feet, dtype=np.float32)
        for i, body_id in enumerate(self.foot_body_ids):
            f = self.data.cfrc_ext[body_id, 3:6]
            forces[i] = float(np.linalg.norm(f))
        self.foot_contact_force[:] = forces
        return forces > self.contact_force_threshold

    def update_feet_air_time(self):
        contact = self._get_foot_contacts()
        first_contact = (self.feet_air_time > 0.0) & contact
        self.feet_air_time += self.control_dt

        for foot_idx in np.where(first_contact)[0]:
            air_time = float(self.feet_air_time[foot_idx])
            self.touchdown_air_times.append(
                (self.control_step, int(foot_idx), air_time)
            )
            foot_name = FOOT_BODY_NAMES[foot_idx]
            print(
                f"[M1 AirTime] step={self.control_step:6d} "
                f"t={self.control_step * self.control_dt:7.2f}s "
                f"{foot_name}: air_time={air_time:.3f}s "
                f"force={self.foot_contact_force[foot_idx]:.2f} "
                f"cmd=({self.cmd[0]:+.2f},{self.cmd[1]:+.2f},{self.cmd[2]:+.2f})"
            )
            if self.csv_writer is not None:
                self.csv_writer.writerow(
                    [
                        self.control_step,
                        f"{self.control_step * self.control_dt:.4f}",
                        foot_name,
                        f"{air_time:.6f}",
                        f"{self.cmd[0]:.4f}",
                        f"{self.cmd[1]:.4f}",
                        f"{self.cmd[2]:.4f}",
                    ]
                )
                self.csv_file.flush()

        self.last_contacts = contact
        self.feet_air_time *= ~contact

    def log_command_and_velocity(self):
        super().log_command_and_velocity()
        if len(self.touchdown_air_times) == 0:
            print("[M1 AirTime] no touchdowns yet")
            return

        recent = self.touchdown_air_times[-20:]
        vals = np.array([x[2] for x in recent], dtype=np.float32)
        all_vals = np.array([x[2] for x in self.touchdown_air_times], dtype=np.float32)
        print(
            "[M1 AirTime] "
            f"n={len(all_vals)} "
            f"mean_all={all_vals.mean():.3f}s "
            f"mean_last20={vals.mean():.3f}s "
            f"min={all_vals.min():.3f}s max={all_vals.max():.3f}s | "
            f"current_air={np.array2string(self.feet_air_time, precision=3)} "
            f"contact={self.last_contacts.astype(int).tolist()} "
            f"force={np.array2string(self.foot_contact_force, precision=1)}"
        )

    def reset(self, full_reset=False):
        super().reset(full_reset=full_reset)
        # Parent DeployM1.__init__ calls reset before subclass buffers exist.
        if not hasattr(self, "feet_air_time"):
            return
        self.feet_air_time[:] = 0.0
        self.last_contacts[:] = False
        self.foot_contact_force[:] = 0.0
        if full_reset and hasattr(self, "touchdown_air_times"):
            self.touchdown_air_times.clear()

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
                    self.update_feet_air_time()

                    if self.control_step % self.log_interval_steps == 0:
                        self.log_command_and_velocity()

                viewer.sync()
                remaining = self.model.opt.timestep - (time.time() - step_start)
                if remaining > 0:
                    time.sleep(remaining)

    def shutdown(self):
        if len(self.touchdown_air_times) > 0:
            vals = np.array([x[2] for x in self.touchdown_air_times], dtype=np.float32)
            print(
                f"[M1 AirTime] FINAL n={len(vals)} "
                f"mean={vals.mean():.3f}s std={vals.std():.3f}s "
                f"min={vals.min():.3f}s max={vals.max():.3f}s"
            )
        else:
            print("[M1 AirTime] FINAL: no touchdowns recorded")
        if self.csv_file is not None:
            self.csv_file.close()
            print(f"[M1 AirTime] wrote CSV: {self.csv_path}")
        super().shutdown()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deploy M1 policy and log feet air time on touchdown."
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--onnx", type=str, default=None)
    parser.add_argument(
        "--contact_force_threshold",
        type=float,
        default=1.0,
        help="||F_ext|| on FOOT body above this counts as contact.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional CSV path for touchdown air times.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    deploy = DeployM1AirTimeLog(
        cfg,
        onnx_override=args.onnx,
        contact_force_threshold=args.contact_force_threshold,
        csv_path=args.csv,
    )
    try:
        deploy.run()
    finally:
        deploy.shutdown()


if __name__ == "__main__":
    main()
