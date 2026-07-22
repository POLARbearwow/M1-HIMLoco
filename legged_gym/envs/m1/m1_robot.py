import torch
from isaacgym.torch_utils import quat_rotate_inverse, torch_rand_float

from legged_gym.envs import LeggedRobot
from legged_gym.utils.helpers import class_to_dict

from .m1_config import M1RoughCfg


class M1HimRobot(LeggedRobot):
    cfg: M1RoughCfg

    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.stair_command_ranges = class_to_dict(self.cfg.commands.stair_ranges)

    def _init_buffers(self):
        super()._init_buffers()
        self.y_only_env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.yaw_only_env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        foot_pos_translated = self.feet_pos - self.root_states[:, :3].unsqueeze(1)
        foot_pos_in_body = torch.zeros_like(foot_pos_translated)
        for i in range(self.feet_indices.shape[0]):
            foot_pos_in_body[:, i, :] = quat_rotate_inverse(self.base_quat, foot_pos_translated[:, i, :])
        mean_y = torch.mean(foot_pos_in_body[:, :, 1], dim=0)
        self.feet_side_sign = torch.where(mean_y >= 0.0, torch.ones_like(mean_y), -torch.ones_like(mean_y))
        self._init_gait_reward_buffers(foot_pos_in_body)

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        self._resample_y_only_envs(env_ids)
        self._resample_yaw_only_envs(env_ids)
        super().reset_idx(env_ids)
        self.gait_air_time[env_ids] = 0.0
        self.gait_contact_time[env_ids] = 0.0

    def _init_gait_reward_buffers(self, foot_pos_in_body):
        foot_mean_pos = torch.mean(foot_pos_in_body, dim=0)
        foot_label_to_index = {}
        for foot_idx in range(foot_mean_pos.shape[0]):
            x_coord = foot_mean_pos[foot_idx, 0].item()
            y_coord = foot_mean_pos[foot_idx, 1].item()
            side = "L" if y_coord >= 0.0 else "R"
            axle = "F" if x_coord >= 0.0 else "H"
            foot_label_to_index[side + axle] = foot_idx

        gait_pair_labels = self.cfg.rewards.gait_synced_feet_pair_labels
        self.gait_synced_feet_pairs = []
        for pair_labels in gait_pair_labels:
            if len(pair_labels) != 2:
                raise ValueError("Each gait synced feet pair must contain exactly two foot labels.")
            try:
                self.gait_synced_feet_pairs.append(
                    (foot_label_to_index[pair_labels[0]], foot_label_to_index[pair_labels[1]])
                )
            except KeyError as exc:
                raise ValueError(
                    f"Unknown gait foot label '{exc.args[0]}'. Resolved labels: {sorted(foot_label_to_index.keys())}"
                ) from exc

        self.gait_air_time = torch.zeros_like(self.feet_air_time)
        self.gait_contact_time = torch.zeros_like(self.feet_air_time)

    def _get_stair_env_mask(self, env_ids):
        if len(env_ids) == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        if not hasattr(self, "terrain_type_ids"):
            return torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)

        terrain_type = self.terrain_type_ids[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        return (terrain_type == 2) | (terrain_type == 3)

    def _resample_y_only_envs(self, env_ids):
        self.y_only_env_mask[env_ids] = False
        ratio = float(self.cfg.commands.y_only_env_ratio)
        if ratio <= 0.0 or len(env_ids) == 0:
            return

        num_envs = int(round(len(env_ids) * ratio))
        if ratio > 0.0 and num_envs == 0:
            num_envs = 1
        num_envs = min(num_envs, len(env_ids))
        if num_envs <= 0:
            return

        perm = torch.randperm(len(env_ids), device=self.device)
        selected = env_ids[perm[:num_envs]]
        self.y_only_env_mask[selected] = True

    def _resample_yaw_only_envs(self, env_ids):
        self.yaw_only_env_mask[env_ids] = False
        available_env_ids = env_ids[~self.y_only_env_mask[env_ids]]
        ratio = float(self.cfg.commands.yaw_only_env_ratio)
        if ratio <= 0.0 or len(available_env_ids) == 0:
            return

        num_envs = int(round(len(env_ids) * ratio))
        if ratio > 0.0 and num_envs == 0:
            num_envs = 1
        num_envs = min(num_envs, len(available_env_ids))
        if num_envs <= 0:
            return

        perm = torch.randperm(len(available_env_ids), device=self.device)
        selected = available_env_ids[perm[:num_envs]]
        self.yaw_only_env_mask[selected] = True

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self.commands[self.y_only_env_mask, 0] = 0.0
        self.commands[self.y_only_env_mask, 2] = 0.0
        if self.cfg.commands.heading_command:
            self.commands[self.y_only_env_mask, 3] = 0.0
        self.commands[self.yaw_only_env_mask, :2] = 0.0
        self._update_gait_timers()

    def _update_gait_timers(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        self.gait_contact_time = torch.where(
            contact,
            self.gait_contact_time + self.dt,
            torch.zeros_like(self.gait_contact_time),
        )
        self.gait_air_time = torch.where(
            contact,
            torch.zeros_like(self.gait_air_time),
            self.gait_air_time + self.dt,
        )

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return

        super()._resample_commands(env_ids)

        stair_env_ids = env_ids[self._get_stair_env_mask(env_ids)]
        if len(stair_env_ids) > 0:
            stair_lin_vel_x = self.stair_command_ranges.get("lin_vel_x")
            stair_lin_vel_y = self.stair_command_ranges.get("lin_vel_y")
            stair_ang_vel_yaw = self.stair_command_ranges.get("ang_vel_yaw")
            stair_heading = self.stair_command_ranges.get("heading")

            if stair_lin_vel_x is not None:
                self.commands[stair_env_ids, 0] = torch_rand_float(
                    stair_lin_vel_x[0], stair_lin_vel_x[1], (len(stair_env_ids), 1), device=self.device
                ).squeeze(1)
            if stair_lin_vel_y is not None:
                self.commands[stair_env_ids, 1] = torch_rand_float(
                    stair_lin_vel_y[0], stair_lin_vel_y[1], (len(stair_env_ids), 1), device=self.device
                ).squeeze(1)
            if self.cfg.commands.heading_command:
                if stair_heading is not None:
                    self.commands[stair_env_ids, 3] = torch_rand_float(
                        stair_heading[0], stair_heading[1], (len(stair_env_ids), 1), device=self.device
                    ).squeeze(1)
            elif stair_ang_vel_yaw is not None:
                self.commands[stair_env_ids, 2] = torch_rand_float(
                    stair_ang_vel_yaw[0], stair_ang_vel_yaw[1], (len(stair_env_ids), 1), device=self.device
                ).squeeze(1)

        y_only_env_ids = env_ids[self.y_only_env_mask[env_ids]]
        if len(y_only_env_ids) > 0:
            self.commands[y_only_env_ids, 0] = 0.0
            self.commands[y_only_env_ids, 2] = 0.0
            if self.cfg.commands.heading_command:
                self.commands[y_only_env_ids, 3] = 0.0

        yaw_only_env_ids = env_ids[self.yaw_only_env_mask[env_ids]]
        if len(yaw_only_env_ids) > 0:
            self.commands[yaw_only_env_ids, :2] = 0.0
            if self.cfg.commands.heading_command:
                self.commands[yaw_only_env_ids, 3] = torch_rand_float(
                    self.command_ranges["heading"][0],
                    self.command_ranges["heading"][1],
                    (len(yaw_only_env_ids), 1),
                    device=self.device,
                ).squeeze(1)

    def _reward_feet_distance_y_exp(self):
        foot_pos_translated = self.feet_pos - self.root_states[:, :3].unsqueeze(1)
        foot_pos_in_body = torch.zeros_like(foot_pos_translated)
        for i in range(self.feet_indices.shape[0]):
            foot_pos_in_body[:, i, :] = quat_rotate_inverse(self.base_quat, foot_pos_translated[:, i, :])

        target_y = 0.5 * self.cfg.rewards.feet_distance_y_target * self.feet_side_sign.unsqueeze(0)
        stance_diff = torch.square(target_y - foot_pos_in_body[:, :, 1])
        reward = torch.exp(-torch.sum(stance_diff, dim=1) / (self.cfg.rewards.feet_distance_y_sigma ** 2))
        reward *= torch.clamp(-self.projected_gravity[:, 2], 0.0, 0.7) / 0.7
        return reward

    def _get_gait_reward_gate(self):
        lateral_cmd = torch.abs(self.commands[:, 1])
        yaw_cmd = torch.abs(self.commands[:, 2])
        return (
            (lateral_cmd > float(self.cfg.rewards.gait_lateral_cmd_threshold))
            | (yaw_cmd > float(self.cfg.rewards.gait_yaw_cmd_threshold))
        ).float()

    def _get_wheel_first_air_mask(self):
        foot_contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        first_air = (~foot_contact) & self.last_contacts
        return first_air.float()

    def _reward_feet_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        first_contact = (self.feet_air_time > 0.0) * contact
        self.feet_air_time += self.dt

        reward = torch.sum(
            (self.feet_air_time - self.cfg.rewards.feet_air_time_target) * first_contact,
            dim=1,
        )
        reward *= self._get_gait_reward_gate()
        # reward *= torch.clamp(-self.projected_gravity[:, 2], 0.0, 0.7) / 0.7

        self.last_contacts = contact
        self.feet_air_time *= ~contact
        return reward

    def _reward_feet_height_body(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        swing_mask = (~contact).float()
        feet_height = self._get_feet_heights()

        height_error = torch.square(feet_height - self.cfg.rewards.feet_height_body_target)
        reward = torch.exp(-height_error / (self.cfg.rewards.feet_height_body_sigma ** 2))
        reward = torch.sum(reward * swing_mask, dim=1)
        reward *= torch.clamp(-self.projected_gravity[:, 2], 0.0, 0.7) / 0.7
        return reward

    def _reward_wheel_vel_penalty(self):
        cmd = torch.linalg.norm(self.commands[:, :3], dim=1)
        body_vel = torch.linalg.norm(self.base_lin_vel[:, :2], dim=1)
        joint_vel = torch.abs(self.dof_vel[:, self.wheel_indices])
        in_air = self._get_wheel_first_air_mask()
        running_penalty = torch.sum(in_air * joint_vel, dim=1)
        standing_penalty = torch.sum(joint_vel, dim=1)
        return torch.where(
            torch.logical_or(
                cmd > float(self.cfg.rewards.wheel_vel_penalty_command_threshold),
                body_vel > float(self.cfg.rewards.wheel_vel_penalty_velocity_threshold),
            ),
            running_penalty,
            standing_penalty,
        )

    def _reward_wheel_vel_air_side(self):
        """Penalize wheels when airborne or side-loading.

        Conditions (per foot/wheel, OR):
          1) off ground: |Fz| < contact threshold
          2) side load: ||F_xy|| > ratio * |Fz|

        Strengthening vs plain |omega|:
          - active bias when condition holds (fires even if omega~0)
          - quadratic wheel speed
          - continuous side-force excess weight
        """
        foot_forces = self.contact_forces[:, self.feet_indices, :]
        f_xy = torch.norm(foot_forces[:, :, :2], dim=2)
        f_z = torch.abs(foot_forces[:, :, 2])
        contact_thresh = float(self.cfg.rewards.wheel_air_contact_force_threshold)
        side_ratio = float(self.cfg.rewards.wheel_side_force_ratio)
        vel_sq_coef = float(self.cfg.rewards.wheel_vel_air_side_vel_square_coef)
        active_bias = float(self.cfg.rewards.wheel_vel_air_side_active_bias)
        force_coef = float(self.cfg.rewards.wheel_vel_air_side_force_coef)

        in_air = f_z < contact_thresh
        side_excess = (f_xy - side_ratio * f_z).clip(min=0.0)
        side_load = side_excess > 0.0
        mask = (in_air | side_load).float()

        joint_vel = torch.abs(self.dof_vel[:, self.wheel_indices])
        vel_term = joint_vel + vel_sq_coef * torch.square(joint_vel)
        side_weight = 1.0 + force_coef * side_excess
        return torch.sum(mask * side_weight * (vel_term + active_bias), dim=1)

    def _reward_gait(self):
        sync_reward_0 = self._sync_reward_func(*self.gait_synced_feet_pairs[0])
        sync_reward_1 = self._sync_reward_func(*self.gait_synced_feet_pairs[1])
        sync_reward = sync_reward_0 * sync_reward_1

        async_reward_0 = self._async_reward_func(
            self.gait_synced_feet_pairs[0][0], self.gait_synced_feet_pairs[1][0]
        )
        async_reward_1 = self._async_reward_func(
            self.gait_synced_feet_pairs[0][1], self.gait_synced_feet_pairs[1][1]
        )
        async_reward_2 = self._async_reward_func(
            self.gait_synced_feet_pairs[0][0], self.gait_synced_feet_pairs[1][1]
        )
        async_reward_3 = self._async_reward_func(
            self.gait_synced_feet_pairs[1][0], self.gait_synced_feet_pairs[0][1]
        )
        reward = sync_reward * async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        reward *= self._get_gait_reward_gate()
        reward *= torch.clamp(-self.projected_gravity[:, 2], 0.0, 0.7) / 0.7
        return reward

    def _sync_reward_func(self, foot_0: int, foot_1: int):
        se_air = torch.clip(
            torch.square(self.gait_air_time[:, foot_0] - self.gait_air_time[:, foot_1]),
            max=float(self.cfg.rewards.gait_max_err) ** 2,
        )
        se_contact = torch.clip(
            torch.square(self.gait_contact_time[:, foot_0] - self.gait_contact_time[:, foot_1]),
            max=float(self.cfg.rewards.gait_max_err) ** 2,
        )
        return torch.exp(-(se_air + se_contact) / float(self.cfg.rewards.gait_std))

    def _async_reward_func(self, foot_0: int, foot_1: int):
        se_act_0 = torch.clip(
            torch.square(self.gait_air_time[:, foot_0] - self.gait_contact_time[:, foot_1]),
            max=float(self.cfg.rewards.gait_max_err) ** 2,
        )
        se_act_1 = torch.clip(
            torch.square(self.gait_contact_time[:, foot_0] - self.gait_air_time[:, foot_1]),
            max=float(self.cfg.rewards.gait_max_err) ** 2,
        )
        return torch.exp(-(se_act_0 + se_act_1) / float(self.cfg.rewards.gait_std))
