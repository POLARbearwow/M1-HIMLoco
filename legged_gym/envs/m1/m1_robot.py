import torch
from isaacgym.torch_utils import quat_rotate_inverse, torch_rand_float

from legged_gym.envs import LeggedRobot
from legged_gym.utils.helpers import class_to_dict

from .m1_config import M1RoughCfg


class M1HimRobot(LeggedRobot):
    cfg: M1RoughCfg

    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        if not self.cfg.rewards.enable_dreamwaq_joint_penalties:
            for reward_name in (
                "smoothness_2",
                "torque_limits",
                "raw_torques",
                "joint_power",
                "wheel_dof_vel_limits",
                "knee_dof_vel_limits",
            ):
                self.reward_scales[reward_name] = 0.0
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
        # m1-dreamwaq torque/velocity transplant buffers
        self.raw_torques = torch.zeros_like(self.torques)
        knee_indices = [i for i, name in enumerate(self.dof_names) if "KNEE" in name]
        self.knee_indices = torch.tensor(knee_indices, dtype=torch.long, device=self.device)

    # m1-dreamwaq torque/velocity limit transplant
    def _process_dof_props(self, props, env_id):
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.rated_torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()

                rated_torque_limit_ratio = self.cfg.control.rated_torque_limit_ratio
                dof_name = self.dof_names[i]
                if "HAA" in dof_name:
                    rated_torque_limit_ratio = self.cfg.control.rated_torque_limit_ratio_haa
                elif "HFE" in dof_name:
                    rated_torque_limit_ratio = self.cfg.control.rated_torque_limit_ratio_hfe
                self.rated_torque_limits[i] = rated_torque_limit_ratio * self.torque_limits[i]

                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _compute_torques(self, actions):
        dof_err = self.default_dof_pos - self.dof_pos
        dof_err[:, self.wheel_indices] = 0
        actions_scaled = actions * self.cfg.control.action_scale
        actions_scaled[:, self.wheel_indices] = 0
        vel_ref = torch.zeros_like(actions_scaled)
        vel_tmp = actions * self.cfg.control.vel_scale
        vel_ref[:, self.wheel_indices] = vel_tmp[:, self.wheel_indices]

        control_type = self.cfg.control.control_type
        if control_type == "P":
            raw_torques = (
                self.p_gains * self.Kp_factors * (actions_scaled + dof_err)
                + self.d_gains * self.Kd_factors * (vel_ref - self.dof_vel)
            )
        elif control_type == "V":
            raw_torques = (
                self.p_gains * (actions_scaled - self.dof_vel)
                - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
            )
        elif control_type == "T":
            raw_torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        raw_torques *= self.motor_strength_factors
        self.raw_torques = raw_torques
        return torch.clip(raw_torques, -self.rated_torque_limits, self.rated_torque_limits)
    # end m1-dreamwaq torque/velocity limit transplant

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

    def _get_stairs_up_env_mask(self):
        """True for envs currently on stairs-up terrain (type id 2)."""
        if not hasattr(self, "terrain_type_ids"):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        terrain_type = self.terrain_type_ids[self.terrain_levels, self.terrain_types]
        return terrain_type == 2

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

    def _update_terrain_curriculum(self, env_ids):
        """Use command-scaled progression thresholds on stair terrains."""
        if not self.init_done:
            return

        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        stair_mask = self._get_stair_env_mask(env_ids)

        move_up = distance > (self.terrain.env_length / 2)
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5) & ~move_up

        if torch.any(stair_mask):
            commanded_distance = torch.norm(self.commands[env_ids, :2], dim=1) * self.max_episode_length_s
            stair_move_up = distance > commanded_distance * 0.5
            stair_move_down = (distance < commanded_distance * 0.5) & ~stair_move_up
            move_up = torch.where(stair_mask, stair_move_up, move_up)
            move_down = torch.where(stair_mask, stair_move_down, move_down)

        self.terrain_levels[env_ids] += move_up.to(torch.long) - move_down.to(torch.long)
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0),
        )
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

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

    def _reward_roll_stability(self):
        # Further suppress trunk roll using the body-frame lateral gravity component.
        return torch.square(self.projected_gravity[:, 1])

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
        """Swing-foot clearance reward for stairs-up only (body-frame / trunk-relative).

        Foot height is measured in the base/trunk frame so stair-edge height-field
        jumps do not create discontinuous targets mid-swing.

        clearance ≈ foot_z_body + base_height_target
          - standing near default height ≈ 0
          - lifting the foot toward trunk increases clearance
        Reward peaks at feet_height_body_target (default 0.2 m).

        Gated by stairs_up only, forward cmd, and foot xy speed.
        (No upright/tilt gate.)
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        swing_mask = (~contact).float()

        # World foot pos/vel relative to trunk, then rotate into body frame.
        foot_pos_translated = self.feet_pos - self.root_states[:, :3].unsqueeze(1)
        foot_vel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        foot_pos_body = torch.zeros_like(foot_pos_translated)
        foot_vel_body = torch.zeros_like(foot_vel_translated)
        for i in range(self.feet_indices.shape[0]):
            foot_pos_body[:, i, :] = quat_rotate_inverse(self.base_quat, foot_pos_translated[:, i, :])
            foot_vel_body[:, i, :] = quat_rotate_inverse(self.base_quat, foot_vel_translated[:, i, :])

        # Positive when the foot is raised toward/above the nominal base height.
        base_h = float(self.cfg.rewards.base_height_target)
        feet_clearance = foot_pos_body[:, :, 2] + base_h

        target = float(self.cfg.rewards.feet_height_body_target)
        sigma = float(self.cfg.rewards.feet_height_body_sigma)
        tanh_mult = float(self.cfg.rewards.feet_height_tanh_mult)
        cmd_thr = float(self.cfg.rewards.feet_height_cmd_threshold)

        height_error = torch.square(feet_clearance - target)
        clearance_reward = torch.exp(-height_error / (sigma ** 2))
        foot_xy_vel = torch.norm(foot_vel_body[:, :, :2], dim=2)
        vel_gate = torch.tanh(tanh_mult * foot_xy_vel)
        reward = torch.sum(clearance_reward * vel_gate * swing_mask, dim=1)

        stairs_up_gate = self._get_stairs_up_env_mask().float()
        forward_gate = (self.commands[:, 0] > cmd_thr).float()
        reward *= stairs_up_gate * forward_gate
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

    # m1-dreamwaq torque/velocity reward transplant
    def _reward_joint_power(self):
        return torch.sum(torch.abs(self.dof_vel) * torch.abs(self.torques), dim=1)

    def _reward_raw_torques(self):
        return torch.sum((torch.abs(self.raw_torques) - self.rated_torque_limits).clip(min=0.0), dim=1)

    def _reward_torque_limits(self):
        return torch.sum(
            (torch.abs(self.torques) - self.rated_torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.0),
            dim=1,
        )

    def _reward_dof_vel(self):
        vel = self.dof_vel.clone()
        vel[:, self.wheel_indices] = 0
        return torch.sum(torch.square(vel), dim=1)

    def _reward_wheel_dof_vel_limits(self):
        wheel_vel = torch.abs(self.dof_vel[:, self.wheel_indices])
        wheel_limits = self.dof_vel_limits[self.wheel_indices] * self.cfg.rewards.wheel_soft_dof_vel_limit
        return torch.sum((wheel_vel - wheel_limits).clip(min=0.0, max=1.0), dim=1)

    def _reward_knee_dof_vel_limits(self):
        knee_vel = torch.abs(self.dof_vel[:, self.knee_indices])
        knee_limits = self.dof_vel_limits[self.knee_indices] * self.cfg.rewards.knee_soft_dof_vel_limit
        return torch.sum((knee_vel - knee_limits).clip(min=0.0, max=1.0), dim=1)
    # end m1-dreamwaq torque/velocity reward transplant

    def _reward_smoothness_2(self):
        """Second-order action smoothness (Lab smoothness_2).

        Penalize ||a_t - 2 a_{t-1} + a_{t-2}||^2, i.e. change of action rate.
        First-order term is already covered by base _reward_action_rate.
        Zero history after reset is ignored (same idea as Lab).
        """
        diff = torch.square(self.actions - 2.0 * self.last_actions + self.last_last_actions)
        valid = (torch.any(self.last_actions != 0.0, dim=1) & torch.any(self.last_last_actions != 0.0, dim=1)).float()
        return torch.sum(diff, dim=1) * valid

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
