import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyExporterHIMNoCmd(nn.Module):
    """Export wrapper for HIMActorCriticNoCmd.

    External deploy still feeds full obs_history (with commands).
    Commands are stripped only for the estimator encoder.
    """

    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.estimator = copy.deepcopy(actor_critic.estimator.encoder)
        self.num_one_step_obs = int(actor_critic.num_one_step_obs)
        self.history_size = int(actor_critic.history_size)
        self.history_cmd_start = int(actor_critic.estimator.history_cmd_start)
        self.history_cmd_end = int(actor_critic.estimator.history_cmd_end)

    def strip_history_commands(self, obs_history):
        history = obs_history.view(-1, self.history_size, self.num_one_step_obs)
        keep_prefix = history[..., :self.history_cmd_start]
        keep_suffix = history[..., self.history_cmd_end:]
        stripped = torch.cat((keep_prefix, keep_suffix), dim=-1)
        return stripped.reshape(obs_history.shape[0], -1)

    def forward(self, obs_history):
        obs_no_cmd = self.strip_history_commands(obs_history)
        parts = self.estimator(obs_no_cmd)
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2.0)
        obs_curr = obs_history[..., :self.num_one_step_obs]
        actor_in = torch.cat([obs_curr, vel, z], dim=-1)
        return self.actor(actor_in)

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
