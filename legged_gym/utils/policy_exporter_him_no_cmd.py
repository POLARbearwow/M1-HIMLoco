import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyExporterHIMNoCmd(nn.Module):
    """Export wrapper for HIMActorCriticNoCmd (deploy-side strip).

    ONNX / JIT inputs:
      - obs_history_no_cmd: [B, history_size * (num_one_step_obs - 3)]  e.g. 324
      - obs_curr:           [B, num_one_step_obs]                        e.g. 57 (includes command)
    """

    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.estimator = copy.deepcopy(actor_critic.estimator.encoder)
        self.num_one_step_obs = int(actor_critic.num_one_step_obs)
        self.history_size = int(actor_critic.history_size)
        self.num_one_step_obs_no_cmd = int(actor_critic.estimator.num_one_step_obs_no_cmd)
        self.obs_history_no_cmd_dim = self.history_size * self.num_one_step_obs_no_cmd

    def forward(self, obs_history_no_cmd, obs_curr):
        parts = self.estimator(obs_history_no_cmd)
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2.0)
        actor_in = torch.cat([obs_curr, vel, z], dim=-1)
        return self.actor(actor_in)

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
