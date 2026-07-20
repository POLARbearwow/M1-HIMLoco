# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import os
import sys
import yaml
from enum import Enum
import json

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RSL_ROOT = os.path.join(ROOT_DIR, "rsl_rl")
for path in (RSL_ROOT, ROOT_DIR):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import class_to_dict, get_args, task_registry
import torch


def _to_yaml_safe(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {_to_yaml_safe(key): _to_yaml_safe(val) for key, val in obj.items()}
    if isinstance(obj, list):
        return [_to_yaml_safe(item) for item in obj]
    if isinstance(obj, tuple):
        return [_to_yaml_safe(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return _to_yaml_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Enum):
        return obj.name
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return _to_yaml_safe(obj.item())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return _to_yaml_safe(vars(obj))
    try:
        yaml.safe_dump(obj)
        return obj
    except Exception:
        return str(obj)


def _write_structured_config(config_dir, stem, data):
    yaml_path = os.path.join(config_dir, f"{stem}.yaml")
    try:
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        return
    except Exception:
        if os.path.exists(yaml_path):
            os.remove(yaml_path)

    json_path = os.path.join(config_dir, f"{stem}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _save_run_configs(log_dir, env_cfg, train_cfg, args):
    if log_dir is None:
        return None

    config_dir = os.path.join(log_dir, "configs")
    os.makedirs(config_dir, exist_ok=True)

    env_cfg_dict = _to_yaml_safe(class_to_dict(env_cfg))
    train_cfg_dict = _to_yaml_safe(class_to_dict(train_cfg))
    args_dict = _to_yaml_safe(vars(args))

    _write_structured_config(config_dir, "env_cfg", env_cfg_dict)
    _write_structured_config(config_dir, "train_cfg", train_cfg_dict)
    _write_structured_config(config_dir, "args", args_dict)

    return {
        "env_cfg": env_cfg_dict,
        "train_cfg": train_cfg_dict,
        "args": args_dict,
    }


def train(args, headless=True):
    args.headless = headless
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    checkpoint_metadata = _save_run_configs(ppo_runner.log_dir, env_cfg, train_cfg, args)
    ppo_runner.set_checkpoint_metadata(checkpoint_metadata)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args, headless=True)
