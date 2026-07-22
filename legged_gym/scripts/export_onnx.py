import argparse
import ast
import copy
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RSL_ROOT = os.path.join(ROOT_DIR, "rsl_rl")
for path in (RSL_ROOT, ROOT_DIR):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

from rsl_rl.modules import HIMActorCritic, HIMActorCriticNoCmd


class PolicyExporterHIM(nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.estimator = copy.deepcopy(actor_critic.estimator.encoder)
        self.num_one_step_obs = int(actor_critic.num_one_step_obs)

    def forward(self, obs_history):
        parts = self.estimator(obs_history)
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2.0)
        obs_curr = obs_history[..., :self.num_one_step_obs]
        actor_in = torch.cat([obs_curr, vel, z], dim=-1)
        return self.actor(actor_in)


class PolicyExporterHIMNoCmd(nn.Module):
    """Deploy-side strip: ONNX expects pre-stripped history + full current obs.

    Inputs:
      - obs_history_no_cmd: [B, history_size * (num_one_step_obs - 3)]  e.g. 324
      - obs_curr:           [B, num_one_step_obs]                        e.g. 57
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


def get_args():
    parser = argparse.ArgumentParser(description="Export HIM M1 checkpoint to ONNX without Isaac Gym")
    parser.add_argument("--task", type=str, default="m1_him", help="Task name: m1_him | m1_him_no_cmd")
    parser.add_argument("--experiment_name", type=str, help="Experiment name. Overrides config if provided.")
    parser.add_argument("--run_name", type=str, help="Run name. Overrides config if provided.")
    parser.add_argument("--load_run", type=str, help="Run directory name to load. If omitted, uses config or latest run.")
    parser.add_argument("--checkpoint", type=int, help="Checkpoint number to load. If omitted, uses config or latest checkpoint.")
    parser.add_argument("--rl_device", type=str, default="cpu", help="Device used to load checkpoint before exporting.")
    parser.add_argument("--seed", type=int, help="Random seed. Overrides config if provided.")
    parser.add_argument("--max_iterations", type=int, help="Maximum number of training iterations. Overrides config if provided.")
    parser.add_argument("--output_dir", type=str, help="Directory to save ONNX. Default: logs/<experiment>/exported/policies")
    parser.add_argument("--output_name", type=str, default="policy.onnx", help="ONNX filename.")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version.")
    return parser.parse_args()


TASK_CONFIGS = {
    "m1_him": {
        "config_path": os.path.join(ROOT_DIR, "legged_gym", "envs", "m1", "m1_config.py"),
        "env_class": "M1RoughCfg",
        "ppo_class": "M1RoughCfgPPO",
        "policy_class": HIMActorCritic,
        "exporter_class": PolicyExporterHIM,
    },
    "m1_him_no_cmd": {
        "config_path": os.path.join(ROOT_DIR, "legged_gym", "envs", "m1", "m1_config.py"),
        "env_class": "M1RoughCfg",
        "ppo_class": "M1RoughCfgPPONoCmd",
        "policy_class": HIMActorCriticNoCmd,
        "exporter_class": PolicyExporterHIMNoCmd,
    },
}


def eval_expr(node, scope):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in scope:
            raise KeyError(f"Unknown name '{node.id}' in config expression")
        return scope[node.id]
    if isinstance(node, ast.List):
        return [eval_expr(elt, scope) for elt in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(eval_expr(elt, scope) for elt in node.elts)
    if isinstance(node, ast.Dict):
        return {
            eval_expr(key, scope): eval_expr(value, scope)
            for key, value in zip(node.keys, node.values)
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -eval_expr(node.operand, scope)
    if isinstance(node, ast.BinOp):
        left = eval_expr(node.left, scope)
        right = eval_expr(node.right, scope)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    raise ValueError(f"Unsupported config expression: {ast.dump(node)}")


def parse_nested_class(parent_class, nested_class_name):
    for node in parent_class.body:
        if isinstance(node, ast.ClassDef) and node.name == nested_class_name:
            values = {}
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                    values[stmt.targets[0].id] = eval_expr(stmt.value, values)
            return values
    raise ValueError(f"Nested class '{nested_class_name}' not found in '{parent_class.name}'")


def _find_class(tree, class_name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None


def parse_task_config(task_name):
    if task_name not in TASK_CONFIGS:
        supported = ", ".join(TASK_CONFIGS.keys())
        raise ValueError(f"Unsupported task '{task_name}'. Supported tasks: {supported}")

    spec = TASK_CONFIGS[task_name]
    with open(spec["config_path"], "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=spec["config_path"])

    env_class = _find_class(tree, spec["env_class"])
    ppo_class = _find_class(tree, spec["ppo_class"])
    if env_class is None or ppo_class is None:
        raise ValueError(f"Failed to find config classes for task '{task_name}'")

    # Resolve nested classes with inheritance (e.g. M1RoughCfgPPONoCmd.runner from parent).
    base_cfg_path = os.path.join(ROOT_DIR, "legged_gym", "envs", "base", "legged_robot_config.py")
    with open(base_cfg_path, "r", encoding="utf-8") as f:
        base_tree = ast.parse(f.read(), filename=base_cfg_path)

    base_env = _find_class(base_tree, "LeggedRobotCfg")
    base_ppo = _find_class(base_tree, "LeggedRobotCfgPPO")

    env_cfg = {}
    if base_env is not None:
        try:
            env_cfg.update(parse_nested_class(base_env, "env"))
        except ValueError:
            pass
    env_cfg.update(parse_nested_class(env_class, "env"))

    policy_cfg = {}
    if base_ppo is not None:
        try:
            policy_cfg.update(parse_nested_class(base_ppo, "policy"))
        except ValueError:
            pass
    try:
        policy_cfg.update(parse_nested_class(ppo_class, "policy"))
    except ValueError:
        pass

    runner_cfg = {}
    if base_ppo is not None:
        try:
            runner_cfg.update(parse_nested_class(base_ppo, "runner"))
        except ValueError:
            pass

    # Walk bases for runner overrides when using nested inheritance like M1RoughCfgPPONoCmd.
    ppo_bases = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    if base_ppo is not None:
        ppo_bases[base_ppo.name] = base_ppo

    def collect_runner(class_node, visited=None):
        visited = visited or set()
        if class_node is None or class_node.name in visited:
            return {}
        visited.add(class_node.name)
        values = {}
        for base in class_node.bases:
            if isinstance(base, ast.Name) and base.id in ppo_bases:
                values.update(collect_runner(ppo_bases[base.id], visited))
            elif isinstance(base, ast.Attribute) and base.attr in ppo_bases:
                values.update(collect_runner(ppo_bases[base.attr], visited))
        try:
            values.update(parse_nested_class(class_node, "runner"))
        except ValueError:
            pass
        return values

    runner_cfg.update(collect_runner(ppo_class))

    top_level = {}
    for stmt in ppo_class.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            top_level[stmt.targets[0].id] = eval_expr(stmt.value, top_level)
    if base_ppo is not None:
        for stmt in base_ppo.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                name = stmt.targets[0].id
                if name not in top_level:
                    top_level[name] = eval_expr(stmt.value, top_level)

    train_cfg = SimpleNamespace(
        seed=top_level.get("seed", 1),
        policy=SimpleNamespace(**policy_cfg),
        runner=SimpleNamespace(**runner_cfg),
    )
    env_cfg = SimpleNamespace(env=SimpleNamespace(**env_cfg))
    return env_cfg, train_cfg


def get_load_path(root, load_run=-1, checkpoint=-1):
    runs = os.listdir(root)
    runs.sort()
    if "exported" in runs:
        runs.remove("exported")
    if not runs:
        raise ValueError(f"No runs in this directory: {root}")

    last_run = os.path.join(root, runs[-1])
    run_dir = last_run if load_run == -1 else os.path.join(root, load_run)

    if checkpoint == -1:
        models = [file for file in os.listdir(run_dir) if "model" in file]
        models.sort(key=lambda name: f"{name:0>15}")
        if not models:
            raise ValueError(f"No checkpoint models found in: {run_dir}")
        model_name = models[-1]
    else:
        model_name = f"model_{checkpoint}.pt"

    return os.path.join(run_dir, model_name)


def update_train_cfg_from_args(train_cfg, args):
    if args.seed is not None:
        train_cfg.seed = args.seed
    if args.max_iterations is not None:
        train_cfg.runner.max_iterations = args.max_iterations
    if args.experiment_name is not None:
        train_cfg.runner.experiment_name = args.experiment_name
    if args.run_name is not None:
        train_cfg.runner.run_name = args.run_name
    if args.load_run is not None:
        train_cfg.runner.load_run = args.load_run
    if args.checkpoint is not None:
        train_cfg.runner.checkpoint = args.checkpoint
    return train_cfg


def build_actor_critic(task_name, args):
    env_cfg, train_cfg = parse_task_config(task_name)
    train_cfg = update_train_cfg_from_args(train_cfg, args)
    policy_cls = TASK_CONFIGS[task_name]["policy_class"]

    num_actor_obs = env_cfg.env.num_observations
    num_critic_obs = env_cfg.env.num_privileged_obs
    num_one_step_obs = env_cfg.env.num_one_step_observations
    num_actions = env_cfg.env.num_actions

    actor_critic = policy_cls(
        num_actor_obs,
        num_critic_obs,
        num_one_step_obs,
        num_actions,
        **vars(train_cfg.policy),
    )
    return actor_critic, env_cfg, train_cfg


def resolve_checkpoint_path(train_cfg):
    log_root = os.path.join(ROOT_DIR, "logs", train_cfg.runner.experiment_name)
    return get_load_path(
        log_root,
        load_run=train_cfg.runner.load_run,
        checkpoint=train_cfg.runner.checkpoint,
    )


def load_model(actor_critic, ckpt_path, device):
    try:
        loaded = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        loaded = torch.load(ckpt_path, map_location=device)
    actor_critic.load_state_dict(loaded["model_state_dict"])
    return loaded


def export_onnx(args):
    actor_critic, env_cfg, train_cfg = build_actor_critic(args.task, args)
    ckpt_path = resolve_checkpoint_path(train_cfg)
    load_model(actor_critic, ckpt_path, args.rl_device)

    exporter_cls = TASK_CONFIGS[args.task]["exporter_class"]
    exporter = exporter_cls(actor_critic)
    exporter.to("cpu")
    exporter.eval()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(
            ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, args.output_name)

    is_no_cmd = args.task == "m1_him_no_cmd"
    if is_no_cmd:
        history_dim = int(exporter.obs_history_no_cmd_dim)
        curr_dim = int(exporter.num_one_step_obs)
        dummy_inputs = (
            torch.zeros(1, history_dim, dtype=torch.float32),
            torch.zeros(1, curr_dim, dtype=torch.float32),
        )
        export_kwargs = dict(
            export_params=True,
            do_constant_folding=True,
            input_names=["obs_history_no_cmd", "obs_curr"],
            output_names=["actions"],
            dynamic_axes={
                "obs_history_no_cmd": {0: "batch"},
                "obs_curr": {0: "batch"},
                "actions": {0: "batch"},
            },
            opset_version=args.opset,
        )
    else:
        dummy_inputs = (
            torch.zeros(1, env_cfg.env.num_observations, dtype=torch.float32),
        )
        export_kwargs = dict(
            export_params=True,
            do_constant_folding=True,
            input_names=["obs_history"],
            output_names=["actions"],
            dynamic_axes={
                "obs_history": {0: "batch"},
                "actions": {0: "batch"},
            },
            opset_version=args.opset,
        )

    # PyTorch>=2.6 defaults dynamo=True and may require onnxscript; prefer legacy path.
    try:
        with torch.no_grad():
            torch.onnx.export(exporter, dummy_inputs, output_path, dynamo=False, **export_kwargs)
    except TypeError:
        with torch.no_grad():
            torch.onnx.export(exporter, dummy_inputs, output_path, **export_kwargs)

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Exported ONNX model to: {output_path}")
    print(f"task: {args.task}")
    if is_no_cmd:
        print(f"obs_history_no_cmd shape: {tuple(dummy_inputs[0].shape)}")
        print(f"obs_curr shape: {tuple(dummy_inputs[1].shape)}")
    else:
        print(f"obs_history shape: {tuple(dummy_inputs[0].shape)}")
    print(f"num_one_step_obs: {env_cfg.env.num_one_step_observations}")
    print(f"num_actions: {env_cfg.env.num_actions}")


if __name__ == "__main__":
    export_onnx(get_args())
