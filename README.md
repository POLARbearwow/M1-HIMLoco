## HIMLoco for Go2W  

### 🚀 Project Overview  
This project is based on [HIMLoco](https://github.com/OpenRobotLab/HIMLoco) and implements RL training for Go2W in Isaac Gym.  
- Train a policy:
```bash
cd legged_gym/scripts
python train.py --task=go2w
```
- Play and export the latest policy:
```bash
cd legged_gym/scripts
python play.py --task=go2w
```
### 🕹️ MuJoCo Validation
Update the paths in `config.yaml` to your local absolute paths:  

```bash
cd mujoco
python pdandrl.py
```

For `m1`, export the latest checkpoint as both `jit` and `onnx` first:

```bash
cd legged_gym/scripts
python play.py --task=m1_him
```

Then launch the ONNX-based MuJoCo deploy script:

```bash
cd mujoco
python deploy_mujoco_m1.py
```

`m1` keyboard fallback: `W/S` forward-backward, `Q/E` strafe, `A/D` yaw, `Space` zero command, `R` reset.

| Keyboard   | Function         |
| ----- | ---------------- |
| W/S   | Forward/Backward |
| A/D   | Left/Right       |
| Q/E   | Yaw rotation     |
| Space | Reset zero       |

**⚠️ Known Issues**
During standing, the PD controller should track an interpolation curve, but currently only the final target state is tracked. You can  **Reset**  to achieve successful standing.
For more accurate Sim-to-Sim validation and real-world deployment, refer to [rl_sar](https://github.com/fan-ziqi/rl_sar)
### 📚 References
- [HIMLoco](https://github.com/OpenRobotLab/HIMLoco)

- [legged_gym](https://github.com/leggedrobotics/legged_gym)

- [rl_sar](https://github.com/fan-ziqi/rl_sar)
