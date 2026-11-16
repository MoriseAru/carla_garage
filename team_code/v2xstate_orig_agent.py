"""V2XState-orig closed-loop agent for Bench2Drive (MIYABI, HANDOVER 11/16 V2XState-orig sections).

The original V2XState (VTC2026: CenterPoint + WaypointPlannerV2XStateLearnableGate) retrained on PDM-Lite, from the
V2XState_Real repo package `v2xo` (V2XO_REPO, default the sibling checkout). Sensors, GPS/UKF localisation, route planner,
command and LiDAR half-sweep merging are SensorAgent's (TF++); the model runs through v2xo.infer.V2XStateOrigRunner
(the same input assembly as training).

Control:
- longitudinal = the original V2X_Controller (simulation/leaderboard/team_code/v2x_controller.py, PID from
  pnp_config_v2xstate_5_10.yaml): desired speed = 5 x mean spacing of the first 3 waypoints (0.2 s apart),
  delta clipped to [0, 0.35], throttle <= 0.75, brake when speed > 1.1 x desired, throttle 0 while braking,
  100 stopped ticks -> 12 ticks of throttle 0.8;
- lateral (user-approved change 2) = TF++ control_pid waypoint aiming (aim distance 2.25 m below 5.5 m/s desired speed,
  3.0 m above; turn PID 1.25 / 0.75 / 0.3, window 20) on the predicted waypoints instead of the original target-point
  steering. The 2.25 m minimum aim distance already satisfies the LAT_CL_FLOOR=2 floor.

TEAM_CONFIG = a directory with percep.pth, light.pth, planner.pth and the TF++ config.json of tfpp_base_000 (sensor /
route-planner settings only; no model_*.pth, so SensorAgent loads no TF++ network).
Env: V2XO_REPO (repo path), V2XO_STRIDE (history stride in ticks, default 5 = 0.25 s at 20 Hz), V2XO_LOG (jsonl per-tick log).
"""
import json
import os
import sys
from collections import deque
from copy import deepcopy

import carla
import numpy as np
import torch

from sensor_agent import SensorAgent
import transfuser_utils as t_u

_REPO = os.environ.get('V2XO_REPO', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'V2XState_Real'))
if _REPO not in sys.path:
  sys.path.insert(0, _REPO)
os.environ.setdefault('V2XO_MAP_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'birds_eye_view',
                                                   'maps_2ppm_cv'))
from v2xo.infer import V2XStateOrigRunner  # pylint: disable=wrong-import-position


def get_entry_point():
  return 'V2XStateOrigAgent'


class _PID:
  """simulation/leaderboard/team_code/v2x_controller.py PIDController."""

  def __init__(self, k_p, k_i, k_d, n):
    self.k_p, self.k_i, self.k_d = k_p, k_i, k_d
    self.window = deque([0.0] * n, maxlen=n)

  def step(self, error):
    self.window.append(error)
    integral = np.mean(self.window) if len(self.window) >= 2 else 0.0
    derivative = self.window[-1] - self.window[-2] if len(self.window) >= 2 else 0.0
    return self.k_p * error + self.k_i * integral + self.k_d * derivative


class V2XStateOrigAgent(SensorAgent):

  def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
    super().setup(path_to_conf_file, route_index, traffic_manager)
    assert len(self.nets) == 0, 'TEAM_CONFIG for V2XState-orig must not contain TF++ model_*.pth files'
    self.stop_sign_controller = 0
    self.runner = V2XStateOrigRunner(self.config_path, hist_stride=int(os.environ.get('V2XO_STRIDE', 5)))
    # original longitudinal controller (pnp_config_v2xstate_5_10.yaml)
    self.speed_pid = _PID(5.0, 1.0, 0.1, 20)
    self.max_speed, self.clip_delta, self.max_throttle, self.brake_ratio = 5.0, 0.35, 0.75, 1.1
    self.stop_steps, self.forced_forward_steps = 0, 0
    # TF++ waypoint lateral control (model.py control_pid)
    self.turn_pid = t_u.PIDController(k_p=self.config.turn_kp, k_i=self.config.turn_ki, k_d=self.config.turn_kd,
                                      n=self.config.turn_n)
    self.town = None
    self.v2xo_log = open(os.environ['V2XO_LOG'], 'a', encoding='utf-8') if os.environ.get('V2XO_LOG') else None
    print(f'V2XState-orig agent: ckpt {self.config_path}, history stride {self.runner.stride} ticks', flush=True)

  def _init(self):
    super()._init()
    from srunner.scenariomanager.carla_data_provider import CarlaDataProvider  # pylint: disable=import-outside-toplevel
    self.town = CarlaDataProvider.get_map().name.split('/')[-1]
    self.runner.reset()

  def _control(self, wp, speed):
    # --- original longitudinal (v2x_controller.py run_step) ---
    if speed < 0.2:
      self.stop_steps += 1
    else:
      self.stop_steps = max(0, self.stop_steps - 10)
    displacement = np.linalg.norm(wp, ord=2, axis=1)
    desired_speed = float(np.mean(np.diff(displacement)[:3]) * 5 * self.max_speed / 5)
    delta = np.clip(desired_speed - speed, 0.0, self.clip_delta)
    throttle = float(np.clip(self.speed_pid.step(delta), 0.0, self.max_throttle))
    brake = speed > desired_speed * self.brake_ratio
    if self.stop_steps > 100:
      self.forced_forward_steps = 12
      self.stop_steps = 0
    if self.forced_forward_steps > 0:
      throttle, brake = 0.8, False
      self.forced_forward_steps -= 1
    # --- TF++ control_pid lateral on the predicted waypoints (x fwd, y right) ---
    aim_distance = self.config.aim_distance_slow if desired_speed < self.config.aim_distance_threshold \
        else self.config.aim_distance_fast
    aim_index = wp.shape[0] - 1
    for index, p in enumerate(wp):
      if np.linalg.norm(p) >= aim_distance:
        aim_index = index
        break
    angle = np.degrees(np.arctan2(wp[aim_index][1], wp[aim_index][0])) / 90.0
    if speed < 0.01 or brake:
      angle = 0.0
    steer = float(np.clip(self.turn_pid.step(angle), -1.0, 1.0))
    if brake:
      throttle = 0.0   # original post-processing: brake > 0.1 -> throttle 0
    return steer, throttle, float(brake), desired_speed

  @torch.inference_mode()
  def run_step(self, input_data, timestamp, sensors=None):  # pylint: disable=unused-argument
    self.step += 1
    if not self.initialized:
      self._init()
      self.control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
      tick_data = self.tick(input_data)
      self.lidar_last = deepcopy(tick_data['lidar'])
      return self.control

    tick_data = self.tick(input_data)
    ego_x, ego_y, ego_theta = self.state_log[-1][0], self.state_log[-1][1], self.state_log[-1][2]
    lx, ly, lt = self.state_log[-2][0], self.state_log[-2][1], self.state_log[-2][2]
    lidar_last = self.align_lidar(self.lidar_last, lx, ly, lt, ego_x, ego_y, ego_theta)
    lidar_full = np.concatenate((deepcopy(tick_data['lidar']), lidar_last), axis=0)
    self.lidar_last = deepcopy(tick_data['lidar'])

    image = tick_data['rgb'][0].permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()      # RGB, TF++ crop, 384 x 1024
    command = int(tick_data['command'][0].argmax().item()) + 1                             # inverse of command_to_one_hot
    target = tick_data['target_point'][0].cpu().numpy()
    speed = float(tick_data['speed'][0].item())
    out = self.runner.step(lidar_full, image, speed, command, target, (ego_x, ego_y, ego_theta), self.town)
    wp = np.around(out['waypoints'], decimals=2)                                          # original rounds to 2 dp
    steer, throttle, brake, desired = self._control(wp, speed)
    self.control = carla.VehicleControl(steer=steer, throttle=throttle, brake=brake)
    if self.v2xo_log is not None:
      self.v2xo_log.write(json.dumps({'step': self.step, 'speed': speed, 'desired': desired, 'steer': steer,
                                      'throttle': throttle, 'brake': brake, 'cmd': command,
                                      'tp': [float(v) for v in target], 'wp': wp.tolist(),
                                      'flags': [float(v) for v in out['flags']], 'tokens': out['num_tokens'],
                                      'gate': [float(v) for v in out['gate']]}) + '\n')
    return self.control

  def destroy(self, results=None):
    if self.v2xo_log is not None:
      self.v2xo_log.close()
    del self.runner
    torch.cuda.empty_cache()
