#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
Created based on the same script in gym_ducketown
"""

import sys
import argparse
import pyglet
from pyglet.window import key
import numpy as np
import cv2
import logging
import matplotlib.pyplot as plt
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.simulator import Simulator
from gym_duckietown.simulator import WHEEL_DIST
from duckietown_utils.wrappers.reward_wrappers import *
from duckietown_utils.wrappers.observation_wrappers import ClipImageWrapper, ResizeWrapper, MotionBlurWrapper, \
    RandomFrameRepeatingWrapper
from duckietown_utils.wrappers.simulator_mod_wrappers import ObstacleSpawningWrapper, ForwardObstacleSpawnnigWrapper
from duckietown_utils.wrappers.aido_wrapper import AIDOWrapper
import math
import copy

from gym import envs
all_envs = envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs if "Duckietown" in env_spec.id]
print(*env_ids, sep="\n")

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

parser = argparse.ArgumentParser()
parser.add_argument('--map-name', default='loop_empty', help="Specify the map")
parser.add_argument('--distortion', action='store_true', help='Simulate lens distortion')
parser.add_argument('--draw-curve', action='store_true', help='Draw the lane following curve')
parser.add_argument('--domain-rand', action='store_true', help='Enable domain randomization')
parser.add_argument('--top-view', action='store_true',
                    help="View the simulation from a fixed bird's eye view, instead of the robot's view")
parser.add_argument('--spawn-vehicle-ahead', action='store_true',
                    help="Generate an obstacle vehicle a few tiles ahead of the controlled one")
parser.add_argument('--show-observations', action='store_true',
                    help='Show the cropped, downscaled observations, used as the policy input')
args = parser.parse_args()

if args.top_view:
    render_mode = 'top_down'
else:
    render_mode = 'human'

env = Simulator(
    seed=1234,
    map_name=args.map_name,
    domain_rand=args.domain_rand,
    dynamics_rand=args.domain_rand,
    camera_rand=args.domain_rand,
    distortion=args.distortion,
    accept_start_angle_deg=1,
    full_transparency=True,
    draw_curve=args.draw_curve,
    # user_tile_start=[2,1]
)
env = AIDOWrapper(env)
if args.show_observations:
    env = ClipImageWrapper(env, top_margin_divider=3)
    # env = ResizeWrapper(env, (84, 84))
# env = MotionBlurWrapper(env)
# env = RandomFrameRepeatingWrapper(env, {"frame_repeating": 0.33333})
# env = ObstacleSpawningWrapper(env, {'obstacles': {'duckie': {'density': 0.35,
#                                                              'static': True},
#                                                   'duckiebot': {'density': 0.25,
#                                                                 'static': True},
#                                                   'cone': {'density': 0.25,
#                                                                 'static': True},
#                                                   'barrier': {'density': 0.15,
#                                                            'static': True},
#                                                   },
#                                     'spawn_obstacles': True
#                                     }
if args.spawn_vehicle_ahead:
    env = ForwardObstacleSpawnnigWrapper(env, {'spawn_forward_obstacle': True})
# env = DtRewardWrapper(env)
# env = DtRewardClipperWrapper(env)
# env = DtRewardWrapperDistanceTravelled(env)
env = DtRewardPosAngle(env)
env = DtRewardVelocity(env)
env = DtRewardCollisionAvoidance(env)

env.reset()
env.reset()
env.render(render_mode)



# # import numpy as np
# import math
# import copy
# from gym_duckietown.simulator import WHEEL_DIST  # Use the true value



class BoltzmannPolicyAgent:
    def __init__(self):
        
        # self.trim = 0.0
        # self.radius = 0.0318
        self.WHEEL_DIST = WHEEL_DIST  # 10.2 cm between wheels (actual value)
        # self.robot_width = ROBOT_WIDTH
        # self.robot_length = ROBOT_LENGTH
        # self.gain = 2.
        # self.k = 27.0
        # self.limit = 1.0
        self.temperature = 0.01  # Exploration parameter (lower = more greedy)
        self.action_space = self._get_action_space()
        
    def _get_action_space(self):
        # Define discrete action space (steering values)
        steering_vals = np.linspace(-1, 1, 21)  # 7 steering levels
        actions = []
        for steer in steering_vals:
            left = np.clip(1 + steer, 0.0, 1.0)
            right = np.clip(1 - steer, 0.0, 1.0)
            actions.append([left, right])
        return actions
    
    def predict_next_state(self, current_pos, current_angle, action, dt=0.1):
        # Simplified kinematic model
        left_speed, right_speed = action
        linear_vel = (left_speed + right_speed) / 2
        angular_vel = (right_speed - left_speed) / (self.WHEEL_DIST)
        
        delta_angle = angular_vel * dt
        new_angle = current_angle + delta_angle
        
        delta_x = linear_vel * math.cos(new_angle) * dt
        delta_z = linear_vel * math.sin(new_angle) * dt
        new_pos = current_pos + np.array([delta_x, 0, delta_z])
        
        return new_pos, new_angle
    
    # def compute_cost(self, simulator, predicted_pos, predicted_angle,action):
    #     # Cost based on path alignment
    #     closest_point, closest_tangent = simulator.closest_curve_point(predicted_pos, predicted_angle)
    #     if closest_point is None:
    #         return 1000  # High cost if off the path
        
    #     # Distance error
    #     dist_error = np.linalg.norm(predicted_pos - closest_point)
        
    #     # Heading alignment error
    #     right_vec = np.array([math.sin(predicted_angle), 0, math.cos(predicted_angle)])
    #     heading_error = 1 - np.dot(right_vec, closest_tangent)
        
    #     return 1.0 * dist_error + 0.5 * heading_error
    
    def compute_cost(self, simulator, predicted_pos, predicted_angle, action):
        closest_point, closest_tangent = simulator.closest_curve_point(predicted_pos, predicted_angle)
        
        iterations = 0

        lookup_distance = self.follow_dist
        curve_point = None
        while iterations < self.max_iterations:
            # Project a point ahead along the curve tangent,
            # then find the closest point to to that
            follow_point = closest_point + closest_tangent * lookup_distance
            curve_point, curve_tangent = simulator.closest_curve_point(follow_point, simulator.cur_angle)

            # If we have a valid point on the curve, stop
            if curve_point is not None:
                break

            iterations += 1
            lookup_distance *= 0.5
        
        if closest_point is None:
            return 1000  # Extreme penalty for being off-road

        # Non-linear distance penalty (quadratic)
        dist_error = np.linalg.norm(predicted_pos - closest_point)
        dist_cost = 5.0 * (dist_error ** 2)  # Heavily penalize large deviations

        # Heading alignment cost (dot product)
        right_vec = np.array([math.sin(predicted_angle), 0, math.cos(predicted_angle)])
        heading_cost = 2.0 * (1 - np.dot(right_vec, closest_tangent))

        # # Additional penalty for sharp steering
        steering = (action[1] - action[0]) / 2  # action is [left, right]
        steering_cost = 0.3 * abs(steering)  # Penalize aggressive turns

        return dist_cost + heading_cost + steering_cost
    
    def step(self, simulator):
        current_pos = simulator.cur_pos
        current_angle = simulator.cur_angle
        costs = []
        
        # Evaluate all actions
        for action in self.action_space:
            pred_pos, pred_angle = self.predict_next_state(current_pos, current_angle, action)
            cost = self.compute_cost(simulator, pred_pos, pred_angle, action)
            costs.append(cost)
        
        # Softmax probabilities
        costs = np.array(costs)
        probs = np.exp(-costs/1.0)
        probs /= probs.sum()
        
        # Sample action
        action_idx = np.random.choice(len(self.action_space), p=probs)
        return self.action_space[action_idx]

class BaselinePIDAgent:
    def __init__(self):
        self.follow_dist = 0.15
        self.P = 0.5
        self.D = 5
        # self.trim = 0.0
        # self.radius = 0.0318
        # self.wheel_dist = WHEEL_DIST
        # self.robot_width = ROBOT_WIDTH
        # self.robot_length = ROBOT_LENGTH
        # self.gain = 2.
        # self.k = 27.0
        # self.limit = 1.0
        self.max_iterations = 1000
        self.prev_e = 0

    def step(self, simulator: Simulator):
        """
        Take a step, implemented as a PID controller
        """

        # Find the curve point closest to the agent, and the tangent at that point
        closest_point, closest_tangent = simulator.closest_curve_point(simulator.cur_pos, simulator.cur_angle)

        iterations = 0

        lookup_distance = self.follow_dist
        curve_point = None
        while iterations < self.max_iterations:
            # Project a point ahead along the curve tangent,
            # then find the closest point to to that
            follow_point = closest_point + closest_tangent * lookup_distance
            curve_point, curve_tangent = simulator.closest_curve_point(follow_point, simulator.cur_angle)

            # If we have a valid point on the curve, stop
            if curve_point is not None:
                break

            iterations += 1
            lookup_distance *= 0.5

        # Compute a normalized vector to the curve point
        point_vec = curve_point - simulator.cur_pos
        point_vec /= np.linalg.norm(point_vec)

        magic = (curve_tangent + point_vec) / np.linalg.norm(np.linalg.norm(point_vec))
        e = np.dot(self.get_right_vec(simulator.cur_angle), magic)
        de = e - self.prev_e
        self.prev_e = e
        steering = self.P * e + self.D * de
        return np.clip(np.array([1 + steering, 1 - steering]), 0., 1.)


    @staticmethod
    def get_right_vec(angle):
        x = math.sin(angle)
        z = math.cos(angle)
        return np.array([x, 0, z])


# Initialize PID agent
# After PID agent initialization
pid_agent = BaselinePIDAgent()
boltzmann_agent = BoltzmannPolicyAgent()
mode = "manual"  # Can be "manual", "pid", "boltzmann"
# use_pid_agent = False  # Start in manual mode

# @env.unwrapped.window.event
# def on_key_press(symbol, modifiers):
#     global use_pid_agent
#     # ... (existing code) ...
#     if symbol == key.P:
#         use_pid_agent = not use_pid_agent
#         mode = "PID" if use_pid_agent else "Manual"
#         print(f"Switched to {mode} mode")

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    global mode
    """
    This handler processes keyboard commands that
    control the simulation
    """
    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)
    
    if symbol == key.P:
        mode = "pid"
        print("Switched to PID mode")
    elif symbol == key.B:
        mode = "boltzmann"
        print("Switched to Boltzmann mode")
    elif symbol == key.M:
        mode = "manual"
        print("Switched to Manual mode")

# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)
def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    global mode
    
    if mode == "manual":
        # Manual control logic
        action = np.array([0.0, 0.0])
        if key_handler[key.UP]:
            action = np.array([1., 1.])
        if key_handler[key.DOWN]:
            action = np.array([-1., -1.])
        if key_handler[key.LEFT]:
            action = np.array([0, 1.])
        if key_handler[key.RIGHT]:
            action = np.array([1, 0.])
        if key_handler[key.SPACE]:
            action = np.array([0, 0])
        if key_handler[key.R]:
            action = np.array([0, 0])
            env.reset()
            
    elif mode == "pid":
        action = pid_agent.step(env.unwrapped)
        
    elif mode == "boltzmann":
        action = boltzmann_agent.step(env.unwrapped)

    obs, reward, done, info = env.step(action)
    if args.show_observations:
        # obs = cv2.resize(obs, (300, 300))
        cv2.imshow("Observation", cv2.cvtColor(obs, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)
    print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))

    if key_handler[key.RETURN]:
        from PIL import Image
        im = Image.fromarray(obs)
        im.save('screen.png')

    if done:
        print('done!')
        env.reset()
        env.render(render_mode)

    env.render(render_mode)

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
