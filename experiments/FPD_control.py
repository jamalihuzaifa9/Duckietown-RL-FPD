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
from scipy import stats as st

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

# print('position:',Simulator.cur_pos())

import numpy as np
import math
from itertools import product

class StochasticPolicyAgentLookAhead:
    def __init__(self, simulator,num_v=5, num_omega=5, beta=1.0, follow_dist=0.06, max_iterations=1000):
        """
        Args:
            simulator: The simulator object (used for accessing current state and the reference curve).
            num_v: Number of linear velocity samples.
            num_omega: Number of angular velocity samples.
            beta: Temperature parameter for the softmax (higher beta -> more deterministic).
            follow_dist: Look-ahead distance on the curve.
            max_iterations: Maximum iterations to search for a valid look-ahead point.
        """
        self.simulator = simulator
        self.num_v = num_v
        self.num_omega = num_omega
        self.beta = beta
        self.follow_dist = follow_dist
        self.max_iterations = max_iterations
        # self.q_action = q_action
        self.WHEEL_DIST = WHEEL_DIST

        # Define a grid for the two-dimensional action space: (v, omega)
        # For example, let v vary between 0.5 and 1.0 and omega between -0.5 and 0.5.
        # v_values = np.linspace(0.5, 1.5, num_v)
        # omega_values = np.linspace(-0.5, 0.5, num_omega)
        # self.action_space = np.array(list(product(v_values,omega_values)))
        self.action_space = np.linspace(-1.0, 1.0, num_v)

    def cost_function(self, predicted_state, desired_point, curve_tangent):
        """
        Computes the cost of a predicted state relative to the desired point.
        For simplicity, we use the squared Euclidean distance in the (x, z) position space.

        Args:
            predicted_state: The predicted next state as [x, z, theta].
            desired_point: The desired point on the curve as [x, z] or [x, z, theta].

        Returns:
            A scalar cost.
        """
        # Compute a normalized vector to the curve point
        point_vec = desired_point - predicted_state[:3]
        point_vec /= np.linalg.norm(point_vec)
        
        magic = (curve_tangent + point_vec) / np.linalg.norm(np.linalg.norm(point_vec))
        e = np.dot(self.get_right_vec(predicted_state[-1]), magic)
       
        return e

    def predict_next_state(self, simulator, steering):
        """
        Predicts the next state given a candidate action (v, omega) using a simple unicycle model.

        Args:
            simulator: The simulator object to access the current state.
            v: Linear velocity.
            omega: Angular velocity.

        Returns:
            The predicted next state as a numpy array: [x_next, z_next, theta_next].
        """
        left_speed, right_speed =  np.clip(np.array([1 + steering, 1 - steering]), 0., 1.)
        v = (left_speed + right_speed) / 2
        omega = (right_speed - left_speed) / (self.WHEEL_DIST)
        
        dt = 0.1  # Time step (this can be adjusted or made a parameter)
        x,_,z = simulator.cur_pos
        theta = simulator.cur_angle

        # Simple kinematics
        x_next = x + dt * v * math.cos(theta)
        z_next = z + dt * v * math.sin(theta)
        theta_next = theta + dt * omega

        pred_state = np.array([x_next,0.0,z_next,theta_next])      
        return pred_state

    def step(self, simulator, q_action):
        """
        1. Uses a look-ahead along the reference curve to get the desired point.
        2. For each candidate action, predicts the next state and computes the cost.
        3. Converts the costs into a probability distribution and samples an action.
        4. Returns the differential drive commands for that action.
        """
        # --- Step 1: Look Ahead on the Curve ---
        # Get the closest point and its tangent on the reference curve
    
        q_pf = st.norm.pdf(q_action,self.action_space,0.08)
        
        closest_point, closest_tangent = simulator.closest_curve_point(simulator.cur_pos, simulator.cur_angle)

        # Look ahead along the tangent from the closest point
        iterations = 0
        lookup_distance = self.follow_dist
        desired_point = None
        desired_tangent = None

        while iterations < self.max_iterations:
            follow_point = closest_point + closest_tangent * lookup_distance
            desired_point, desired_tangent = simulator.closest_curve_point(follow_point, simulator.cur_angle)
            if desired_point is not None:
                break  # Found a valid desired point
            iterations += 1
            lookup_distance *= 0.5

        # If no valid look-ahead point is found, fall back to the closest point.
        if desired_point is None:
            desired_point = closest_point

        # --- Step 2: Evaluate Candidate Actions ---
        costs = []
        candidate_states = []

        for v in self.action_space:
            # Predict the next state for action (v, omega)
            predicted_state = self.predict_next_state(simulator, v)
            candidate_states.append(predicted_state)
            # Compute the cost relative to the desired point on the curve
            cost = self.cost_function(predicted_state, desired_point,desired_tangent)
            costs.append(cost)

        costs = np.array(costs)

        # --- Step 3: Convert Costs to a Probability Distribution ---
        # Lower cost should mean higher probability.
        probabilities = q_pf*np.exp(-self.beta * costs)
        probabilities /= np.sum(probabilities)

        # Sample one action from the candidate actions using the computed probabilities
        chosen_index = np.random.choice(len(self.action_space), p=probabilities)
        chosen_action = self.action_space[chosen_index]
        steering = chosen_action

        # --- Step 4: Map the (v, omega) Action to Differential Drive Commands ---
        # For example, one simple mapping is:
        left_wheel = np.clip(1 + steering, 0., 1.)
        right_wheel = np.clip(1 - steering, 0., 1.)

        return np.array([left_wheel, right_wheel])
    
    @staticmethod
    def get_right_vec(angle):
        x = math.sin(angle)
        z = math.cos(angle)
        return np.array([x, 0, z])



class BoltzmannPolicyAgent:
    def __init__(self):
        
        self.follow_dist = 0.15
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

        lookup_distance = 0.15
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
        self.follow_dist = 0.06
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
        nsteering = np.clip(steering,-1.0,1.0)
        return nsteering, np.clip(np.array([1 + steering, 1 - steering]), 0., 1.)

    @staticmethod
    def get_right_vec(angle):
        x = math.sin(angle)
        z = math.cos(angle)
        return np.array([x, 0, z])


# Initialize PID agent
# After PID agent initialization
pid_agent = BaselinePIDAgent()
# boltzmann_agent = BoltzmannPolicyAgent()
boltzmann_agent = StochasticPolicyAgentLookAhead(Simulator)
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
        steering,action = pid_agent.step(env.unwrapped)
        
    elif mode == "boltzmann":
        steering,q_action = pid_agent.step(env.unwrapped)
        action = boltzmann_agent.step(env.unwrapped,steering)

    obs, reward, done, info = env.step(action)
    if args.show_observations:
        # obs = cv2.resize(obs, (300, 300))
        cv2.imshow("Observation", cv2.cvtColor(obs, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)
    print('obs:',env.unwrapped.cur_pos)
    # print('step_count = %s, obs=%.3f' % (env.unwrapped.step_count, obs))

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
