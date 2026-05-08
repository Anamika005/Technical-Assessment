import numpy as np
import math

class SafeRLAgent:
    def __init__(self):

        self.lambda_multiplier = 0.1
        self.lambda_lr = 0.05

        self.state_dim = 512 + 4 + 10

        self.action_dim = 3

        print("Safe RL Agent Initialized with Lagrangian Relaxation.")

    def compute_reward(self, ego_speed, speed_limit, distance_to_center, traffic_light_state, jerk=0.0, energy_used=0.0):

        reward = 0.0

        speed_diff = abs(speed_limit - ego_speed)
        if ego_speed <= speed_limit:
            reward += (ego_speed / speed_limit) * 10.0
        else:

            reward -= speed_diff * 2.0

        reward -= (distance_to_center ** 2) * 5.0

        if traffic_light_state == 'Red' and ego_speed > 0.5:
            reward -= 100.0

        reward -= abs(jerk) * 0.5

        reward -= energy_used * 0.01

        return reward

    def compute_constraint_cost(self, ego_transform, predicted_obstacles):

        cost = 0.0
        ego_x, ego_y = ego_transform['x'], ego_transform['y']

        for obs_id, pred_pos in predicted_obstacles.items():
            dist = math.sqrt((ego_x - pred_pos[0])**2 + (ego_y - pred_pos[1])**2)
            if dist < 2.0:
                cost += 1.0

        return cost

    def update_lagrangian(self, constraint_cost, cost_limit=0.0):

        self.lambda_multiplier += self.lambda_lr * (constraint_cost - cost_limit)
        self.lambda_multiplier = max(0.0, self.lambda_multiplier)

    def emergency_brake_layer(self, ego_transform, ego_speed, obstacles):

        ego_x, ego_y = ego_transform['x'], ego_transform['y']

        for obs in obstacles:
            obs_x, obs_y = obs['x'], obs['y']
            dist = math.sqrt((ego_x - obs_x)**2 + (ego_y - obs_y)**2)

            if ego_speed > 0.1:
                ttc = dist / ego_speed
            else:
                ttc = float('inf')

            if ttc < 1.5 and dist < 10.0:
                print(f"!!! EMERGENCY BRAKE TRIGGERED !!! TTC: {ttc:.2f}s, Dist: {dist:.2f}m")
                return True

        return False

    def select_action(self, state, ego_transform, ego_speed, obstacles, predicted_obstacles):

        if self.emergency_brake_layer(ego_transform, ego_speed, obstacles):

            return [0.0, 0.0, 1.0]

        policy_action = [0.5, 0.0, 0.0]

        return policy_action