import numpy as np
import math

class SafeRLAgent:
    def __init__(self):
        """
        Initializes the Safe RL Agent.
        This agent uses Lagrangian Relaxation for hard safety constraints
        and an Emergency Brake Layer for immediate collision avoidance.
        """
        # Lagrangian Multiplier (lambda) for the safety constraint
        # Starts small, grows if safety constraints are violated
        self.lambda_multiplier = 0.1
        self.lambda_lr = 0.05  # Learning rate for the multiplier
        
        # State space dimensions (Camera Features + Ego State + Obstacles)
        self.state_dim = 512 + 4 + 10 # Example dimensions
        
        # Action space: [Throttle (0 to 1), Steer (-1 to 1), Brake (0 to 1)]
        self.action_dim = 3
        
        print("Safe RL Agent Initialized with Lagrangian Relaxation.")

    def compute_reward(self, ego_speed, speed_limit, distance_to_center, traffic_light_state, jerk=0.0, energy_used=0.0):
        """
        Computes the primary objective reward (Efficiency, Comfort, Law Adherence)
        """
        reward = 0.0
        
        # 1. Maintain Speed (Efficiency)
        speed_diff = abs(speed_limit - ego_speed)
        if ego_speed <= speed_limit:
            reward += (ego_speed / speed_limit) * 10.0
        else:
            # Overspeeding Penalty
            reward -= speed_diff * 2.0
            
        # 2. Stay in Lane
        # Penalty increases quadratically with distance from center
        reward -= (distance_to_center ** 2) * 5.0
        
        # 3. Traffic Law (Traffic Light)
        if traffic_light_state == 'Red' and ego_speed > 0.5:
            reward -= 100.0 # Huge penalty for running a red light

        # 4. Comfort (Jerk Penalty)
        # Higher jerk = more uncomfortable driving = more penalty
        reward -= abs(jerk) * 0.5
        
        # 5. Energy Efficiency (Consumption Penalty)
        # Higher energy usage = more penalty
        reward -= energy_used * 0.01
            
        return reward

    def compute_constraint_cost(self, ego_transform, predicted_obstacles):
        """
        Computes the cost for the Lagrangian constraint (Safety).
        Returns a high cost if ego is dangerously close to predicted obstacle paths.
        """
        cost = 0.0
        ego_x, ego_y = ego_transform['x'], ego_transform['y']
        
        for obs_id, pred_pos in predicted_obstacles.items():
            dist = math.sqrt((ego_x - pred_pos[0])**2 + (ego_y - pred_pos[1])**2)
            if dist < 2.0:  # Collision threshold
                cost += 1.0 # 1 collision
                
        return cost

    def update_lagrangian(self, constraint_cost, cost_limit=0.0):
        """
        Updates the Lagrangian multiplier based on constraint violations.
        """
        # If cost > limit, lambda increases (penalizing policy more).
        # If cost < limit, lambda decreases (up to 0).
        self.lambda_multiplier += self.lambda_lr * (constraint_cost - cost_limit)
        self.lambda_multiplier = max(0.0, self.lambda_multiplier) # Cannot be negative

    def emergency_brake_layer(self, ego_transform, ego_speed, obstacles):
        """
        Safety Shield: Deterministic rule-based override.
        Calculates Time-To-Collision (TTC) and forces a stop if critical.
        """
        ego_x, ego_y = ego_transform['x'], ego_transform['y']
        
        for obs in obstacles:
            obs_x, obs_y = obs['x'], obs['y']
            dist = math.sqrt((ego_x - obs_x)**2 + (ego_y - obs_y)**2)
            
            # Simple TTC calculation (assuming obstacle is static for worst-case)
            if ego_speed > 0.1:
                ttc = dist / ego_speed
            else:
                ttc = float('inf')
                
            # If TTC is less than 1.5 seconds, trigger emergency brake
            if ttc < 1.5 and dist < 10.0:
                print(f"!!! EMERGENCY BRAKE TRIGGERED !!! TTC: {ttc:.2f}s, Dist: {dist:.2f}m")
                return True
                
        return False

    def select_action(self, state, ego_transform, ego_speed, obstacles, predicted_obstacles):
        """
        Selects an action using the RL policy, but overrides if the safety shield triggers.
        """
        # 1. Check Safety Shield (Emergency Brake Layer)
        if self.emergency_brake_layer(ego_transform, ego_speed, obstacles):
            # Override Policy: Throttle=0, Steer=0 (or maintain), Brake=1
            return [0.0, 0.0, 1.0]
            
        # 2. Get policy action (Placeholder for actual Neural Network forward pass)
        # In a real setup, action = self.actor_network(state)
        # Here we simulate a dummy action: [Throttle, Steer, Brake]
        policy_action = [0.5, 0.0, 0.0] 
        
        return policy_action
