import carla
import random
import time
import math
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces

random.seed(42)
np.random.seed(42)

from perception import CameraPerception, LidarPerception
from predict import TrajectoryPredictor
from rl_agent import SafeRLAgent
import sensor_preprocess

class CarlaEnv(gym.Env):
    def __init__(self, host='localhost', port=2000):
        super(CarlaEnv, self).__init__()

        print("\nConnecting to CARLA...")
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

        self.camera_perception = CameraPerception()
        self.lidar_perception = LidarPerception()
        self.predictor = TrajectoryPredictor()
        self.agent = SafeRLAgent()

        self.prev_speed = 0.0
        self.prev_accel = 0.0
        self.dt = 0.05

        self.action_space = spaces.Box(low=np.array([-1.0, 0.0, 0.0]),
                                       high=np.array([1.0, 1.0, 1.0]),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.agent.state_dim,),
                                            dtype=np.float32)

        self.vehicle = None
        self.camera = None
        self.lidar = None
        self.latest_image = None
        self.actors_to_cleanup = []

    def _camera_callback(self, image):
        try:
            array = np.ndarray(
                shape=(image.height, image.width, 4),
                dtype=np.uint8,
                buffer=image.raw_data
            )
            self.latest_image = np.copy(array)

            cv2.imshow("AI Driver Camera View", array[:, :, :3])
            cv2.waitKey(1)
        except Exception as e:
            print(f"Warning: Camera buffer error -> {e}")

    def set_scenario(self, weather, time_of_day):

        weather_preset = carla.WeatherParameters()
        if weather == 'Rain':
            weather_preset.precipitation = 80.0
            weather_preset.wetness = 100.0
        elif weather == 'Clear':
            weather_preset.precipitation = 0.0
            weather_preset.wetness = 0.0

        if time_of_day == 'Night':
            weather_preset.sun_altitude_angle = -15.0
        elif time_of_day == 'Day':
            weather_preset.sun_altitude_angle = 45.0

        self.world.set_weather(weather_preset)
        print(f"Scenario Set: {weather}, {time_of_day}")

    def reset(self, scenario='Clear_Day', spawn_obstacles=False, n_traffic=0):

        for actor in self.actors_to_cleanup:
            if actor is not None:
                actor.destroy()
        self.actors_to_cleanup = []

        if self.vehicle is not None:
            self.vehicle.destroy()
        if self.camera is not None:
            self.camera.destroy()

        parts = scenario.split('_')
        weather = parts[0]
        time_of_day = parts[1]
        self.set_scenario(weather, time_of_day)

        vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_points = self.world.get_map().get_spawn_points()
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

        self.vehicle.set_autopilot(True)
        print(f"AI Autopilot Enabled - {scenario}")

        if spawn_obstacles:
            self._spawn_adversarial_actors()

        if n_traffic > 0:
            self._spawn_background_traffic(n_traffic)

        cam_bp = self.blueprint_library.find('sensor.camera.rgb')
        cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(cam_bp, cam_transform, attach_to=self.vehicle)
        self.camera.listen(lambda image: self._camera_callback(image))

        self.prev_speed = 0.0
        self.prev_accel = 0.0
        time.sleep(2)

        return self._get_obs()

    def _spawn_background_traffic(self, n_vehicles):

        print(f"Spawning {n_vehicles} background vehicles...")
        blueprints = self.blueprint_library.filter('vehicle.*')
        spawn_points = self.world.get_map().get_spawn_points()

        tm = self.client.get_trafficmanager(8000)
        tm.set_global_distance_to_leading_vehicle(2.5)

        for i in range(n_vehicles):
            bp = random.choice(blueprints)
            if 'bike' in bp.id or 'cycle' in bp.id: continue

            spawn_point = random.choice(spawn_points)
            vehicle = self.world.try_spawn_actor(bp, spawn_point)
            if vehicle:
                vehicle.set_autopilot(True, tm.get_port())
                self.actors_to_cleanup.append(vehicle)

    def _spawn_adversarial_actors(self):

        print("Spawning Pedestrians and Cyclists...")
        v_trans = self.vehicle.get_transform()
        v_fwd = v_trans.get_forward_vector()

        peds = self.blueprint_library.filter('walker.pedestrian.*')
        if peds:
            ped_bp = random.choice(peds)
            ped_loc = v_trans.location + v_fwd * 15.0 + carla.Location(z=1.0)
            ped = self.world.try_spawn_actor(ped_bp, carla.Transform(ped_loc))
            if ped: self.actors_to_cleanup.append(ped)

        all_vehicles = self.blueprint_library.filter('vehicle.*')
        bike_bp = None
        for bp in all_vehicles:
            if any(x in bp.id.lower() for x in ['bike', 'cycle', 'diamond', 'bh', 'gazelle']):
                bike_bp = bp
                break

        if bike_bp:
            bike_loc = v_trans.location + v_fwd * 25.0 + carla.Location(z=0.5)
            bike = self.world.try_spawn_actor(bike_bp, carla.Transform(bike_loc))
            if bike: self.actors_to_cleanup.append(bike)

    def _get_obs(self):

        if self.latest_image is not None:
            preproc_img = sensor_preprocess.preprocess_camera_frame(self.latest_image)
            img_features = self.camera_perception.extract_features(preproc_img)
        else:
            img_features = np.zeros(512)

        v = self.vehicle.get_velocity()
        speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)
        steer = self.vehicle.get_control().steer
        ego_state = np.array([speed, steer, 0.0, 0.0])

        obstacles = []
        for actor in self.world.get_actors().filter('*vehicle*'):
            if actor.id != self.vehicle.id:
                loc = actor.get_location()
                obstacles.append({'id': actor.id, 'x': loc.x, 'y': loc.y})

        return np.concatenate([img_features, ego_state, np.zeros(10)])

    def step(self, action):

        steer, throttle, brake = action

        t = self.vehicle.get_transform()
        ego_transform = {'x': t.location.x, 'y': t.location.y}
        v = self.vehicle.get_velocity()
        ego_speed = math.sqrt(v.x**2 + v.y**2)

        accel = (ego_speed - self.prev_speed) / self.dt
        jerk = (accel - self.prev_accel) / self.dt

        energy_used = abs(accel) * ego_speed * 1.5

        self.prev_speed = ego_speed
        self.prev_accel = accel

        traffic_light_state = 'Green'
        if self.vehicle.is_at_traffic_light():
            tl = self.vehicle.get_traffic_light()
            if tl.get_state() == carla.TrafficLightState.Red:
                traffic_light_state = 'Red'
            elif tl.get_state() == carla.TrafficLightState.Yellow:
                traffic_light_state = 'Yellow'

        obstacles = []
        for actor in self.world.get_actors().filter('*vehicle*'):
            if actor.id != self.vehicle.id:
                loc = actor.get_location()
                obstacles.append({'id': actor.id, 'x': loc.x, 'y': loc.y})

        for actor in self.actors_to_cleanup:
             loc = actor.get_location()
             obstacles.append({'id': actor.id, 'x': loc.x, 'y': loc.y})

        obs_seqs = {obs['id']: np.array([[obs['x'], obs['y']], [obs['x']+0.1, obs['y']+0.1]]) for obs in obstacles}
        predicted_obstacles = self.predictor.predict_all_obstacles(obs_seqs)

        if self.agent.emergency_brake_layer(ego_transform, ego_speed, obstacles):
            throttle, steer, brake = 0.0, 0.0, 1.0

        spectator = self.world.get_spectator()
        spectator.set_transform(carla.Transform(t.location + carla.Location(z=30), carla.Rotation(pitch=-90)))

        time.sleep(self.dt)

        reward = self.agent.compute_reward(
            ego_speed, speed_limit=14.0,
            distance_to_center=0.5,
            traffic_light_state=traffic_light_state,
            jerk=jerk,
            energy_used=energy_used
        )

        cost = self.agent.compute_constraint_cost(ego_transform, predicted_obstacles)
        self.agent.update_lagrangian(cost, cost_limit=0.0)

        state = self._get_obs()
        done = False
        info = {
            'cost': cost,
            'lambda': self.agent.lambda_multiplier,
            'jerk': jerk,
            'energy': energy_used,
            'light': traffic_light_state
        }

        return state, reward, done, info

    def close(self):
        cv2.destroyAllWindows()
        if self.camera: self.camera.destroy()
        if self.vehicle: self.vehicle.destroy()
        for actor in self.actors_to_cleanup:
            if actor is not None: actor.destroy()
        print("CARLA Environment Closed.")

if __name__ == '__main__':
    env = CarlaEnv()

    test_scenarios = [
        ("Clear_Day", False, 0),
        ("Clear_Night", False, 0),
        ("Rain_Day", False, 0),
        ("Clear_Day", True, 0),
        ("Clear_Day", False, 30)
    ]

    try:
        for scenario_name, has_obstacles, traffic_count in test_scenarios:
            print(f"\n{'='*40}")
            print(f"STARTING SCENARIO: {scenario_name} (Traffic: {traffic_count})")
            print(f"{'='*40}")

            state = env.reset(scenario=scenario_name, spawn_obstacles=has_obstacles, n_traffic=traffic_count)

            for i in range(300):
                action = [0.0, 0.5, 0.0]
                next_state, reward, done, info = env.step(action)

                if i % 50 == 0:
                    print(f"Step {i}: Reward={reward:.2f}, Cost={info['cost']}, Jerk={info['jerk']:.2f}, Light={info['light']}")

            print(f"Scenario {scenario_name} Finished.")
            time.sleep(2)
    except Exception as e:
        print(f"Error during simulation: {e}")
    finally:
        env.close()