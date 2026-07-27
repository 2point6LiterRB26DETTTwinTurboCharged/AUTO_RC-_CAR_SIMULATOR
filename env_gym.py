# env_gym.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class RCCarEnv(gym.Env):
    """Gymnasium Environment wrapping the 2D/3D RC Car LiDAR Simulator."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, room_size=10.0, num_beams=120, max_range=5.0):
        super(RCCarEnv, self).__init__()

        self.room_size = room_size
        self.num_beams = num_beams
        self.max_range = max_range

        # Default obstacle list matching app.py
        self.default_obstacles = [
            {"id": 1, "x_min": 4.0, "x_max": 6.0, "y_min": 4.0, "y_max": 6.0},
            {"id": 2, "x_min": 1.0, "x_max": 3.0, "y_min": 7.0, "y_max": 8.5},
            {"id": 3, "x_min": 7.0, "x_max": 8.5, "y_min": 1.5, "y_max": 3.0},
        ]
        self.obstacles = list(self.default_obstacles)

        # Action Space: Continuous Steering [-0.25 rad, +0.25 rad] & Speed [0.1 m/s, 1.0 m/s]
        self.action_space = spaces.Box(
            low=np.array([-0.25, 0.1], dtype=np.float32),
            high=np.array([0.25, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation Space:
        # 1. Normalized LiDAR distance array (num_beams values between 0.0 and 1.0)
        # 2. Distance to target waypoint (normalized)
        # 3. Relative target angle relative to car heading [-pi, pi]
        obs_dim = self.num_beams + 2
        self.observation_space = spaces.Box(
            low=-np.pi, high=10.0, shape=(obs_dim,), dtype=np.float32
        )

        # Car Internal State
        self.car_x = 2.0
        self.car_y = 2.0
        self.car_heading = 0.0
        self.waypoints = [[8.5, 8.5]]
        self.current_wp_idx = 0
        self.step_count = 0
        self.max_steps = 500

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.car_x = 2.0
        self.car_y = 2.0
        self.car_heading = 0.0
        self.waypoints = [[8.5, 8.5]]
        self.current_wp_idx = 0
        self.step_count = 0

        # Allow passing custom obstacles dynamically
        if options and "obstacles" in options:
            self.obstacles = options["obstacles"]
        else:
            self.obstacles = list(self.default_obstacles)

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action):
        self.step_count += 1
        steering, speed = float(action[0]), float(action[1])

        # 1. Dynamics Update (Kinematic Bicycle / Unicycle Model)
        self.car_heading += steering
        # Keep heading normalized to [-pi, pi]
        self.car_heading = np.arctan2(np.sin(self.car_heading), np.cos(self.car_heading))

        self.car_x += speed * np.cos(self.car_heading) * 0.2
        self.car_y += speed * np.sin(self.car_heading) * 0.2

        # 2. Check Boundary & Obstacle Collisions
        lidar_distances = self._cast_lidar_rays()
        min_lidar_dist = np.min(lidar_distances)

        is_wall_collision = (
            self.car_x <= 0.5
            or self.car_x >= (self.room_size - 0.5)
            or self.car_y <= 0.5
            or self.car_y >= (self.room_size - 0.5)
        )
        is_obstacle_collision = min_lidar_dist < 0.25

        terminated = is_wall_collision or is_obstacle_collision
        truncated = self.step_count >= self.max_steps

        # 3. Waypoint Tracking & Distance
        target = self.waypoints[self.current_wp_idx]
        dx = target[0] - self.car_x
        dy = target[1] - self.car_y
        dist_to_target = np.hypot(dx, dy)

        # Check if waypoint is reached
        waypoint_reached = False
        if dist_to_target < 0.6:
            waypoint_reached = True
            if self.current_wp_idx < len(self.waypoints) - 1:
                self.current_wp_idx += 1
            else:
                terminated = True  # Reached final waypoint!

        # 4. Reward Engineering
        reward = 0.0
        if terminated:
            if waypoint_reached and self.current_wp_idx == len(self.waypoints) - 1:
                reward += 200.0  # Big reward for completing path
            else:
                reward -= 100.0  # Penalty for crashing
        else:
            # Reward for moving closer to target
            reward += (5.0 - dist_to_target) * 0.5
            # Small penalty for proximity to obstacles (safety margin)
            if min_lidar_dist < 0.8:
                reward -= (0.8 - min_lidar_dist) * 2.0
            # Small step penalty to encourage speed
            reward -= 0.1

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _cast_lidar_rays(self):
        """Simulates 360-degree Time-of-Flight LiDAR raycasting against obstacles."""
        angles = np.linspace(-np.PI if hasattr(np, 'PI') else -np.pi, np.pi, self.num_beams, endpoint=False)
        distances = np.full(self.num_beams, self.max_range, dtype=np.float32)

        for i, angle in enumerate(angles):
            ray_angle = self.car_heading + angle
            cos_a = np.cos(ray_angle)
            sin_a = np.sin(ray_angle)

            # Raycast against wall boundaries
            for step_d in np.arange(0.1, self.max_range, 0.1):
                rx = self.car_x + step_d * cos_a
                ry = self.car_y + step_d * sin_a

                # Wall Check
                if rx <= 0 or rx >= self.room_size or ry <= 0 or ry >= self.room_size:
                    distances[i] = step_d
                    break

                # Box Obstacle Check
                hit_obs = False
                for obs in self.obstacles:
                    if (obs["x_min"] <= rx <= obs["x_max"]) and (obs["y_min"] <= ry <= obs["y_max"]):
                        distances[i] = step_d
                        hit_obs = True
                        break
                if hit_obs:
                    break

        return distances

    def _get_observation(self):
        lidar_distances = self._cast_lidar_rays()
        norm_lidar = lidar_distances / self.max_range

        target = self.waypoints[self.current_wp_idx]
        dx = target[0] - self.car_x
        dy = target[1] - self.car_y
        dist_to_target = np.hypot(dx, dy)

        target_angle = np.arctan2(dy, dx)
        angle_diff = target_angle - self.car_heading
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

        return np.concatenate([norm_lidar, [dist_to_target, angle_diff]]).astype(np.float32)

    def _get_info(self):
        return {
            "car_x": self.car_x,
            "car_y": self.car_y,
            "car_heading": self.car_heading,
            "current_wp_idx": self.current_wp_idx,
            "step_count": self.step_count,
        }