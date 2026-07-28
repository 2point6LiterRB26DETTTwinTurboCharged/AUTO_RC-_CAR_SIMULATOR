import os
import numpy as np
from flask import Flask, jsonify, request, send_file

try:
    from stable_baselines3 import PPO
except Exception:  # pragma: no cover - optional dependency fallback
    PPO = None

app = Flask(__name__, static_folder=".", template_folder=".")
PORT = int(os.environ.get("PORT", "5000"))

# -------------------- INITIALIZE STATE --------------------
DEFAULT_STATE = {
    "car_x": 2.0,
    "car_y": 2.0,
    "car_heading": 0.0,  # in radians
    "waypoints": [[8.5, 8.5]], # Target points [x, y]
    "current_waypoint_idx": 0,
    "map_points": [],
    "status": "NOMINAL - NAVIGATING",
    "collisions": 0,
}

sim_state = dict(DEFAULT_STATE)

# Dynamic customizable obstacles list
obstacles = [
    {"id": 1, "x_min": 4.0, "x_max": 6.0, "y_min": 4.0, "y_max": 6.0},
    {"id": 2, "x_min": 1.0, "x_max": 3.0, "y_min": 7.0, "y_max": 8.5},
    {"id": 3, "x_min": 7.0, "x_max": 8.5, "y_min": 1.5, "y_max": 3.0},
]

# Load RL model if present
rl_model = None
try:
    if PPO is not None and (os.path.exists("ppo_rccar_model.zip") or os.path.exists("ppo_rccar_model")):
        rl_model = PPO.load("ppo_rccar_model")
        print("🤖 Trained RL Model successfully loaded!")
except Exception as e:
    print(f"ℹ️ RL Model load skipped/failed ({e}). Using algorithmic navigation.")

# -------------------- HELPER FUNCTIONS --------------------
def cast_lidar_ray(x, y, angle, max_r, current_obstacles, room_size):
    """Simulate a single LiDAR ray cast in a given angle."""
    step = 0.05
    r = 0.0
    while r < max_r:
        check_x = x + r * np.cos(angle)
        check_y = y + r * np.sin(angle)

        if check_x <= 0 or check_x >= room_size or check_y <= 0 or check_y >= room_size:
            return r, check_x, check_y

        for obs in current_obstacles:
            if (obs["x_min"] <= check_x <= obs["x_max"]) and (
                obs["y_min"] <= check_y <= obs["y_max"]
            ):
                return r, check_x, check_y

        r += step
    
    end_x = x + max_r * np.cos(angle)
    end_y = y + max_r * np.sin(angle)
    return max_r, end_x, end_y

def calculate_navigation(car_x, car_y, car_heading, waypoints, current_idx, lidar_distances, angles, speed, safety_margin, max_range):
    """Calculates smoothed steering towards active waypoint with dynamic skipping for close waypoints."""
    if not waypoints or current_idx >= len(waypoints):
        sim_state["status"] = "🎉 ALL WAYPOINTS REACHED!"
        return 0.0, 0.0, current_idx

    target = waypoints[current_idx]
    dx = target[0] - car_x
    dy = target[1] - car_y
    dist_to_target = np.hypot(dx, dy)

    # Dynamic acceptance radius based on speed
    acceptance_radius = max(0.25, speed * 0.4)

    # Skip waypoints that are already too close or passed to avoid getting stuck in loops
    while dist_to_target < acceptance_radius and current_idx < len(waypoints) - 1:
        current_idx += 1
        target = waypoints[current_idx]
        dx = target[0] - car_x
        dy = target[1] - car_y
        dist_to_target = np.hypot(dx, dy)

    if current_idx >= len(waypoints) - 1 and dist_to_target < acceptance_radius:
        sim_state["status"] = "🎉 ALL WAYPOINTS REACHED!"
        return 0.0, 0.0, current_idx

    target_angle = np.arctan2(dy, dx)
    heading_diff = target_angle - car_heading
    heading_diff = np.arctan2(np.sin(heading_diff), np.cos(heading_diff))

    # Calculate proportional steering angle towards target
    desired_steering = float(np.clip(0.35 * heading_diff, -0.35, 0.35))

    # Obstacle avoidance override
    front_indices = np.where((angles >= -np.pi / 4) & (angles <= np.pi / 4))[0]
    min_front = np.min(lidar_distances[front_indices]) if len(front_indices) > 0 else max_range

    if min_front < safety_margin:
        sim_state["status"] = f"⚠️ OBSTACLE AHEAD - DODGING (Target #{current_idx + 1})"
        left_indices = np.where((angles > np.pi / 6) & (angles <= np.pi / 2))[0]
        right_indices = np.where((angles < -np.pi / 6) & (angles >= -np.pi / 2))[0]
        mean_left = np.mean(lidar_distances[left_indices]) if len(left_indices) > 0 else max_range
        mean_right = np.mean(lidar_distances[right_indices]) if len(right_indices) > 0 else max_range
        
        avoid_steering = -0.4 if mean_left < mean_right else 0.4
        return speed * 0.4, avoid_steering, current_idx

    sim_state["status"] = f"🟢 ROUTE ACTIVE - WAYPOINT {current_idx + 1}/{len(waypoints)}"
    return speed, desired_steering, current_idx

# -------------------- ROUTES --------------------
@app.route("/")
def landing():
    return send_file("landing.html")

@app.route("/health")
def health_check():
    return jsonify({"status": "ok"})

@app.route("/2d")
def main_app():
    return send_file("index.html")

@app.route("/3d")
def page_3d():
    return send_file("view3d.html")

@app.route("/api/reset", methods=["POST"])
def reset():
    global sim_state
    sim_state["car_x"] = 2.0
    sim_state["car_y"] = 2.0
    sim_state["car_heading"] = 0.0
    sim_state["current_waypoint_idx"] = 0
    sim_state["map_points"] = []
    sim_state["collisions"] = 0
    sim_state["status"] = "NOMINAL - RESET COMPLETE"
    return jsonify({"status": "reset_success"})

@app.route("/api/obstacles", methods=["POST"])
def update_obstacles():
    global obstacles
    data = request.json or {}
    obstacles = data.get("obstacles", obstacles)
    return jsonify({"status": "success", "obstacles": obstacles})

@app.route("/api/step", methods=["POST"])
def step():
    data = request.json or {}
    speed = float(data.get("speed", 0.5))
    num_beams = int(data.get("num_beams", 120))
    max_range = float(data.get("max_range", 5.0))
    safety_margin = float(data.get("safety_margin", 1.2))
    sensor_mode = data.get("sensor_mode", "360_sweep")
    waypoints = data.get("waypoints", sim_state["waypoints"])
    sim_state["waypoints"] = waypoints
    sim_state["current_waypoint_idx"] = int(data.get("current_waypoint_idx", sim_state["current_waypoint_idx"]))

    room_size = 10.0

    # Sensor Angle Configurations
    if sensor_mode == "4_sensors":
        angles = np.array([0.0, np.pi / 2, np.pi, -np.pi / 2])
    else:
        angles = np.linspace(-np.pi, np.pi, num_beams, endpoint=False)

    lidar_distances = []
    all_rays = []

    for angle in angles:
        ray_angle = sim_state["car_heading"] + angle
        dist, hit_x, hit_y = cast_lidar_ray(
            sim_state["car_x"], sim_state["car_y"], ray_angle, max_range, obstacles, room_size
        )
        lidar_distances.append(dist)
        all_rays.append({"angle": ray_angle, "dist": dist, "end_x": hit_x, "end_y": hit_y})

        if dist < max_range * 0.98:
            sim_state["map_points"].append({"x": round(hit_x, 2), "y": round(hit_y, 2)})

    if len(sim_state["map_points"]) > 1200:
        sim_state["map_points"] = sim_state["map_points"][-1200:]

    lidar_distances = np.array(lidar_distances)

    calc_speed, steering, updated_wp_idx = calculate_navigation(
        sim_state["car_x"],
        sim_state["car_y"],
        sim_state["car_heading"],
        sim_state["waypoints"],
        sim_state["current_waypoint_idx"],
        lidar_distances,
        angles,
        speed,
        safety_margin,
        max_range,
    )

    sim_state["current_waypoint_idx"] = updated_wp_idx
    sim_state["car_heading"] += steering
    sim_state["car_x"] += calc_speed * np.cos(sim_state["car_heading"]) * 0.2
    sim_state["car_y"] += calc_speed * np.sin(sim_state["car_heading"]) * 0.2

    sim_state["car_x"] = float(np.clip(sim_state["car_x"], 0.5, room_size - 0.5))
    sim_state["car_y"] = float(np.clip(sim_state["car_y"], 0.5, room_size - 0.5))

    return jsonify({
        "car_x": sim_state["car_x"],
        "car_y": sim_state["car_y"],
        "car_heading": sim_state["car_heading"],
        "waypoints": sim_state["waypoints"],
        "current_waypoint_idx": sim_state["current_waypoint_idx"],
        "all_rays": all_rays,
        "active_sensor_count": len(angles),
        "map_points": sim_state["map_points"],
        "obstacles": obstacles,
        "status": sim_state["status"],
        "min_sensor_hit": float(np.min(lidar_distances)),
        "collisions": sim_state["collisions"]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)