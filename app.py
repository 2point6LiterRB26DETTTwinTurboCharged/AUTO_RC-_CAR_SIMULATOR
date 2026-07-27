import os
import webbrowser
import numpy as np
from flask import Flask, jsonify, request, send_file
from stable_baselines3 import PPO

app = Flask(__name__, static_folder=".", template_folder=".")

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
    if os.path.exists("ppo_rccar_model.zip") or os.path.exists("ppo_rccar_model"):
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
    """Calculates smoothed steering towards active waypoint with reactive obstacle avoidance."""
    if not waypoints or current_idx >= len(waypoints):
        sim_state["status"] = "🎉 ALL WAYPOINTS REACHED!"
        return 0.0, 0.0, current_idx

    target = waypoints[current_idx]
    dx = target[0] - car_x
    dy = target[1] - car_y
    dist_to_target = np.hypot(dx, dy)

    # Check if close enough to waypoint to advance
    if dist_to_target < 0.6:
        current_idx += 1
        if current_idx >= len(waypoints):
            sim_state["status"] = "🎉 ALL WAYPOINTS REACHED!"
            return 0.0, 0.0, current_idx
        target = waypoints[current_idx]
        dx = target[0] - car_x
        dy = target[1] - car_y

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
    global sim_state, obstacles

    data = request.json or {}
    speed = float(data.get("speed", 0.5))
    num_beams = int(data.get("num_beams", 120))
    max_range = float(data.get("max_range", 5.0))
    safety_margin = float(data.get("safety_margin", 1.2))
    sensor_mode = data.get("sensor_mode", "360_sweep")
    room_size = float(data.get("room_size", 10.0))

    if "waypoints" in data:
        sim_state["waypoints"] = data["waypoints"]
    
    if "current_waypoint_idx" in data:
        sim_state["current_waypoint_idx"] = data["current_waypoint_idx"]

    if sensor_mode == "4_sensors":
        angles = np.array([0.0, np.pi / 4, -np.pi / 4, np.pi])
    else:
        angles = np.linspace(-np.pi, np.pi, num_beams)

    lidar_distances = []
    all_rays = []

    for a in angles:
        global_angle = sim_state["car_heading"] + a
        dist, end_x, end_y = cast_lidar_ray(
            sim_state["car_x"], sim_state["car_y"], global_angle, max_range, obstacles, room_size
        )
        lidar_distances.append(dist)
        all_rays.append({"end_x": float(end_x), "end_y": float(end_y)})
        
        if dist < max_range:
            sim_state["map_points"].append({"x": float(end_x), "y": float(end_y)})

    # Calculate next step motion
    move_speed, steering_angle, next_wp_idx = calculate_navigation(
        sim_state["car_x"], sim_state["car_y"], sim_state["car_heading"],
        sim_state["waypoints"], sim_state["current_waypoint_idx"],
        np.array(lidar_distances), angles, speed, safety_margin, max_range
    )

    sim_state["current_waypoint_idx"] = next_wp_idx
    sim_state["car_heading"] += steering_angle
    sim_state["car_x"] += move_speed * np.cos(sim_state["car_heading"]) * 0.2
    sim_state["car_y"] += move_speed * np.sin(sim_state["car_heading"]) * 0.2

    # Collision check with boundary walls
    if sim_state["car_x"] <= 0.5 or sim_state["car_x"] >= (room_size - 0.5) or \
       sim_state["car_y"] <= 0.5 or sim_state["car_y"] >= (room_size - 0.5):
        sim_state["collisions"] += 1

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
        "min_sensor_hit": float(np.min(lidar_distances)) if len(lidar_distances) > 0 else 0.0,
        "collisions": sim_state["collisions"]
    })

if __name__ == "__main__":
    port = 5000
    url = f"http://127.0.0.1:{port}"

    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open(url)

    app.run(host="0.0.0.0", port=port, debug=True)