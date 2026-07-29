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

import numpy as np

def calculate_lidar_rays(car_x, car_y, car_heading, num_beams, max_range, obstacles, room_size=10.0):
    # 1. Generate all beam angles at once using NumPy
    if num_beams == 4:
        # Front, Right, Rear, Left
        relative_angles = np.array([0.0, np.pi/2, np.pi, -np.pi/2])
    else:
        relative_angles = np.linspace(0, 2 * np.pi, num_beams, endpoint=False)

    angles = car_heading + relative_angles
    
    # Unit direction vectors for all rays simultaneously
    dx = np.cos(angles)
    dy = np.sin(angles)

    distances = np.full(num_beams, max_range, dtype=np.float64)

    # 2. Vectorized Wall Intersections (Room boundaries)
    with np.errstate(divide='ignore'):
        t_right  = np.where(dx > 0, (room_size - car_x) / dx, np.inf)
        t_left   = np.where(dx < 0, (0.0 - car_x) / dx, np.inf)
        t_top    = np.where(dy > 0, (room_size - car_y) / dy, np.inf)
        t_bottom = np.where(dy < 0, (0.0 - car_y) / dy, np.inf)

    wall_distances = np.minimum(np.minimum(t_right, t_left), np.minimum(t_top, t_bottom))
    distances = np.minimum(distances, wall_distances)

    # 3. Vectorized Obstacle Raycast Box Intersections
    for obs in obstacles:
        x_min, x_max = obs['x_min'], obs['x_max']
        y_min, y_max = obs['y_min'], obs['y_max']

        with np.errstate(divide='ignore', invalid='ignore'):
            tx1 = np.where(dx != 0, (x_min - car_x) / dx, -np.inf)
            tx2 = np.where(dx != 0, (x_max - car_x) / dx, np.inf)
            ty1 = np.where(dy != 0, (y_min - car_y) / dy, -np.inf)
            ty2 = np.where(dy != 0, (y_max - car_y) / dy, np.inf)

            tmin = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))
            tmax = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))

        # Check valid box hits in front of the ray (tmin < tmax and tmin > 0)
        hit_mask = (tmax >= tmin) & (tmin > 0) & (tmin < distances)
        distances[hit_mask] = tmin[hit_mask]

    # Calculate end points for all rays
    end_x = car_x + distances * dx
    end_y = car_y + distances * dy

    all_rays = [
        {"angle": float(a), "dist": float(d), "end_x": float(ex), "end_y": float(ey)}
        for a, d, ex, ey in zip(angles, distances, end_x, end_y)
    ]

    return distances, angles, all_rays
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