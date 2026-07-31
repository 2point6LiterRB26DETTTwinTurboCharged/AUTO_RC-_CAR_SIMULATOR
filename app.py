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
    "waypoints": [[8.5, 8.5]],  # Target points [x, y]
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


# -------------------- HELPER & NAVIGATION FUNCTIONS --------------------
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


def segment_intersects_obstacle(x1, y1, x2, y2, current_obstacles, vehicle_radius=0.25):
    """Check whether the planned motion segment crosses an obstacle box."""
    if not current_obstacles:
        return False

    num_samples = max(4, int(np.hypot(x2 - x1, y2 - y1) / 0.05) + 1)
    for t in np.linspace(0.0, 1.0, num_samples):
        sample_x = x1 + (x2 - x1) * t
        sample_y = y1 + (y2 - y1) * t
        for obs in current_obstacles:
            if (
                obs["x_min"] - vehicle_radius <= sample_x <= obs["x_max"] + vehicle_radius
                and obs["y_min"] - vehicle_radius <= sample_y <= obs["y_max"] + vehicle_radius
            ):
                return True
    return False


def calculate_navigation(
    car_x,
    car_y,
    car_heading,
    waypoints,
    current_wp_idx,
    lidar_distances,
    relative_angles,  # Relative angles where 0 is straight ahead
    speed,
    safety_margin=1.2,
    max_range=5.0
):
    """
    Calculates speed and steering angle considering both waypoint navigation
    and active LiDAR obstacle avoidance.
    """
    if not waypoints or current_wp_idx >= len(waypoints):
        return 0.0, 0.0, current_wp_idx, "DESTINATION REACHED"

    target_x, target_y = waypoints[current_wp_idx]
    dx = target_x - car_x
    dy = target_y - car_y
    dist_to_target = float(np.hypot(dx, dy))

    # Adaptive arrival threshold based on speed to prevent orbiting loops
    arrival_threshold = max(0.45, speed * 0.4)

    while dist_to_target < arrival_threshold:
        if current_wp_idx < len(waypoints) - 1:
            current_wp_idx += 1
            target_x, target_y = waypoints[current_wp_idx]
            dx = target_x - car_x
            dy = target_y - car_y
            dist_to_target = float(np.hypot(dx, dy))
        else:
            return 0.0, 0.0, current_wp_idx, "DESTINATION REACHED"

    # Direction angle to target
    target_angle = np.arctan2(dy, dx)
    heading_diff = target_angle - car_heading
    heading_diff = float(np.arctan2(np.sin(heading_diff), np.cos(heading_diff)))

    # Proportional waypoint steering (reduced near targets to avoid overshooting)
    target_steering = 0.4 * heading_diff

    # ----------------------------------------------------
    # LIDAR OBSTACLE AVOIDANCE OVERRIDE
    # ----------------------------------------------------
    status = "NOMINAL - NAVIGATING"
    calc_speed = speed
    
    # Automatically slow down when approaching target to allow tight turns
    if dist_to_target < 1.5:
        calc_speed = speed * max(0.3, dist_to_target / 1.5)

    avoidance_steering = 0.0
    nearest_obstacle_dist = max_range
    obstacle_detected = False

    for rel_angle, dist in zip(relative_angles, lidar_distances):
        if not np.isfinite(dist) or dist >= max_range or dist >= safety_margin:
            continue

        obstacle_detected = True
        nearest_obstacle_dist = min(nearest_obstacle_dist, float(dist))
        
        # Turn away from obstacle direction
        turn_dir = -1.0 if rel_angle >= 0 else 1.0
        
        # Front sensors exert higher influence than side/rear sensors
        angle_factor = max(0.1, np.cos(rel_angle)) if abs(rel_angle) < (np.pi / 2) else 0.1
        influence = ((safety_margin - dist) / safety_margin) * angle_factor
        avoidance_steering += turn_dir * 1.2 * influence

    if obstacle_detected:
        status = "AVOIDING OBSTACLE"
        # Reduce speed proportionally to proximity
        speed_factor = max(0.1, nearest_obstacle_dist / safety_margin)
        calc_speed *= speed_factor

        # Prioritize obstacle avoidance when obstacle is close
        avoidance_weight = np.clip((safety_margin - nearest_obstacle_dist) / safety_margin, 0.0, 1.0)
        steering = (1.0 - avoidance_weight) * target_steering + avoidance_weight * avoidance_steering
    else:
        steering = target_steering

    # Clamp total steering angle per step
    steering = float(np.clip(steering, -0.6, 0.6))

    return calc_speed, steering, current_wp_idx, status


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

@app.route("/api/obstacles", methods=["GET", "POST"])
def update_obstacles():
    global obstacles
    if request.method == "POST":
        data = request.json or {}
        obstacles = data.get("obstacles", obstacles)
        return jsonify({"status": "success", "obstacles": obstacles})
    return jsonify({"obstacles": obstacles})

@app.route("/api/obstacles/<int:obs_id>", methods=["PUT", "DELETE"])
def item_obstacle(obs_id):
    global obstacles
    if request.method == "DELETE":
        obstacles = [o for o in obstacles if o["id"] != obs_id]
        return jsonify({"status": "deleted"})
    elif request.method == "PUT":
        data = request.json or {}
        for o in obstacles:
            if o["id"] == obs_id:
                o["x_min"] = data.get("x_min", o["x_min"])
                o["x_max"] = data.get("x_max", o["x_max"])
                o["y_min"] = data.get("y_min", o["y_min"])
                o["y_max"] = data.get("y_max", o["y_max"])
                break
        return jsonify({"status": "updated"})

@app.route("/api/step", methods=["POST"])
def step():
    data = request.json or {}
    move = data.get("move", True)  # Flag to determine whether vehicle moves
    speed = float(data.get("speed", 0.5))
    num_beams = int(data.get("num_beams", 120))
    max_range = float(data.get("max_range", 5.0))
    safety_margin = float(data.get("safety_margin", 1.2))
    sensor_mode = data.get("sensor_mode", "360_sweep")

    waypoints = data.get("waypoints", sim_state["waypoints"])
    sim_state["waypoints"] = waypoints

    if "current_waypoint_idx" in data:
        max_idx = max(0, len(sim_state["waypoints"]) - 1)
        sim_state["current_waypoint_idx"] = min(int(data["current_waypoint_idx"]), max_idx)

    room_size = 10.0

    if sensor_mode == "4_sensors":
        relative_angles = np.array([0.0, np.pi / 2, np.pi, -np.pi / 2])
    else:
        relative_angles = np.linspace(-np.pi, np.pi, num_beams, endpoint=False)

    lidar_distances = []
    all_rays = []

    for rel_angle in relative_angles:
        ray_world_angle = sim_state["car_heading"] + rel_angle
        dist, hit_x, hit_y = cast_lidar_ray(
            sim_state["car_x"], sim_state["car_y"], ray_world_angle, max_range, obstacles, room_size
        )
        lidar_distances.append(dist)
        all_rays.append({"angle": ray_world_angle, "dist": dist, "end_x": hit_x, "end_y": hit_y})

        if dist < max_range * 0.98:
            sim_state["map_points"].append({"x": round(hit_x, 2), "y": round(hit_y, 2)})

    if len(sim_state["map_points"]) > 1200:
        sim_state["map_points"] = sim_state["map_points"][-1200:]

    lidar_distances_arr = np.array(lidar_distances)

    if move and float(np.min(lidar_distances_arr)) < 0.25:
        sim_state["collisions"] += 1

    calc_speed, steering, updated_wp_idx, status_msg = calculate_navigation(
        sim_state["car_x"],
        sim_state["car_y"],
        sim_state["car_heading"],
        sim_state["waypoints"],
        sim_state["current_waypoint_idx"],
        lidar_distances_arr,
        relative_angles,
        speed,
        safety_margin,
        max_range,
    )

    # ONLY move the vehicle and update position state if move parameter is True
    if move:
        sim_state["current_waypoint_idx"] = updated_wp_idx
        sim_state["status"] = status_msg

        if status_msg != "DESTINATION REACHED":
            sim_state["car_heading"] += steering
            sim_state["car_heading"] = float(np.arctan2(np.sin(sim_state["car_heading"]), np.cos(sim_state["car_heading"])))

        next_x = sim_state["car_x"] + calc_speed * np.cos(sim_state["car_heading"]) * 0.2
        next_y = sim_state["car_y"] + calc_speed * np.sin(sim_state["car_heading"]) * 0.2

        vehicle_radius = 0.25
        blocked = segment_intersects_obstacle(
            sim_state["car_x"],
            sim_state["car_y"],
            next_x,
            next_y,
            obstacles,
            vehicle_radius,
        )

        if not blocked:
            sim_state["car_x"] = next_x
            sim_state["car_y"] = next_y
        else:
            sim_state["status"] = "BLOCKED BY OBSTACLE"

        sim_state["car_x"] = float(np.clip(sim_state["car_x"], 0.5, room_size - 0.5))
        sim_state["car_y"] = float(np.clip(sim_state["car_y"], 0.5, room_size - 0.5))

    return jsonify({
        "car_x": sim_state["car_x"],
        "car_y": sim_state["car_y"],
        "car_heading": sim_state["car_heading"],
        "waypoints": sim_state["waypoints"],
        "current_waypoint_idx": sim_state["current_waypoint_idx"],
        "all_rays": all_rays,
        "active_sensor_count": len(relative_angles),
        "map_points": sim_state["map_points"],
        "obstacles": obstacles,
        "status": sim_state["status"],
        "min_sensor_hit": float(np.min(lidar_distances_arr)),
        "collisions": sim_state["collisions"]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)