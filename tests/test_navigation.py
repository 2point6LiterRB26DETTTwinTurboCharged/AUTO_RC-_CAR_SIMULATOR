import numpy as np

from app import calculate_navigation, segment_intersects_obstacle


def test_segment_intersects_box_obstacle():
    obstacles = [
        {"x_min": 4.0, "x_max": 6.0, "y_min": 4.0, "y_max": 6.0},
    ]

    assert segment_intersects_obstacle(3.8, 4.0, 4.2, 4.0, obstacles, vehicle_radius=0.0) is True
    assert segment_intersects_obstacle(3.0, 3.0, 3.4, 3.0, obstacles, vehicle_radius=0.0) is False


def test_final_waypoint_stops_navigation():
    speed, steering, wp_idx, status = calculate_navigation(
        8.4, 8.4, 0.0, [[8.5, 8.5]], 0, np.array([5.0]), np.array([0.0]), 0.5
    )

    assert wp_idx == 0
    assert speed == 0.0
    assert steering == 0.0
    assert status == "DESTINATION REACHED"
