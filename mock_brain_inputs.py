"""
Mock Brain Inputs Script for Unitree Go2 Brain Logic
Generates synthetic odometry and navigation goals to test Mission Control autonomy modules offline.
"""

import time
import json

def run_mock_brain():
    print("[MOCK BRAIN] Starting synthetic autonomy & Nav2 input simulator...")
    goal_id = 0
    try:
        while True:
            goal_id += 1
            nav_goal = {
                "goal_id": f"goal_{goal_id}",
                "target_pose": {"x": 2.5 + goal_id * 0.1, "y": 1.0, "theta": 0.0},
                "status": "ACCEPTED"
            }
            print(f"[MOCK BRAIN] Dispatched Nav2 Goal #{goal_id}: {json.dumps(nav_goal)}")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[MOCK BRAIN] Stopped.")

if __name__ == "__main__":
    run_mock_brain()
