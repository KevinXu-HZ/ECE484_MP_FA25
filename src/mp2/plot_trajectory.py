#!/usr/bin/env python3
"""
Problem 7: Vehicle Trajectory Visualization
Generates an x-y plot showing the vehicle trajectory, waypoints, and initial position.
"""

import matplotlib.pyplot as plt
import csv
import os
import sys
import numpy as np

# Add the src directory to path to import waypoint_list
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from waypoint_list import WayPoints

def load_trajectory_data(filename):
    """Load trajectory data from CSV file"""
    x_data = []
    y_data = []

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x_data.append(float(row['x']))
            y_data.append(float(row['y']))

    return np.array(x_data), np.array(y_data)

def plot_trajectory(x_data, y_data, waypoints, output_file='trajectory_plot.png'):
    """Generate x-y trajectory plot with waypoints and initial position"""

    plt.figure(figsize=(14, 10))

    # Plot vehicle trajectory
    plt.plot(x_data, y_data, 'b-', linewidth=2, label='Vehicle Trajectory', alpha=0.7)

    # Plot waypoints
    waypoint_array = np.array(waypoints)
    plt.scatter(waypoint_array[:, 0], waypoint_array[:, 1],
               c='orange', s=30, marker='o', alpha=0.5,
               label='Waypoints', zorder=3)

    # Mark initial position (waypoint 0)
    initial_x, initial_y = waypoints[0]
    plt.scatter([initial_x], [initial_y],
               c='green', s=200, marker='*',
               edgecolors='black', linewidths=1.5,
               label='Initial Position', zorder=5)

    # Mark final waypoint
    final_x, final_y = waypoints[-1]
    plt.scatter([final_x], [final_y],
               c='red', s=200, marker='s',
               edgecolors='black', linewidths=1.5,
               label='Final Waypoint', zorder=5)

    # Annotate some key waypoints
    annotation_indices = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, len(waypoints)-1]
    for idx in annotation_indices:
        if idx < len(waypoints):
            wp_x, wp_y = waypoints[idx]
            plt.annotate(f'WP{idx}',
                        xy=(wp_x, wp_y),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8,
                        alpha=0.6)

    plt.xlabel('X Position (m)', fontsize=12)
    plt.ylabel('Y Position (m)', fontsize=12)
    plt.title('Vehicle Trajectory with Waypoints', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    plt.axis('equal')
    plt.tight_layout()

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Trajectory plot saved to: {output_file}")

    return output_file

def analyze_trajectory(x_data, y_data, waypoints):
    """Analyze trajectory statistics"""

    # Calculate total distance traveled
    distances = np.sqrt(np.diff(x_data)**2 + np.diff(y_data)**2)
    total_distance = np.sum(distances)

    # Calculate waypoint path length
    waypoint_array = np.array(waypoints)
    waypoint_distances = np.sqrt(np.diff(waypoint_array[:, 0])**2 + np.diff(waypoint_array[:, 1])**2)
    waypoint_path_length = np.sum(waypoint_distances)

    # Calculate tracking error (distance from actual path to nearest waypoints)
    # Sample every 10th point to speed up computation
    sample_indices = range(0, len(x_data), 10)
    errors = []

    for i in sample_indices:
        point = np.array([x_data[i], y_data[i]])
        distances_to_waypoints = np.sqrt(np.sum((waypoint_array - point)**2, axis=1))
        min_distance = np.min(distances_to_waypoints)
        errors.append(min_distance)

    mean_error = np.mean(errors)
    max_error = np.max(errors)

    return {
        'total_distance': total_distance,
        'waypoint_path_length': waypoint_path_length,
        'mean_tracking_error': mean_error,
        'max_tracking_error': max_error,
        'num_trajectory_points': len(x_data),
        'num_waypoints': len(waypoints)
    }

def main():
    # Load trajectory data
    data_dir = os.path.expanduser("~/mp2_data")
    traj_file = os.path.join(data_dir, "trajectory_data.csv")

    if not os.path.exists(traj_file):
        print(f"Error: Data file not found at {traj_file}")
        print("Please run the controller first to generate data.")
        return

    print("Loading trajectory data...")
    x_data, y_data = load_trajectory_data(traj_file)

    # Load waypoints
    waypoint_obj = WayPoints()
    waypoints = waypoint_obj.getWayPoints()

    print("\n" + "="*60)
    print("TRAJECTORY ANALYSIS (Problem 7)")
    print("="*60)

    # Analyze trajectory
    analysis = analyze_trajectory(x_data, y_data, waypoints)

    print(f"\nTrajectory Statistics:")
    print(f"  Total distance traveled: {analysis['total_distance']:.2f} m")
    print(f"  Waypoint path length: {analysis['waypoint_path_length']:.2f} m")
    print(f"  Number of waypoints: {analysis['num_waypoints']}")
    print(f"  Trajectory data points: {analysis['num_trajectory_points']}")

    print(f"\nTracking Performance:")
    print(f"  Mean tracking error: {analysis['mean_tracking_error']:.2f} m")
    print(f"  Max tracking error: {analysis['max_tracking_error']:.2f} m")

    # Generate plot
    print("\n" + "-"*60)
    output_file = os.path.join(data_dir, "trajectory_plot.png")
    plot_trajectory(x_data, y_data, waypoints, output_file)

    print("\n" + "="*60)
    print("Trajectory visualization complete!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
