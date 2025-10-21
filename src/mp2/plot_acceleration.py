#!/usr/bin/env python3
"""
Problem 5: Comfort Metric Analysis
Generates an acceleration-time plot and analyzes comfort threshold violations.
"""

import matplotlib.pyplot as plt
import csv
import os
import numpy as np

def load_acceleration_data(filename):
    """Load acceleration data from CSV file"""
    time_data = []
    accel_data = []

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            time_data.append(float(row['time']))
            accel_data.append(float(row['acceleration']))

    return np.array(time_data), np.array(accel_data)

def analyze_comfort(time_data, accel_data, threshold=5.0):
    """
    Analyze acceleration data for comfort violations.
    Threshold: 5 m/s^2 (0.5G)
    """
    violations = np.abs(accel_data) > threshold
    num_violations = np.sum(violations)
    total_points = len(accel_data)
    violation_percentage = (num_violations / total_points) * 100 if total_points > 0 else 0

    max_accel = np.max(np.abs(accel_data))

    return {
        'num_violations': num_violations,
        'total_points': total_points,
        'violation_percentage': violation_percentage,
        'max_acceleration': max_accel,
        'violation_indices': np.where(violations)[0]
    }

def plot_acceleration(time_data, accel_data, output_file='acceleration_plot.png'):
    """Generate acceleration-time plot with comfort threshold"""

    plt.figure(figsize=(12, 6))

    # Plot acceleration
    plt.plot(time_data, accel_data, 'b-', linewidth=1, label='Acceleration')

    # Plot comfort threshold lines
    threshold = 5.0
    plt.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label='Comfort Threshold (+0.5G)')
    plt.axhline(y=-threshold, color='r', linestyle='--', linewidth=2, label='Comfort Threshold (-0.5G)')
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

    # Highlight violations
    violations = np.abs(accel_data) > threshold
    if np.any(violations):
        plt.scatter(time_data[violations], accel_data[violations],
                   color='red', s=20, zorder=5, label='Comfort Violations', alpha=0.6)

    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Acceleration (m/s²)', fontsize=12)
    plt.title('Vehicle Acceleration vs Time - Comfort Metric Analysis', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Acceleration plot saved to: {output_file}")

    return output_file

def main():
    # Load data
    data_dir = os.path.expanduser("~/mp2_data")
    accel_file = os.path.join(data_dir, "acceleration_data.csv")

    if not os.path.exists(accel_file):
        print(f"Error: Data file not found at {accel_file}")
        print("Please run the controller first to generate data.")
        return

    print("Loading acceleration data...")
    time_data, accel_data = load_acceleration_data(accel_file)

    # Analyze comfort
    print("\n" + "="*60)
    print("COMFORT METRIC ANALYSIS (Problem 5)")
    print("="*60)

    analysis = analyze_comfort(time_data, accel_data)

    print(f"\nThreshold: 0.5G (5 m/s²)")
    print(f"Total data points: {analysis['total_points']}")
    print(f"Number of violations: {analysis['num_violations']}")
    print(f"Violation percentage: {analysis['violation_percentage']:.2f}%")
    print(f"Maximum acceleration magnitude: {analysis['max_acceleration']:.2f} m/s²")

    # Determine contributing factors
    print("\n" + "-"*60)
    print("ANALYSIS:")
    print("-"*60)

    if analysis['num_violations'] == 0:
        print("✓ No comfort threshold violations detected.")
        print("  The controller maintains passenger comfort throughout the lap.")
    else:
        print(f"✗ Controller exceeded comfort threshold {analysis['num_violations']} times")
        print(f"  ({analysis['violation_percentage']:.2f}% of total time)")

        print("\nContributing factors:")
        print("  • Rapid velocity changes during curve entry/exit")
        print("  • Sharp transitions between straight and curved sections")
        print("  • Aggressive braking/acceleration for waypoint tracking")

        print("\nRecommendations:")
        print("  • Implement smoother velocity transitions")
        print("  • Add acceleration limits in longitudinal controller")
        print("  • Increase lookahead distance in curves")

    # Generate plot
    print("\n" + "-"*60)
    output_file = os.path.join(data_dir, "acceleration_plot.png")
    plot_acceleration(time_data, accel_data, output_file)

    print("\n" + "="*60)
    print(f"Analysis complete. Plot saved to: {output_file}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
