#!/usr/bin/env python

import xml.etree.ElementTree as ET
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

def extract_obstacles_from_sdf(sdf_path):
    """Extract obstacle information from SDF file"""
    print(f"Parsing SDF file: {sdf_path}")
    
    obstacle_list = []
    
    # Supported angles (consistent with sdf_reader.py)
    x_angle = set([0.0, 0.5, -0.5, 0.3, -0.3, 3.1, -3.1])
    
    try:
        tree = ET.parse(sdf_path)
        root = tree.getroot()
        

        model = root.find(".//model")
        if model is None:
            print("Error: Model element not found in SDF file!")
            return obstacle_list
        for link in model.findall("link"):
            size_elem = link.find(".//box/size")
            if size_elem is None:
                continue
                
            dimension_str = size_elem.text
            parse_dim = dimension_str.split()
            if len(parse_dim) < 2:
                continue
                
            x_dim = float(parse_dim[0])
            y_dim = float(parse_dim[1])
            
            pose_elem = link.find("pose")
            if pose_elem is None:
                continue
                
            position_str = pose_elem.text
            parse_pos = position_str.split()
            if len(parse_pos) < 6:
                continue
                
            x_pos = float(parse_pos[0]) + 2  
            y_pos = float(parse_pos[1]) - 4  
            angle = float(parse_pos[5])
            angle = round(angle, 1)
            if x_dim > 100 and y_dim > 70:
                continue
            
            print(f"Found obstacle: pos=({x_pos}, {y_pos}), size=({x_dim}, {y_dim}), angle={angle}")
            
            # Add obstacle points to list based on orientation
            if angle in x_angle:
                for i in range(int((x_pos-(x_dim/2+1))), int((x_pos+(x_dim/2)))):
                    for j in range(int((y_pos-(y_dim/2+1))), int((y_pos+(y_dim/2+1)))):
                        obstacle_list.append((i, j))
            else:
                for j in range(int((y_pos-(x_dim/2+1))), int((y_pos+(x_dim/2+1)))):
                    for i in range(int((x_pos-(y_dim/2+1))), int((x_pos+(y_dim/2)))):
                        obstacle_list.append((i, j))
    
    except Exception as e:
        print(f"Error parsing SDF file: {e}")
    
    return obstacle_list

def save_obstacle_list(obstacle_list, output_path):
    with open(output_path, 'wb') as filehandle:
        pickle.dump(obstacle_list, filehandle)
    print(f"Saved {len(obstacle_list)} obstacle points to {output_path}")

def visualize_obstacles(obstacle_list, output_path=None):
    grid_size = 200  # Adjust as needed
    grid = np.zeros((grid_size, grid_size))
    
    for x, y in obstacle_list:
        if 0 <= x + 100 < grid_size and 0 <= y + 100 < grid_size:
            grid[y + 100, x + 100] = 1
    
    # Display grid
    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap='gray_r', origin='lower')
    plt.title(f'Obstacle Distribution ({len(obstacle_list)} points)')
    plt.colorbar(label='Obstacle')
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path)
        print(f"Obstacle visualization saved to {output_path}")
    else:
        plt.show()

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    sdf_path = os.path.join(project_dir, "models", "eceb_mp3", "model.sdf")
    output_data_path = os.path.join(script_dir, "obstacle_list.data")
    visualization_path = os.path.join(script_dir, "obstacle_visualization.png")
    
    obstacle_list = extract_obstacles_from_sdf(sdf_path)
    
    save_obstacle_list(obstacle_list, output_data_path)
    # Visualize obstacles
    visualize_obstacles(obstacle_list, visualization_path)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()