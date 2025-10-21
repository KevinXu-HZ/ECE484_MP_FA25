import xml.etree.ElementTree as ET
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
import os

model_path = "../models/eceb_mp3/model.sdf"

# Coordinate system transformation parameters
X_OFFSET = 2
Y_OFFSET = -4

# Define what angles are considered "aligned" with axes
# (in radians, with some tolerance)
ALIGNED_ANGLES = [0.0, 1.57, 3.14, 4.71] 
ANGLE_TOLERANCE = 0.1  # radians

def is_aligned_with_axes(angle):
    angle = angle % (2 * math.pi)
    
    for aligned_angle in ALIGNED_ANGLES:
        if abs(angle - aligned_angle) < ANGLE_TOLERANCE:
            return True
    
    return False

def rotation_matrix_2d(angle):
    return np.array([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)]
    ])

def calculate_rotated_rectangle_points(center_x, center_y, width, height, angle):
    half_width = width / 2.0
    half_height = height / 2.0
    corners = np.array([
        [-half_width, -half_height],
        [half_width, -half_height],
        [half_width, half_height],
        [-half_width, half_height]
    ])
    
    rot_matrix = rotation_matrix_2d(angle)
    rotated_corners = np.dot(corners, rot_matrix.T)
    
    rotated_corners[:, 0] += center_x
    rotated_corners[:, 1] += center_y
    
    return rotated_corners.tolist()

def polygon_to_grid(vertices, min_x, max_x, min_y, max_y, resolution=1.0):
    x_range = np.arange(min_x, max_x + resolution, resolution)
    y_range = np.arange(min_y, max_y + resolution, resolution)
    grid_points = []
    
    for x in x_range:
        for y in y_range:
            if point_in_polygon(x, y, vertices):
                grid_points.append((int(x), int(y)))
    
    return grid_points

def point_in_polygon(x, y, polygon):
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def process_aligned_obstacle(x_pos, y_pos, x_dim, y_dim, angle):
    points = []
    is_aligned_with_x = (angle % math.pi < ANGLE_TOLERANCE) or (abs(angle % math.pi - math.pi) < ANGLE_TOLERANCE)
    
    if is_aligned_with_x:
        for i in range(int((x_pos-(x_dim/2+1))), int((x_pos+(x_dim/2)))):
            for j in range(int((y_pos-(y_dim/2+1))), int((y_pos+(y_dim/2+1)))):
                points.append((i, j))
    else:

        for j in range(int((y_pos-(x_dim/2+1))), int((y_pos+(x_dim/2+1)))):
            for i in range(int((x_pos-(y_dim/2+1))), int((x_pos+(y_dim/2)))):
                points.append((i, j))
    
    print("  - Using original method: {} points".format(len(points)))
    return points

def process_rotated_obstacle(x_pos, y_pos, x_dim, y_dim, angle):
    corners = calculate_rotated_rectangle_points(x_pos, y_pos, x_dim, y_dim, angle)

    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    min_x, max_x = math.floor(min(xs)) - 1, math.ceil(max(xs)) + 1
    min_y, max_y = math.floor(min(ys)) - 1, math.ceil(max(ys)) + 1
    

    grid_cells = polygon_to_grid(corners, min_x, max_x, min_y, max_y)
    
    print("  - Using rotation method: {} points".format(len(grid_cells)))
    return grid_cells

def extract_obstacles_from_sdf():
    obstacle_list = []
    
    try:
        tree = ET.parse(model_path)
        root = tree.getroot()
        
        model = root.find(".//model")
        if model is None:
            print("Error: Model element not found in SDF file!")
            return obstacle_list
        
        for link in model.findall("link"):
            if link.get("name") == "main_structure":
                continue
                
            size_elem = link.find(".//box/size")
            if size_elem is None:
                continue
                
            dimension_str = size_elem.text
            parse_dim = dimension_str.split()
            if len(parse_dim) < 3:
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
                
            x_pos = float(parse_pos[0])+X_OFFSET
            y_pos = float(parse_pos[1])+Y_OFFSET
            z_pos = float(parse_pos[2])
            
            roll = float(parse_pos[3])
            pitch = float(parse_pos[4])
            yaw = float(parse_pos[5])
            
            if x_dim > 100 and y_dim > 70:
                continue
            
            print("Processing obstacle: pos=({}, {}), size=({}, {}), yaw={}".format(
                x_pos, y_pos, x_dim, y_dim, yaw))
            
            if is_aligned_with_axes(yaw):
                points = process_aligned_obstacle(x_pos, y_pos, x_dim, y_dim, yaw)
            else:
                points = process_rotated_obstacle(x_pos, y_pos, x_dim, y_dim, yaw)
            
            obstacle_list.extend(points)
    
    except Exception as e:
        print("Error parsing SDF file: {}".format(e))
        import traceback
        traceback.print_exc()
    
    # Remove duplicates
    obstacle_list = list(set(obstacle_list))
    return obstacle_list

def visualize_obstacles(obstacle_list, output_path="obstacle_visualization.png"):
    grid_size = (200, 200)  
    grid = np.zeros(grid_size)
    

    for x, y in obstacle_list:
        grid_x = int(x) + 100  # Center at 100
        grid_y = int(y) + 100  # Center at 100
        if 0 <= grid_x < grid_size[0] and 0 <= grid_y < grid_size[1]:
            grid[grid_y, grid_x] = 1
    

    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap='binary', origin='lower')
    plt.title('Obstacle Map ({} points)'.format(len(obstacle_list)))
    plt.colorbar(label='Obstacle')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    
def main():
    obstacle_list = extract_obstacles_from_sdf()
    
    with open('obstacle_list.data', 'wb') as filehandle:
        pickle.dump(obstacle_list, filehandle)
    print("Saved {} obstacle points to obstacle_list.data".format(len(obstacle_list)))
    visualize_obstacles(obstacle_list)
    
    # Create a backup of the old obstacle list
    if os.path.exists('obstacle_list.data.bak'):
        os.remove('obstacle_list.data.bak')
    if os.path.exists('obstacle_list.data'):
        import shutil
        shutil.copy('obstacle_list.data', 'obstacle_list.data.bak')
        print("Created backup of previous obstacle_list.data")

if __name__ == "__main__":
    main()