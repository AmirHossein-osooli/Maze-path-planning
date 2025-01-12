import cv2
import numpy as np
from a_star import AStarPlanner

import matplotlib
matplotlib.use('TkAgg')  # Set the backend to TkAgg (you may need to comment this line)

import matplotlib.pyplot as plt

show_animation = True

def image_to_obstacles(image_path, grid_resolution):
    # Load the maze image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize image to fit grid resolution
    scale = grid_resolution / 5.0  # Adjust scale factor as needed
    img_resized = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

    # Threshold the image to binary (black/white)
    _, binary_img = cv2.threshold(img_resized, 128, 255, cv2.THRESH_BINARY)

    # Get obstacle coordinates from binary image
    binary_img = 255 - binary_img  # Invert (black = obstacles)
    obstacle_indices = np.argwhere(binary_img > 0)

    # Convert obstacle indices to x, y coordinates
    ox, oy = [], []
    for coord in obstacle_indices:
        ox.append(coord[1])   # x-coordinate
        oy.append(-coord[0])  # y-coordinate (Using negative to correct the plot) 

    return ox, oy

def main():
    # start and goal positions
    sx, sy = 76.0, 0.0  # Start (x, y)
    gx, gy = 84.0, -160.0  # Goal (x, y)
    grid_size = 2.0  # Grid resolution
    robot_radius = 1.0  # Robot radius

    maze_image_path = "maze.png"
    ox, oy = image_to_obstacles(maze_image_path, grid_size)

    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")

    a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
    rx, ry = a_star.planning(sx, sy, gx, gy)

    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.pause(0.1)
        plt.show()


if __name__ == '__main__':
    main()
