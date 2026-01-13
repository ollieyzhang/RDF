"""
Visualize the sphere collision model of the Franka robot arm.
This script reads the franka_sphere.yaml file and displays the sphere approximation.
"""

import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
import yaml
import argparse
from panda_layer.panda_layer import PandaLayer

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


def load_sphere_model(yaml_path):
    """Load sphere collision model from YAML file."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['collision_spheres']


def create_sphere_mesh(center, radius, color=None):
    """Create a trimesh sphere at given center with given radius."""
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    sphere.apply_translation(center)
    if color is not None:
        sphere.visual.face_colors = color
    return sphere


def visualize_sphere_model_3d(sphere_data, pose, theta, robot, device, save_path=None):
    """
    Visualize the sphere collision model in 3D along with the actual robot mesh.
    
    Args:
        sphere_data: Dictionary of collision spheres per link
        pose: robot base pose (B, 4, 4)
        theta: joint angles (B, 7)
        robot: PandaLayer instance
        device: torch device
        save_path: path to save visualization
    """
    # Get robot transformations for each link
    transforms = robot.get_transformations_each_link(pose, theta)
    
    # Create scene
    scene = trimesh.Scene()
    
    # Get actual robot mesh for comparison
    robot_mesh = robot.get_forward_robot_mesh(pose, theta)[0]
    robot_mesh = np.sum(robot_mesh) if isinstance(robot_mesh, list) else robot_mesh
    robot_mesh.visual.face_colors = [200, 200, 200, 100]  # Semi-transparent gray
    scene.add_geometry(robot_mesh)
    
    # Link name to index mapping
    link_names = ['panda_link0', 'panda_link1', 'panda_link2', 'panda_link3',
                  'panda_link4', 'panda_link5', 'panda_link6', 'panda_link7', 
                  'panda_hand']
    
    # Color map for different links
    colors = [
        [255, 0, 0, 150],      # Red - link0
        [255, 127, 0, 150],    # Orange - link1
        [255, 255, 0, 150],    # Yellow - link2
        [0, 255, 0, 150],      # Green - link3
        [0, 127, 255, 150],    # Light Blue - link4
        [0, 0, 255, 150],      # Blue - link5
        [127, 0, 255, 150],    # Purple - link6
        [255, 0, 255, 150],    # Magenta - link7
        [255, 127, 127, 150],  # Pink - hand
    ]
    
    # Add spheres for each link
    for link_idx, link_name in enumerate(link_names):
        if link_name not in sphere_data:
            continue
            
        spheres = sphere_data[link_name]
        transform = transforms[link_idx].squeeze().cpu().numpy()
        
        for sphere_info in spheres:
            center = np.array(sphere_info['center'])
            radius = sphere_info['radius']
            
            # Transform center to world frame
            center_homo = np.append(center, 1.0)
            center_world = (transform @ center_homo)[:3]
            
            # Create sphere mesh
            sphere_mesh = create_sphere_mesh(center_world, radius, colors[link_idx])
            scene.add_geometry(sphere_mesh)
    
    if save_path:
        scene.export(save_path)
        print(f"Saved 3D scene to {save_path}")
    
    scene.show()


def visualize_sphere_model_2d(sphere_data, pose, theta, robot, device, 
                              z_height=0.1, save_path=None):
    """
    Visualize 2D cross-section of sphere collision model at specified z-height.
    
    Args:
        sphere_data: Dictionary of collision spheres per link
        pose: robot base pose
        theta: joint angles
        robot: PandaLayer instance
        device: torch device
        z_height: height of the observation plane
        save_path: path to save figure
    """
    # Get robot transformations for each link
    transforms = robot.get_transformations_each_link(pose, theta)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get actual robot mesh for comparison
    robot_mesh = robot.get_forward_robot_mesh(pose, theta)[0]
    robot_mesh = np.sum(robot_mesh) if isinstance(robot_mesh, list) else robot_mesh
    vertices = robot_mesh.vertices
    
    # Plot robot projection
    vertices_2d = vertices[:, :2]
    z_values = vertices[:, 2]
    scatter = ax.scatter(vertices_2d[:, 0], vertices_2d[:, 1],
                        c=z_values, s=1, alpha=0.3, cmap='gray')
    
    # Link name to index mapping
    link_names = ['panda_link0', 'panda_link1', 'panda_link2', 'panda_link3',
                  'panda_link4', 'panda_link5', 'panda_link6', 'panda_link7', 
                  'panda_hand']
    
    # Color map for different links
    colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'magenta', 'pink']
    
    # Plot circles for spheres that intersect the z-plane
    for link_idx, link_name in enumerate(link_names):
        if link_name not in sphere_data:
            continue
            
        spheres = sphere_data[link_name]
        transform = transforms[link_idx].squeeze().cpu().numpy()
        
        for sphere_info in spheres:
            center = np.array(sphere_info['center'])
            radius = sphere_info['radius']
            
            # Transform center to world frame
            center_homo = np.append(center, 1.0)
            center_world = (transform @ center_homo)[:3]
            
            # Check if sphere intersects the z-plane
            z_dist = abs(center_world[2] - z_height)
            if z_dist < radius:
                # Calculate circle radius at z-plane
                circle_radius = np.sqrt(radius**2 - z_dist**2)
                
                # Draw circle
                circle = plt.Circle((center_world[0], center_world[1]), 
                                  circle_radius, 
                                  color=colors[link_idx], 
                                  fill=False, 
                                  linewidth=2,
                                  alpha=0.7,
                                  label=link_name if sphere_info == spheres[0] else "")
                ax.add_patch(circle)
    
    ax.set_xlim(-0.3, 0.7)
    ax.set_ylim(-0.5, 0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)', fontsize=13)
    ax.set_ylabel('Y (m)', fontsize=13)
    ax.set_title(f'Sphere Collision Model - Cross Section at z={z_height:.3f}m', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend (filter duplicates)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved 2D visualization to {save_path}")
    
    plt.show()


def visualize_sphere_statistics(sphere_data):
    """Print statistics about the sphere collision model."""
    print("\n" + "="*60)
    print("Sphere Collision Model Statistics")
    print("="*60)
    
    total_spheres = 0
    for link_name, spheres in sphere_data.items():
        n_spheres = len(spheres)
        total_spheres += n_spheres
        
        radii = [s['radius'] for s in spheres]
        avg_radius = np.mean(radii)
        min_radius = np.min(radii)
        max_radius = np.max(radii)
        
        print(f"\n{link_name}:")
        print(f"  Number of spheres: {n_spheres}")
        print(f"  Radius range: [{min_radius:.4f}, {max_radius:.4f}]m")
        print(f"  Average radius: {avg_radius:.4f}m")
    
    print(f"\nTotal spheres: {total_spheres}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize Franka sphere collision model')
    parser.add_argument('--device', default='cpu', type=str, help='Device (cuda/cpu)')
    parser.add_argument('--yaml_path', default=None, type=str, 
                       help='Path to sphere YAML file (default: franka_sphere.yaml)')
    parser.add_argument('--z_height', default=0.1, type=float, 
                       help='Height for 2D cross-section visualization')
    parser.add_argument('--show_3d', action='store_true', 
                       help='Show 3D visualization with trimesh')
    parser.add_argument('--show_2d', action='store_true', 
                       help='Show 2D cross-section visualization')
    parser.add_argument('--save_dir', default='figs', type=str, 
                       help='Directory to save figures')
    args = parser.parse_args()
    
    # Default to showing both if neither specified
    if not args.show_3d and not args.show_2d:
        args.show_2d = True
        args.show_3d = True
    
    # Setup paths
    if args.yaml_path is None:
        args.yaml_path = os.path.join(CUR_DIR, 'panda_layer', 'franka_sphere.yaml')
    
    # Load sphere model
    print(f"Loading sphere model from {args.yaml_path}...")
    sphere_data = load_sphere_model(args.yaml_path)
    
    # Print statistics
    visualize_sphere_statistics(sphere_data)
    
    # Setup device
    device = torch.device(args.device)
    
    # Initialize robot
    print("Initializing robot...")
    panda = PandaLayer(device)
    
    # Set robot configuration (same as vis_2D_planar.py)
    theta = torch.tensor([np.pi/2, np.pi/2, -np.pi/2, 0, 0, 0, np.pi/4]).float().to(device).reshape(-1, 7)
    pose = torch.eye(4).unsqueeze(0).to(device).float()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Show 2D visualization
    if args.show_2d:
        print(f"\nGenerating 2D cross-section at z={args.z_height}m...")
        save_path_2d = os.path.join(args.save_dir, f'sphere_model_2d_z{args.z_height:.3f}.png')
        visualize_sphere_model_2d(sphere_data, pose, theta, panda, device, 
                                 z_height=args.z_height, save_path=save_path_2d)
    
    # Show 3D visualization
    if args.show_3d:
        print("\nGenerating 3D visualization...")
        save_path_3d = os.path.join(args.save_dir, 'sphere_model_3d.obj')
        visualize_sphere_model_3d(sphere_data, pose, theta, panda, device, 
                                 save_path=save_path_3d)
    
    print("\nVisualization complete!")


if __name__ == '__main__':
    main()
