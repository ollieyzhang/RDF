"""
Visualization for 2D planar manipulation tasks using 3D SDF models.

This script visualizes:
1. 3D robot arm in specific configuration
2. 2D top-down view at specific z-height
3. Distance field from object boundary points to robot surface
4. Projected 2D distances from 3D SDF gradients
"""

import torch
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid GUI thread issues on macOS
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
import trimesh
import argparse
from panda_layer.panda_layer import PandaLayer
import bf_sdf
import utils

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


class PlanarManipulationVisualizer:
    """Visualize robot SDF for 2D planar manipulation."""
    
    def __init__(self, bp_sdf, model, robot, device):
        self.bp_sdf = bp_sdf
        self.model = model
        self.robot = robot
        self.device = device
        
    def get_object_boundary_points_2d(self, obj_type='circle', obj_params=None, z_height=0.0, n_points=32):
        """
        Generate boundary points of an object at specific z-height.
        
        Args:
            obj_type: 'circle', 'rectangle', or 'polygon'
            obj_params: parameters for the object
                - circle: {'center': [x, y], 'radius': r}
                - rectangle: {'center': [x, y], 'width': w, 'height': h, 'angle': a}
                - polygon: {'vertices': [[x1,y1], [x2,y2], ...]}
            z_height: height of the plane (z coordinate)
            n_points: number of boundary points to sample
            
        Returns:
            boundary_points: (n_points, 3) array with [x, y, z_height]
        """
        if obj_params is None:
            obj_params = {}
            
        if obj_type == 'circle':
            center = obj_params.get('center', [0.3, 0.0])
            radius = obj_params.get('radius', 0.05)
            theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            z = np.ones_like(x) * z_height
            
        elif obj_type == 'rectangle':
            center = obj_params.get('center', [0.3, 0.0])
            width = obj_params.get('width', 0.1)
            height = obj_params.get('height', 0.08)
            angle = obj_params.get('angle', 0.0)
            
            # Create rectangle boundary
            n_per_side = n_points // 4
            # Bottom edge
            x1 = np.linspace(-width/2, width/2, n_per_side, endpoint=False)
            y1 = np.ones_like(x1) * (-height/2)
            # Right edge
            x2 = np.ones(n_per_side) * width/2
            y2 = np.linspace(-height/2, height/2, n_per_side, endpoint=False)
            # Top edge
            x3 = np.linspace(width/2, -width/2, n_per_side, endpoint=False)
            y3 = np.ones_like(x3) * height/2
            # Left edge
            x4 = np.ones(n_points - 3*n_per_side) * (-width/2)
            y4 = np.linspace(height/2, -height/2, n_points - 3*n_per_side, endpoint=False)
            
            x = np.concatenate([x1, x2, x3, x4])
            y = np.concatenate([y1, y2, y3, y4])
            
            # Rotate
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            x_rot = cos_a * x - sin_a * y + center[0]
            y_rot = sin_a * x + cos_a * y + center[1]
            x, y = x_rot, y_rot
            z = np.ones_like(x) * z_height
            
        elif obj_type == 'polygon':
            vertices = np.array(obj_params.get('vertices', [[0.2, -0.05], [0.4, -0.05], [0.35, 0.05], [0.25, 0.05]]))
            # Sample points along edges
            points_list = []
            n_verts = len(vertices)
            n_per_edge = n_points // n_verts
            for i in range(n_verts):
                v1 = vertices[i]
                v2 = vertices[(i+1) % n_verts]
                t = np.linspace(0, 1, n_per_edge, endpoint=False)
                edge_points = v1[None, :] * (1-t[:, None]) + v2[None, :] * t[:, None]
                points_list.append(edge_points)
            xy = np.vstack(points_list)
            x, y = xy[:, 0], xy[:, 1]
            z = np.ones_like(x) * z_height
            
        else:
            raise ValueError(f"Unknown object type: {obj_type}")
            
        boundary_points = np.stack([x, y, z], axis=1)
        return boundary_points
    
    def compute_sdf_and_normals(self, points, pose, theta, used_links=None):
        """
        Compute SDF values and gradient (normal) vectors for query points.
        
        Args:
            points: (N, 3) numpy array of query points
            pose: (B, 4, 4) robot base pose
            theta: (B, 7) joint angles
            used_links: list of link indices to consider
            
        Returns:
            sdf_values: (B, N) SDF values
            normals: (B, N, 3) gradient/normal vectors (pointing away from surface)
        """
        if used_links is None:
            used_links = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            
        points_torch = torch.from_numpy(points).float().to(self.device)
        
        sdf, normals = self.bp_sdf.get_whole_body_sdf_batch(
            points_torch, pose, theta, self.model,
            use_derivative=True, used_links=used_links
        )
        
        return sdf.detach().cpu().numpy(), normals.detach().cpu().numpy()
    
    def project_3d_distance_to_2d(self, sdf_3d, normals_3d):
        """
        Project 3D distance to 2D planar distance.
        
        For points on a horizontal plane (z = constant), the 2D distance can be 
        approximated by projecting the 3D distance using the angle of the normal vector.
        
        Args:
            sdf_3d: (B, N) 3D SDF values (distance to surface)
            normals_3d: (B, N, 3) normal vectors
            
        Returns:
            sdf_2d: (B, N) projected 2D distances
            normals_2d: (B, N, 2) projected 2D normals (x, y components)
        """
        # Get the angle between normal and horizontal plane
        # Normal components: [nx, ny, nz]
        normals_xy = normals_3d[..., :2]  # (B, N, 2)
        normals_z = normals_3d[..., 2]    # (B, N)
        
        # Project distance: d_2d = d_3d * ||(nx, ny)|| / sqrt(nx^2 + ny^2 + nz^2)
        # Since normals are normalized: d_2d = d_3d * ||(nx, ny)||
        norm_xy = np.linalg.norm(normals_xy, axis=-1)  # (B, N)
        
        # Avoid division by zero (when normal is vertical)
        norm_xy = np.maximum(norm_xy, 1e-6)
        
        # Projected 2D distance
        sdf_2d = sdf_3d * norm_xy
        
        # Normalize 2D normals
        normals_2d = normals_xy / norm_xy[..., None]
        
        return sdf_2d, normals_2d
    
    def visualize_sdf_on_z_plane(self, pose, theta, z_height=0.0, 
                                 obj_type='circle', obj_params=None,
                                 xlim=[-0.2, 0.6], ylim=[-0.4, 0.4],
                                 grid_resolution=100, n_boundary_points=32,
                                 used_links=None, safety_threshold=0.03, save_path=None):
        """
        Visualize 3D SDF field on a horizontal plane at specified z-height.
        
        Args:
            pose: robot base pose
            theta: joint angles
            z_height: height of the observation plane
            obj_type: type of object ('circle', 'rectangle', 'polygon')
            obj_params: parameters for the object
            xlim, ylim: viewing bounds for the plane
            grid_resolution: resolution for SDF field
            n_boundary_points: number of points on object boundary
            used_links: links to consider for SDF
            safety_threshold: safety distance threshold (m) to visualize
            save_path: path to save figure
            
        Returns:
            obj_boundary: object boundary points
            obj_sdf: SDF values at boundary points
            obj_normals: normal vectors at boundary points
        """
        if used_links is None:
            used_links = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # ==================== Panel 1: Top View with Robot ====================
        ax1 = axes[0]
        
        # Get object boundary points at z_height
        obj_boundary = self.get_object_boundary_points_2d(
            obj_type, obj_params, z_height, n_boundary_points
        )
        
        # Get robot mesh
        robot_mesh = self.robot.get_forward_robot_mesh(pose, theta)[0]
        robot_mesh = np.sum(robot_mesh) if isinstance(robot_mesh, list) else robot_mesh
        vertices = robot_mesh.vertices
        
        # Project all robot vertices to 2D with color by height
        vertices_2d = vertices[:, :2]
        z_values = vertices[:, 2]
        
        # Create scatter plot with colormap based on z-coordinate
        scatter = ax1.scatter(vertices_2d[:, 0], vertices_2d[:, 1],
                             c=z_values, s=2, alpha=0.6, cmap='viridis',
                             vmin=z_values.min(), vmax=z_values.max())
        
        # Highlight the z-plane
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        
        # Plot object boundary
        ax1.scatter(obj_boundary[:, 0], obj_boundary[:, 1], 
                   c='red', s=80, marker='o', edgecolors='darkred', 
                   linewidths=2, label='Object boundary', zorder=10)
        
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        ax1.set_aspect('equal')
        ax1.set_xlabel('X (m)', fontsize=13)
        ax1.set_ylabel('Y (m)', fontsize=13)
        ax1.set_title(f'Top View - Robot and Object\n(z-plane = {z_height:.3f}m)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11, loc='upper right')
        ax1.grid(True, alpha=0.3)
        cbar1 = plt.colorbar(scatter, ax=ax1, label='Z height (m)')
        
        # ==================== Panel 2: SDF Field on Z-Plane ====================
        ax2 = axes[1]
        
        # Create grid for SDF computation
        x_grid = np.linspace(xlim[0], xlim[1], grid_resolution)
        y_grid = np.linspace(ylim[0], ylim[1], grid_resolution)
        X, Y = np.meshgrid(x_grid, y_grid)
        grid_points = np.stack([X.ravel(), Y.ravel(), 
                               np.ones_like(X.ravel()) * z_height], axis=1)
        
        # Compute 3D SDF at all grid points
        sdf_values, normals = self.compute_sdf_and_normals(
            grid_points, pose, theta, used_links
        )
        sdf_grid = sdf_values.reshape(grid_resolution, grid_resolution)
        
        # Mask region below safety threshold as "robot occupied"
        robot_region = sdf_grid < safety_threshold
        
        # Plot robot occupied region (distance < safety_threshold)
        ax2.contourf(X, Y, robot_region.astype(float), levels=[0.5, 1.5], 
                    colors=['darkgray'], alpha=0.4, zorder=1)
        
        # Plot SDF as filled contours (only for distances >= safety_threshold)
        max_dist = min(0.2, sdf_grid.max() * 0.8)
        levels = np.linspace(safety_threshold, max_dist, 15)
        contourf = ax2.contourf(X, Y, sdf_grid, levels=levels, cmap='RdYlGn_r', alpha=0.9)
        contour_lines = ax2.contour(X, Y, sdf_grid, levels=levels[::2], colors='black', 
                                    linewidths=0.8, alpha=0.4)
        ax2.clabel(contour_lines, inline=True, fontsize=9, fmt='%.3f')
        
        # Plot safety threshold contour (boundary of robot region)
        safety_contour = ax2.contour(X, Y, sdf_grid, levels=[safety_threshold], 
                                     colors='red', linewidths=3, linestyles='solid')
        ax2.clabel(safety_contour, inline=True, fontsize=12, fmt='%.3f m')
        
        # Compute SDF and normals at object boundary
        obj_sdf, obj_normals = self.compute_sdf_and_normals(
            obj_boundary, pose, theta, used_links
        )
        
        # Plot object boundary points with size proportional to distance
        sizes = 100 + obj_sdf.ravel() * 1000  # Scale for visibility
        ax2.scatter(obj_boundary[:, 0], obj_boundary[:, 1],
                   c=obj_sdf.ravel(), s=sizes, cmap='hot', 
                   edgecolors='black', linewidths=2,
                   vmin=0, vmax=max_dist, zorder=10, alpha=0.9)
        
        # Plot normal vectors at boundary points
        arrow_scale = 0.03
        skip = max(1, n_boundary_points // 12)
        for i in range(0, n_boundary_points, skip):
            ax2.arrow(obj_boundary[i, 0], obj_boundary[i, 1],
                     obj_normals[0, i, 0] * arrow_scale,
                     obj_normals[0, i, 1] * arrow_scale,
                     head_width=0.012, head_length=0.015, 
                     fc='blue', ec='darkblue',
                     linewidth=2, alpha=0.8, zorder=11)
        
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)
        ax2.set_aspect('equal')
        ax2.set_xlabel('X (m)', fontsize=13)
        ax2.set_ylabel('Y (m)', fontsize=13)
        ax2.set_title(f'3D SDF Field on Z-Plane\n(Gray = Robot Region, d < {safety_threshold:.3f}m)', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        cbar2 = plt.colorbar(contourf, ax=ax2, label='Distance (m)')
        
        # Add legend for robot region
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='darkgray', alpha=0.4, label=f'Robot Region (d < {safety_threshold:.3f}m)')]
        ax2.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.close(fig)
        
        return obj_boundary, obj_sdf, obj_normals
        """
        Visualize 2D top-down view of robot and object at specific z-height.
        
        Args:
            pose: robot base pose
            theta: joint angles
            z_height: height of the observation plane
            obj_type: type of object ('circle', 'rectangle', 'polygon')
            obj_params: parameters for the object
            xlim, ylim: viewing bounds
            grid_resolution: resolution for distance field visualization
            n_boundary_points: number of points on object boundary
            used_links: links to consider for SDF
            z_tolerance: tolerance for filtering vertices by z (None = auto-adjust)
            project_all_links: if True, project all robot vertices regardless of z-height
            show_robot_projection: whether to show robot mesh projection
            save_path: path to save figure
        """
        if used_links is None:
            used_links = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # ==================== Panel 1: Robot + Object ====================
        ax1 = axes[0]
        
        # Get object boundary points
        obj_boundary = self.get_object_boundary_points_2d(
            obj_type, obj_params, z_height, n_boundary_points
        )
        
        # Plot object
        ax1.scatter(obj_boundary[:, 0], obj_boundary[:, 1], 
                   c='red', s=50, marker='o', label='Object boundary', zorder=10)
        
        # Show robot projection if requested
        if show_robot_projection:
            # Get robot mesh and project to 2D
            robot_mesh = self.robot.get_forward_robot_mesh(pose, theta)[0]
            robot_mesh = np.sum(robot_mesh) if isinstance(robot_mesh, list) else robot_mesh
            
            # Project vertices to 2D
            vertices = robot_mesh.vertices
            
            if project_all_links:
                # Show all robot vertices projected to 2D (regardless of z)
                vertices_2d = vertices[:, :2]
                colors = plt.cm.viridis((vertices[:, 2] - vertices[:, 2].min()) / 
                                       (vertices[:, 2].max() - vertices[:, 2].min() + 1e-6))
                ax1.scatter(vertices_2d[:, 0], vertices_2d[:, 1],
                           c=colors, s=3, alpha=0.5, label='Robot (all links)')
            else:
                # Filter vertices near the z_height
                if z_tolerance is None:
                    # Auto-adjust tolerance based on robot height range
                    z_range = vertices[:, 2].max() - vertices[:, 2].min()
                    z_tolerance = max(0.15, z_range * 0.3)  # At least 15cm or 30% of height range
                
                near_plane = np.abs(vertices[:, 2] - z_height) < z_tolerance
                if near_plane.sum() > 0:
                    vertices_2d = vertices[near_plane, :2]
                    ax1.scatter(vertices_2d[:, 0], vertices_2d[:, 1],
                               c='blue', s=5, alpha=0.3, 
                               label=f'Robot (z={z_height:.2f}±{z_tolerance:.2f}m)')
                else:
                    ax1.text(0.5, 0.5, f'No robot vertices within\nz={z_height:.2f}±{z_tolerance:.2f}m',
                            ha='center', va='center', transform=ax1.transAxes, fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        ax1.set_aspect('equal')
        ax1.set_xlabel('X (m)', fontsize=12)
        ax1.set_ylabel('Y (m)', fontsize=12)
        ax1.set_title(f'Top View (z={z_height:.3f}m)', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ==================== Panel 2: 3D SDF Field ====================
        ax2 = axes[1]
        
        # Create grid for distance field
        x_grid = np.linspace(xlim[0], xlim[1], grid_resolution)
        y_grid = np.linspace(ylim[0], ylim[1], grid_resolution)
        X, Y = np.meshgrid(x_grid, y_grid)
        grid_points = np.stack([X.ravel(), Y.ravel(), 
                               np.ones_like(X.ravel()) * z_height], axis=1)
        
        # Compute 3D SDF
        sdf_3d, normals_3d = self.compute_sdf_and_normals(
            grid_points, pose, theta, used_links
        )
        sdf_3d = sdf_3d.reshape(grid_resolution, grid_resolution)
        
        # Plot 3D SDF contours
        levels = np.linspace(0, 0.15, 10)
        contour = ax2.contourf(X, Y, sdf_3d, levels=levels, cmap='viridis', alpha=0.8)
        contour_lines = ax2.contour(X, Y, sdf_3d, levels=levels, colors='black', 
                                    linewidths=0.5, alpha=0.5)
        ax2.clabel(contour_lines, inline=True, fontsize=8, fmt='%.3f')
        
        # Plot object boundary
        ax2.scatter(obj_boundary[:, 0], obj_boundary[:, 1],
                   c='red', s=50, marker='o', label='Object', zorder=10)
        
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)
        ax2.set_aspect('equal')
        ax2.set_xlabel('X (m)', fontsize=12)
        ax2.set_ylabel('Y (m)', fontsize=12)
        ax2.set_title('3D SDF (Distance to Robot Surface)', fontsize=14)
        plt.colorbar(contour, ax=ax2, label='Distance (m)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # ==================== Panel 3: Projected 2D Distance ====================
        ax3 = axes[2]
        
        # Project to 2D
        normals_3d_grid = normals_3d.reshape(grid_resolution, grid_resolution, 3)
        sdf_2d, normals_2d = self.project_3d_distance_to_2d(
            sdf_3d[None, ...], normals_3d_grid[None, ...]
        )
        sdf_2d = sdf_2d[0]  # Remove batch dimension
        normals_2d = normals_2d[0]
        
        # Plot 2D projected SDF
        contour2 = ax3.contourf(X, Y, sdf_2d, levels=levels, cmap='plasma', alpha=0.8)
        contour_lines2 = ax3.contour(X, Y, sdf_2d, levels=levels, colors='black',
                                     linewidths=0.5, alpha=0.5)
        ax3.clabel(contour_lines2, inline=True, fontsize=8, fmt='%.3f')
        
        # Compute and visualize normals at object boundary
        obj_sdf_3d, obj_normals_3d = self.compute_sdf_and_normals(
            obj_boundary, pose, theta, used_links
        )
        obj_sdf_2d, obj_normals_2d = self.project_3d_distance_to_2d(
            obj_sdf_3d, obj_normals_3d
        )
        
        # Plot normals as arrows
        arrow_scale = 0.05
        skip = max(1, n_boundary_points // 16)  # Show subset of arrows
        for i in range(0, n_boundary_points, skip):
            ax3.arrow(obj_boundary[i, 0], obj_boundary[i, 1],
                     obj_normals_2d[0, i, 0] * arrow_scale,
                     obj_normals_2d[0, i, 1] * arrow_scale,
                     head_width=0.01, head_length=0.01, fc='red', ec='red',
                     linewidth=1.5, alpha=0.8)
        
        # Plot object boundary
        ax3.scatter(obj_boundary[:, 0], obj_boundary[:, 1],
                   c='red', s=50, marker='o', label='Object', zorder=10)
        
        ax3.set_xlim(xlim)
        ax3.set_ylim(ylim)
        ax3.set_aspect('equal')
        ax3.set_xlabel('X (m)', fontsize=12)
        ax3.set_ylabel('Y (m)', fontsize=12)
        ax3.set_title('Projected 2D Distance (with normals)', fontsize=14)
        plt.colorbar(contour2, ax=ax3, label='Distance (m)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.close(fig)
        
        return obj_boundary, obj_sdf_3d, obj_sdf_2d, obj_normals_2d
    
    def visualize_3d_robot_with_object(self, pose, theta, z_height=0.0,
                                      obj_type='circle', obj_params=None,
                                      n_boundary_points=32, save_path=None,
                                      show_interactive=False):
        """
        Visualize 3D robot mesh with object boundary and distance vectors.
        
        Args:
            pose: robot base pose
            theta: joint angles
            z_height: height of the object plane
            obj_type: type of object
            obj_params: object parameters
            n_boundary_points: number of boundary points
            save_path: path to save the visualization
            show_interactive: whether to show interactive 3D window (can cause issues on macOS)
        """
        # Get robot mesh
        robot_mesh = self.robot.get_forward_robot_mesh(pose, theta)[0]
        robot_mesh = np.sum(robot_mesh) if isinstance(robot_mesh, list) else robot_mesh
        
        # Get object boundary
        obj_boundary = self.get_object_boundary_points_2d(
            obj_type, obj_params, z_height, n_boundary_points
        )
        
        # Compute SDF and normals at boundary
        obj_sdf_3d, obj_normals_3d = self.compute_sdf_and_normals(
            obj_boundary, pose, theta
        )
        
        # Create scene
        scene = trimesh.Scene()
        scene.add_geometry(robot_mesh)
        
        # Add object boundary as point cloud
        obj_pc = trimesh.PointCloud(obj_boundary, colors=[255, 0, 0])
        scene.add_geometry(obj_pc)
        
        # Add normal arrows
        for i in range(0, n_boundary_points, max(1, n_boundary_points // 16)):
            pt = obj_boundary[i]
            normal = obj_normals_3d[0, i] * obj_sdf_3d[0, i]  # Scale by distance
            arrow = utils.create_arrow(normal, pt, vec_length=1.0, color=[255, 100, 0])
            scene.add_geometry(arrow)
        
        if save_path:
            # Export scene
            scene.export(save_path)
            print(f"Saved 3D scene to {save_path}")
        
        if show_interactive:
            print("Warning: Interactive 3D viewer may cause issues on macOS when run from debugger")
            scene.show()
        else:
            print("3D scene created. Use --show_3d to view interactively (may have GUI issues)")


def main():
    parser = argparse.ArgumentParser(description='Visualize 2D planar manipulation with 3D SDF')
    parser.add_argument('--device', default='cpu', type=str, help='Device (cuda/cpu)')
    parser.add_argument('--domain_max', default=1.0, type=float)
    parser.add_argument('--domain_min', default=-1.0, type=float)
    parser.add_argument('--n_func', default=8, type=int, help='Number of basis functions')
    parser.add_argument('--z_height', default=0.6, type=float, help='Height of observation plane')
    parser.add_argument('--obj_type', default='circle', choices=['circle', 'rectangle', 'polygon'],
                       help='Object type')
    parser.add_argument('--obj_x', default=0.3, type=float, help='Object center x')
    parser.add_argument('--obj_y', default=0.0, type=float, help='Object center y')
    parser.add_argument('--obj_size', default=0.05, type=float, help='Object size (radius/width)')
    parser.add_argument('--show_3d', action='store_true', help='Show 3D visualization')
    parser.add_argument('--save_dir', default='figs', type=str, help='Directory to save figures')
    parser.add_argument('--project_all_links', action='store_true', 
                       help='Project all robot links to 2D (regardless of z-height)')
    parser.add_argument('--z_tolerance', type=float, default=None,
                       help='Tolerance for z-height filtering (None = auto)')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device)
    
    # Initialize robot and SDF
    print("Initializing robot and SDF model...")
    panda = PandaLayer(device)
    bp_sdf = bf_sdf.BPSDF(args.n_func, args.domain_min, args.domain_max, panda, device)
    
    # Load trained model
    model_path = os.path.join(CUR_DIR, 'models', f'BP_{args.n_func}.pt')
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please train the model first using: python bf_sdf.py --train")
        return
    
    model = torch.load(model_path, map_location=device, weights_only=False)
    print(f"Loaded model from {model_path}")
    
    # Set robot configuration
    theta = torch.tensor([np.pi/2, np.pi/2, -np.pi/2, 0, 0, np.pi, np.pi/4]).float().to(device).reshape(-1, 7)
    pose = torch.eye(4).unsqueeze(0).to(device).float()
    
    # Create visualizer
    visualizer = PlanarManipulationVisualizer(bp_sdf, model, panda, device)
    
    # Define object parameters
    obj_params = {
        'center': [args.obj_x, args.obj_y],
        'radius': args.obj_size,
        'width': args.obj_size * 2,
        'height': args.obj_size * 1.5,
        'angle': 0.0
    }
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    save_path_2d = os.path.join(args.save_dir, f'planar_2d_z{args.z_height:.3f}_{args.obj_type}.png')
    
    # Visualize SDF on Z-plane
    print(f"\nVisualizing SDF field at z={args.z_height}m...")
    
    obj_boundary, obj_sdf, obj_normals = visualizer.visualize_sdf_on_z_plane(
        pose, theta,
        z_height=args.z_height,
        obj_type=args.obj_type,
        obj_params=obj_params,
        xlim=[-0.4, 1.0],
        ylim=[-1.0, 1.0],
        grid_resolution=100,
        n_boundary_points=64,
        safety_threshold=0.025,
        save_path=save_path_2d
    )
    
    # Print statistics
    print(f"\nObject boundary statistics:")
    print(f"  Number of points: {len(obj_boundary)}")
    print(f"  SDF range: [{obj_sdf.min():.4f}, {obj_sdf.max():.4f}]")
    print(f"  SDF mean: {obj_sdf.mean():.4f}")
    print(f"  Min distance to robot: {obj_sdf.min():.4f}m")
    print(f"  Max distance to robot: {obj_sdf.max():.4f}m")
    
    # Robot region analysis (distance < safety_threshold considered as robot)
    n_in_robot = np.sum(obj_sdf < 0.03)
    print(f"\nRobot region analysis (d < 0.03m = robot occupied):")
    print(f"  Points inside robot region: {n_in_robot}/{len(obj_boundary)}")
    if n_in_robot > 0:
        print(f"  WARNING: {n_in_robot} boundary points are within robot region!")
        print(f"  These points are considered part of the robot arm.")
    else:
        print(f"  Object is outside robot region (safe).")
    
    # Show 3D visualization if requested
    if args.show_3d:
        print("\nShowing 3D visualization...")
        save_path_3d = os.path.join(args.save_dir, f'planar_3d_z{args.z_height:.3f}_{args.obj_type}.obj')
        visualizer.visualize_3d_robot_with_object(
            pose, theta,
            z_height=args.z_height,
            obj_type=args.obj_type,
            obj_params=obj_params,
            save_path=save_path_3d,
            show_interactive=True  # Enable interactive viewer
        )
    
    print("\nVisualization complete!")
    print(f"2D visualization saved to: {save_path_2d}")


if __name__ == '__main__':
    main()

