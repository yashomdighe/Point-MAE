"""
PointMAE 1D Rotation Sweep Evaluation Script.

Sweeps rotations independently along each axis (X, Y, Z) and plots latent distances.
Unit-sphere normalization is always applied (PointMAE requirement).

Usage:
    python eval_rotation_sweep_1d.py \
        --checkpoint experiments/pretrain_ycb/cfgs/ycb_finetune/ckpt-last.pth \
        --objects 025_mug 011_banana --output-root results --noise-std 0.001
"""
import sys
import os

# Add Point-MAE root to path
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation as sciR
import argparse
import pandas as pd
import trimesh

# Import Point-MAE model
from models.Point_MAE import Point_MAE


def load_ycb_object(ycb_root, object_name, input_num=4096, seed=42):
    """Load object point cloud from YCB or MuJoCo dataset (XYZ only)."""
    ycb_root = Path(ycb_root)
    obj_dir = ycb_root / object_name
    
    # Try YCB structure first
    obj_path = obj_dir / 'google_64k' / 'textured.obj'
    if not obj_path.exists():
        obj_path = obj_dir / 'google_64k' / 'nontextured.ply'
    if not obj_path.exists():
        obj_path = obj_dir / 'clouds' / 'merged_cloud.ply'
    
    # Try MuJoCo structure (model.obj)
    if not obj_path.exists():
        obj_path = obj_dir / 'model.obj'
    
    if not obj_path.exists():
        raise ValueError(f"No mesh/PLY file found for {object_name}")
    
    # Load mesh with trimesh
    tm = trimesh.load(str(obj_path), process=True)
    if isinstance(tm, trimesh.Scene):
        tm = trimesh.util.concatenate(tuple(tm.geometry.values()))
    
    # Set random seed for reproducible sampling
    np.random.seed(seed)
    
    # Uniform area-weighted surface sampling
    pts_np, _ = tm.sample(input_num, return_index=True)
    pts_np = pts_np.astype(np.float32)
    
    return pts_np  # XYZ only


def pc_norm(pc):
    """Normalize point cloud to unit sphere (matching PointMAE training)."""
    # pc: (N, 3) numpy array
    centroid = pc.mean(axis=0)
    pc_centered = pc - centroid
    
    # Max distance from origin
    dist = np.sqrt((pc_centered ** 2).sum(axis=1))
    scale = dist.max()
    if scale > 0:
        pc_normalized = pc_centered / scale
    else:
        pc_normalized = pc_centered
    
    return pc_normalized


def transform_points(points, q, t):
    """Transform points by quaternion rotation and translation."""
    from scipy.spatial.transform import Rotation as R
    rot = R.from_quat(q)
    return rot.apply(points) + t


def quaternion_from_rotation_matrix(R):
    """Convert rotation matrix to quaternion."""
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_matrix(R).as_quat()


def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix."""
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_quat(q).as_matrix()


def generate_random_pose(rng=None):
    """Generate random SE(3) pose."""
    if rng is None:
        rng = np.random.RandomState()
    
    q = rng.randn(4)
    q = q / np.linalg.norm(q)
    t = rng.randn(3).astype(np.float32) * 0.01
    
    return q.astype(np.float32), t


def compute_geodesic_distance(q1, q2):
    """Compute geodesic distance in SO(3)."""
    R1 = quaternion_to_rotation_matrix(q1)
    R2 = quaternion_to_rotation_matrix(q2)
    R_diff = R2 @ R1.T
    
    trace = np.trace(R_diff)
    theta = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    return theta


def apply_single_axis_rotation(q_base, axis, angle_deg):
    """Apply rotation around a single axis."""
    R_base = quaternion_to_rotation_matrix(q_base)
    
    axis_vec = {
        'x': np.array([1, 0, 0]),
        'y': np.array([0, 1, 0]),
        'z': np.array([0, 0, 1])
    }[axis]
    
    R_rot = sciR.from_rotvec(np.deg2rad(angle_deg) * axis_vec).as_matrix().astype(np.float32)
    R_rotated = R_rot @ R_base
    return quaternion_from_rotation_matrix(R_rotated)


def apply_augmentations(points, noise_std=0.0, shuffle=False):
    """
    Apply augmentation pipeline.
    Note: unit_sphere_norm is ALWAYS applied for PointMAE.
    """
    n_pts = points.shape[0]
    
    # Add Gaussian noise to XYZ
    if noise_std > 0.0:
        points = points + np.random.randn(n_pts, 3).astype(np.float32) * noise_std
    
    # Unit sphere normalization (ALWAYS applied)
    points = pc_norm(points)
    
    # Random permutation
    if shuffle:
        perm = np.random.permutation(n_pts)
        points = points[perm]
    
    return points


def extract_features(model, points, device):
    """Extract features from PointMAE encoder."""
    # points: (N, 3) numpy array
    # Convert to torch and add batch dim
    points_t = torch.from_numpy(points).unsqueeze(0).float().to(device)  # (1, N, 3)
    
    with torch.no_grad():
        # First divide into groups
        neighborhood, center = model.group_divider(points_t)  # neighborhood: (1, G, M, 3), center: (1, G, 3)
        
        # Then pass through encoder (noaug=True to disable masking)
        x_vis, mask = model.MAE_encoder(neighborhood, center, noaug=True)  # (1, G, C)
        
        # Max pool across groups to get global feature
        features = x_vis.max(dim=1)[0]  # (1, trans_dim)
    
    return features


def sweep_single_axis_rotations(model, points_base, q_base, t_base, device, angle_step=10, 
                                max_angle=180.0, noise_std=0.0, shuffle=False):
    """Sweep rotations independently along each axis."""
    axes = ['x', 'y', 'z']
    results = {}
    
    # Reference
    points_ref = transform_points(points_base.copy(), q_base, t_base)
    points_ref = apply_augmentations(points_ref, noise_std, shuffle)
    z_ref = extract_features(model, points_ref, device)
    
    angles = np.arange(0, max_angle + 1e-6, angle_step)
    
    for axis in axes:
        print(f"  Sweeping {axis.upper()} axis...")
        latent_distances = np.zeros(len(angles))
        geodesic_distances = np.zeros(len(angles))
        
        for i, angle in enumerate(angles):
            # Rotate, then apply fresh augmentations
            q_rot = apply_single_axis_rotation(q_base, axis, angle)
            points_rot = transform_points(points_base.copy(), q_rot, t_base)
            points_rot = apply_augmentations(points_rot, noise_std, shuffle)
            
            # Compute geodesic distance
            geodesic_dist = compute_geodesic_distance(q_base, q_rot)
            geodesic_distances[i] = geodesic_dist
            
            # Extract features and compute distance
            z_rot = extract_features(model, points_rot, device)
            latent_dist = torch.norm(z_ref - z_rot, dim=1).item()
            latent_distances[i] = latent_dist
        
        results[axis] = {
            'angles': angles,
            'geodesic_distances': geodesic_distances,
            'latent_distances': latent_distances
        }
    
    return results


def plot_1d_results(results, output_path, title_prefix):
    """Plot 1D line plots with geodesic distance on x-axis."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{title_prefix}', fontsize=16)
    
    axis_names = ['x', 'y', 'z']
    colors = ['red', 'green', 'blue']
    
    for idx, axis in enumerate(axis_names):
        if axis not in results:
            continue
        
        res = results[axis]
        ax = axes[idx]
        
        ax.plot(res['geodesic_distances'], res['latent_distances'], 
                color=colors[idx], linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Geodesic Distance (radians)')
        ax.set_ylabel('Latent Distance')
        ax.set_title(f'{axis.upper()} Axis Rotation')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([res['geodesic_distances'].min(), res['geodesic_distances'].max()])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
    plt.close()


def save_results_to_csv(results, output_path):
    """Save results to CSV."""
    rows = []
    
    for axis, res in results.items():
        angles = res['angles']
        geodesic_distances = res['geodesic_distances']
        latent_distances = res['latent_distances']
        
        for angle, geo_dist, lat_dist in zip(angles, geodesic_distances, latent_distances):
            rows.append({
                'axis': axis,
                'angle': angle,
                'geodesic_distance': geo_dist,
                'latent_distance': lat_dist
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved CSV to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='PointMAE 1D Rotation Sweep')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to PointMAE checkpoint')
    parser.add_argument('--objects', type=str, nargs='+', required=True, help='YCB object names')
    parser.add_argument('--ycb-root', type=str, default='../E2PN/ycb', help='Path to YCB dataset')
    parser.add_argument('--output-root', type=str, required=True, help='Output directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--input-num', type=int, default=4096, help='Number of points')
    parser.add_argument('--angle-step', type=float, default=10, help='Angle step (degrees)')
    parser.add_argument('--max-angle', type=float, default=180.0, help='Max angle (degrees)')
    parser.add_argument('--noise-std', type=float, default=0.0, help='Gaussian noise std')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle points')
    
    args = parser.parse_args()
    
    # Path handling
    if not os.path.isabs(args.ycb_root):
        args.ycb_root = os.path.join(os.path.dirname(__file__), args.ycb_root)
    if not os.path.isabs(args.checkpoint):
        args.checkpoint = os.path.join(os.path.dirname(__file__), args.checkpoint)
    
    # Build PointMAE model
    print("Building PointMAE model...")
    from utils.config import cfg_from_yaml_file
    from easydict import EasyDict
    
    # Create minimal config for Point-MAE
    config = EasyDict({
        'model': EasyDict({
            'NAME': 'Point_MAE',
            'group_size': 32,
            'num_group': 128,
            'loss': 'cdl1',
            'transformer_config': EasyDict({
                'mask_ratio': 0.6,
                'mask_type': 'rand',
                'trans_dim': 384,
                'depth': 12,
                'drop_path_rate': 0.1,
                'num_heads': 6,
                'encoder_dims': 384,
                'decoder_depth': 4,
                'decoder_num_heads': 6,
            })
        })
    })
    
    model = Point_MAE(config.model).to(args.device)
    
    # Load checkpoint
    print(f"Loading checkpoint {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    if 'base_model' in checkpoint:
        state_dict = checkpoint['base_model']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DataParallel)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    
    model.eval()
    
    # Process objects
    print(f"\nProcessing {len(args.objects)} object(s)...")
    
    for idx, obj_name in enumerate(args.objects):
        print(f"\n[{idx+1}/{len(args.objects)}] Processing {obj_name}...")
        
        try:
            points_base = load_ycb_object(args.ycb_root, obj_name, 
                                           input_num=args.input_num, seed=args.seed)
            
            # Base pose
            rng = np.random.RandomState(args.seed)
            q_base, t_base = generate_random_pose(rng=rng)
            
            # Sweep rotations
            print(f"Sweeping rotations (max_angle={args.max_angle}, step={args.angle_step})...")
            print(f"Augmentations: noise_std={args.noise_std}, shuffle={args.shuffle}, unit_sphere_norm=True (always)")
            
            rot_results = sweep_single_axis_rotations(
                model, points_base, q_base, t_base, args.device,
                angle_step=args.angle_step, max_angle=args.max_angle,
                noise_std=args.noise_std, shuffle=args.shuffle
            )
            
            # Create output directory
            output_dir = Path(args.output_root) / f'rotation_sweep_1d_{obj_name}'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save results
            plot_path = output_dir / f"rotation_sweep_1d_{obj_name}.png"
            plot_1d_results(rot_results, plot_path, f"PointMAE 1D Rotation Sweep: {obj_name}")
            
            csv_path = output_dir / f"rotation_sweep_1d_{obj_name}.csv"
            save_results_to_csv(rot_results, csv_path)
            
        except Exception as e:
            print(f"Error processing {obj_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("Done!")


if __name__ == '__main__':
    main()
