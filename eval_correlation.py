"""
PointMAE Correlation Evaluation Script.

Evaluates correlation between geodesic distance (rotation) and latent distance
using random rotation samples. Generates scatter plots with regression lines.

Usage:
    python eval_correlation.py \
        --checkpoint experiments/pretrain_ycb/cfgs/ycb_finetune/ckpt-last.pth \
        --objects 025_mug 011_banana --output-dir results/correlation --noise-std 0.001
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation as sciR
from scipy import stats
import argparse
import pandas as pd
import trimesh
import yaml

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
    
    tm = trimesh.load(str(obj_path), process=True)
    if isinstance(tm, trimesh.Scene):
        tm = trimesh.util.concatenate(tuple(tm.geometry.values()))
    
    np.random.seed(seed)
    pts_np, _ = tm.sample(input_num, return_index=True)
    pts_np = pts_np.astype(np.float32)
    
    return pts_np


def get_valid_objects(dataset_root):
    """Get list of all valid objects from YCB or MuJoCo dataset.
    
    Detects dataset type automatically:
    - YCB: Objects starting with digits, containing google_64k/ or clouds/ subdirs
    - MuJoCo: Objects in models/ subdirectory with model.obj files
    
    Returns:
        List of object directory names
    """
    dataset_root = Path(dataset_root)
    valid_objects = []
    
    # Check if this is MuJoCo structure (has models/ subdirectory)
    models_dir = dataset_root / 'models'
    if models_dir.exists() and models_dir.is_dir():
        # MuJoCo scanned objects structure
        for item in sorted(models_dir.iterdir()):
            if item.is_dir() and (item / 'model.obj').exists():
                valid_objects.append(f'models/{item.name}')
        if valid_objects:
            print(f"Detected MuJoCo dataset structure: {len(valid_objects)} objects")
            return valid_objects
    
    # YCB structure: objects directly in root, starting with digits
    for item in sorted(dataset_root.iterdir()):
        if item.is_dir() and item.name[0].isdigit():
            # Check for YCB mesh files
            has_mesh = (
                (item / 'google_64k' / 'textured.obj').exists() or
                (item / 'google_64k' / 'nontextured.ply').exists() or
                (item / 'clouds' / 'merged_cloud.ply').exists()
            )
            if has_mesh:
                valid_objects.append(item.name)
    
    if valid_objects:
        print(f"Detected YCB dataset structure: {len(valid_objects)} objects")
    
    return valid_objects


def pc_norm(pc):
    """Normalize to unit sphere."""
    centroid = pc.mean(axis=0)
    pc_centered = pc - centroid
    dist = np.sqrt((pc_centered ** 2).sum(axis=1))
    scale = dist.max()
    if scale > 0:
        pc_normalized = pc_centered / scale
    else:
        pc_normalized = pc_centered
    return pc_normalized


def transform_points(points, q, t):
    """Transform by quaternion + translation."""
    from scipy.spatial.transform import Rotation as R
    rot = R.from_quat(q)
    return rot.apply(points) + t


def quaternion_from_rotation_matrix(R):
    """Rotation matrix to quaternion."""
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_matrix(R).as_quat()


def quaternion_to_rotation_matrix(q):
    """Quaternion to rotation matrix."""
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
    """Geodesic distance in SO(3)."""
    R1 = quaternion_to_rotation_matrix(q1)
    R2 = quaternion_to_rotation_matrix(q2)
    R_diff = R2 @ R1.T
    trace = np.trace(R_diff)
    theta = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    return theta


def apply_augmentations(points, noise_std=0.0, shuffle=False):
    """Apply augmentations (unit-sphere always applied)."""
    n_pts = points.shape[0]
    
    if noise_std > 0.0:
        points = points + np.random.randn(n_pts, 3).astype(np.float32) * noise_std
    
    points = pc_norm(points)
    
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


def evaluate_correlation(model, points_base, device, num_samples=500, 
                         noise_std=0.0, shuffle=False, seed=42):
    """
    Evaluate correlation between geodesic and latent distance.
    
    Generates random rotation pairs and measures:
    - Geodesic distance (SO(3) distance between rotations)
    - Latent L1 distance (Manhattan distance between embeddings)
    - Latent L2 distance (Euclidean distance between embeddings)
    """
    rng = np.random.RandomState(seed)
    
    geodesic_dists = []
    latent_l1_dists = []
    latent_l2_dists = []
    
    print(f"  Generating {num_samples} random rotation pairs...")
    
    for i in range(num_samples):
        # Generate two random poses
        q1, t1 = generate_random_pose(rng)
        q2, t2 = generate_random_pose(rng)
        
        # Transform and augment
        points1 = transform_points(points_base.copy(), q1, t1)
        points1 = apply_augmentations(points1, noise_std, shuffle)
        
        points2 = transform_points(points_base.copy(), q2, t2)
        points2 = apply_augmentations(points2, noise_std, shuffle)
        
        # Compute geodesic distance
        geo_dist = compute_geodesic_distance(q1, q2)
        geodesic_dists.append(geo_dist)
        
        # Compute latent distances (both L1 and L2)
        z1 = extract_features(model, points1, device)
        z2 = extract_features(model, points2, device)
        
        # L1 distance (Manhattan)
        l1_dist = torch.abs(z1 - z2).sum(dim=1).item()
        latent_l1_dists.append(l1_dist)
        
        # L2 distance (Euclidean)
        l2_dist = torch.norm(z1 - z2, dim=1).item()
        latent_l2_dists.append(l2_dist)
    
    geodesic_dists = np.array(geodesic_dists)
    latent_l1_dists = np.array(latent_l1_dists)
    latent_l2_dists = np.array(latent_l2_dists)
    
    # Compute regression for L1
    slope_l1, intercept_l1, r_value_l1, p_value_l1, std_err_l1 = stats.linregress(geodesic_dists, latent_l1_dists)
    
    # Compute regression for L2
    slope_l2, intercept_l2, r_value_l2, p_value_l2, std_err_l2 = stats.linregress(geodesic_dists, latent_l2_dists)
    
    results = {
        'geodesic_distances': geodesic_dists,
        'latent_l1_distances': latent_l1_dists,
        'latent_l2_distances': latent_l2_dists,
        # L1 stats
        'l1_slope': slope_l1,
        'l1_intercept': intercept_l1,
        'l1_r_squared': r_value_l1 ** 2,
        'l1_p_value': p_value_l1,
        'l1_std_err': std_err_l1,
        # L2 stats
        'l2_slope': slope_l2,
        'l2_intercept': intercept_l2,
        'l2_r_squared': r_value_l2 ** 2,
        'l2_p_value': p_value_l2,
        'l2_std_err': std_err_l2,
    }
    
    return results


def plot_correlation(results_dict, output_dir, title_prefix="PointMAE Correlation"):
    """Plot correlation scatter with regression lines for both L1 and L2 (separate figures)."""
    # Individual plots per object (L1 and L2 as separate figures)
    for obj_name, results in results_dict.items():
        geo = results['geodesic_distances']
        geo_line = np.linspace(geo.min(), geo.max(), 100)
        
        # Sanitize object name for filesystem
        obj_name_safe = obj_name.replace('/', '_')
        
        # L1 plot
        lat_l1 = results['latent_l1_distances']
        slope_l1 = results['l1_slope']
        intercept_l1 = results['l1_intercept']
        r2_l1 = results['l1_r_squared']
        
        fig_l1 = plt.figure(figsize=(8, 6))
        ax = fig_l1.add_subplot(111)
        ax.scatter(geo, lat_l1, alpha=0.5, s=20, label='Data points', color='blue')
        lat_line_l1 = slope_l1 * geo_line + intercept_l1
        ax.plot(geo_line, lat_line_l1, 'r-', linewidth=2, 
                label=f'y = {slope_l1:.2f}x + {intercept_l1:.2f}\n$R^2$ = {r2_l1:.4f}')
        ax.set_xlabel('Geodesic Distance (radians)')
        ax.set_ylabel('Latent L1 Distance')
        ax.set_title(f'{title_prefix} (L1): {obj_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path_l1_png = output_dir / f'correlation_l1_{obj_name_safe}.png'
        plot_path_l1_svg = output_dir / f'correlation_l1_{obj_name_safe}.svg'
        plt.savefig(plot_path_l1_png, dpi=150)
        plt.savefig(plot_path_l1_svg)
        print(f"  Saved L1 plot to {plot_path_l1_png} and {plot_path_l1_svg}")
        plt.close()
        
        # L2 plot
        lat_l2 = results['latent_l2_distances']
        slope_l2 = results['l2_slope']
        intercept_l2 = results['l2_intercept']
        r2_l2 = results['l2_r_squared']
        
        fig_l2 = plt.figure(figsize=(8, 6))
        ax = fig_l2.add_subplot(111)
        ax.scatter(geo, lat_l2, alpha=0.5, s=20, label='Data points', color='green')
        lat_line_l2 = slope_l2 * geo_line + intercept_l2
        ax.plot(geo_line, lat_line_l2, 'r-', linewidth=2, 
                label=f'y = {slope_l2:.2f}x + {intercept_l2:.2f}\n$R^2$ = {r2_l2:.4f}')
        ax.set_xlabel('Geodesic Distance (radians)')
        ax.set_ylabel('Latent L2 Distance')
        ax.set_title(f'{title_prefix} (L2): {obj_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path_l2_png = output_dir / f'correlation_l2_{obj_name_safe}.png'
        plot_path_l2_svg = output_dir / f'correlation_l2_{obj_name_safe}.svg'
        plt.savefig(plot_path_l2_png, dpi=150)
        plt.savefig(plot_path_l2_svg)
        print(f"  Saved L2 plot to {plot_path_l2_png} and {plot_path_l2_svg}")
        plt.close()


def save_results_to_csv(results_dict, output_dir):
    """Save correlation results to CSV (both L1 and L2)."""
    # Individual CSVs per object
    for obj_name, results in results_dict.items():
        rows = []
        geo = results['geodesic_distances']
        lat_l1 = results['latent_l1_distances']
        lat_l2 = results['latent_l2_distances']
        
        for g, l1, l2 in zip(geo, lat_l1, lat_l2):
            rows.append({
                'object': obj_name,
                'geodesic_distance': g,
                'latent_l1_distance': l1,
                'latent_l2_distance': l2
            })
        
        df = pd.DataFrame(rows)
        csv_path = output_dir / f'correlation_data_{obj_name}.csv'
        df.to_csv(csv_path, index=False)
        print(f"  Saved data to {csv_path}")
    
    # Summary statistics CSV
    summary_rows = []
    for obj_name, results in results_dict.items():
        summary_rows.append({
            'object': obj_name,
            'l1_slope': results['l1_slope'],
            'l1_intercept': results['l1_intercept'],
            'l1_r_squared': results['l1_r_squared'],
            'l1_p_value': results['l1_p_value'],
            'l1_std_err': results['l1_std_err'],
            'l2_slope': results['l2_slope'],
            'l2_intercept': results['l2_intercept'],
            'l2_r_squared': results['l2_r_squared'],
            'l2_p_value': results['l2_p_value'],
            'l2_std_err': results['l2_std_err'],
            'num_samples': len(results['geodesic_distances'])
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / 'correlation_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved summary to {summary_path}")


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description='PointMAE Correlation Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Using config file
  python eval_correlation.py --config configs/eval_pointmae.yaml
  
  # Legacy CLI arguments
  python eval_correlation.py --checkpoint ckpt.pth --objects 077_rubiks_cube \\
      --ycb-root ../E2PN/ycb --output-dir results/correlation
        """)
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint (overrides config)')
    parser.add_argument('--objects', type=str, nargs='+', default=None, help='YCB objects (overrides config)')
    parser.add_argument('--ycb-root', type=str, default=None, help='YCB dataset path (overrides config)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (overrides config)')
    parser.add_argument('--device', type=str, default=None, help='Device: cuda or cpu (overrides config)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (overrides config)')
    parser.add_argument('--input-num', type=int, default=None, help='Number of points (overrides config)')
    parser.add_argument('--num-samples', type=int, default=None, help='Number of rotation pairs (overrides config)')
    parser.add_argument('--noise-std', type=float, default=None, help='Gaussian noise std (overrides config)')
    parser.add_argument('--shuffle', action='store_true', default=None, help='Shuffle points (overrides config)')
    
    args = parser.parse_args()
    
    # Load config file if provided, otherwise use legacy CLI args
    if args.config:
        config = load_config(args.config)
        
        # Extract config values with CLI overrides
        ycb_root = args.ycb_root if args.ycb_root else config['dataset']['root']
        use_all = config['dataset']['use_all']
        objects_list = args.objects if args.objects else config['dataset'].get('objects', [])
        
        checkpoint = args.checkpoint if args.checkpoint else config['model']['checkpoint']
        input_num = args.input_num if args.input_num is not None else config['pointcloud']['input_num']
        
        noise_std = args.noise_std if args.noise_std is not None else config['augmentation']['noise_std']
        shuffle = args.shuffle if args.shuffle is not None else config['augmentation']['shuffle']
        
        num_samples = args.num_samples if args.num_samples is not None else config['evaluation']['num_samples']
        seed = args.seed if args.seed is not None else config['evaluation']['seed']
        device = args.device if args.device else config['evaluation']['device']
        
        output_dir = args.output_dir if args.output_dir else config['output']['dir']
    else:
        # Legacy mode: all args must be provided via CLI
        if not args.checkpoint or not args.objects or not args.output_dir:
            parser.error('--checkpoint, --objects, and --output-dir are required when not using --config')
        
        ycb_root = args.ycb_root if args.ycb_root else '../E2PN/ycb'
        use_all = False
        objects_list = args.objects
        checkpoint = args.checkpoint
        input_num = args.input_num if args.input_num is not None else 4096
        noise_std = args.noise_std if args.noise_std is not None else 0.0
        shuffle = args.shuffle if args.shuffle is not None else False
        num_samples = args.num_samples if args.num_samples is not None else 500
        seed = args.seed if args.seed is not None else 42
        device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
        output_dir = args.output_dir
    
    # Path handling
    if not os.path.isabs(ycb_root):
        ycb_root = os.path.join(os.path.dirname(__file__), ycb_root)
    if not os.path.isabs(checkpoint):
        checkpoint = os.path.join(os.path.dirname(__file__), checkpoint)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all valid objects if use_all is True
    if args.config and use_all:
        print(f"Scanning dataset root: {ycb_root}")
        all_valid_objects = get_valid_objects(ycb_root)
        objects_to_process = all_valid_objects
        print(f"Using all {len(objects_to_process)} objects from dataset")
    else:
        objects_to_process = objects_list
        print(f"Using {len(objects_to_process)} specified objects")
    
    # Build model
    print("Building PointMAE model...")
    from easydict import EasyDict
    
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
    
    model = Point_MAE(config.model).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint {checkpoint}...")
    ckpt_data = torch.load(checkpoint, map_location=device)
    
    if 'base_model' in ckpt_data:
        state_dict = ckpt_data['base_model']
    elif 'model' in ckpt_data:
        state_dict = ckpt_data['model']
    else:
        state_dict = ckpt_data
    
    # Remove 'module.' prefix if present (from DataParallel)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    
    model.eval()
    
    # Evaluate correlation for each object
    print(f"\nEvaluating {len(objects_to_process)} object(s)...")
    results_dict = {}
    
    for idx, obj_name in enumerate(objects_to_process):
        print(f"\n[{idx+1}/{len(objects_to_process)}] Processing {obj_name}...")
        
        try:
            points_base = load_ycb_object(ycb_root, obj_name, 
                                           input_num=input_num, seed=seed)
            
            print(f"  Evaluating correlation ({num_samples} samples)...")
            print(f"  Augmentations: noise_std={noise_std}, shuffle={shuffle}, unit_sphere_norm=True (always)")
            
            results = evaluate_correlation(
                model, points_base, device,
                num_samples=num_samples,
                noise_std=noise_std,
                shuffle=shuffle,
                seed=seed
            )
            
            results_dict[obj_name] = results
            
            print(f"  Results L1: slope={results['l1_slope']:.4f}, R²={results['l1_r_squared']:.4f}")
            print(f"  Results L2: slope={results['l2_slope']:.4f}, R²={results['l2_r_squared']:.4f}")
            
        except Exception as e:
            print(f"  Error processing {obj_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if results_dict:
        # Generate plots
        print("\nGenerating plots...")
        plot_correlation(results_dict, output_dir)
        
        # Save CSVs
        print("\nSaving results to CSV...")
        save_results_to_csv(results_dict, output_dir)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
