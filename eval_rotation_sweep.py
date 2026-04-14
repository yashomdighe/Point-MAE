"""
PointMAE 2D Rotation Sweep Evaluation Script.

Sweeps rotations jointly over pairs of axes (XY, YZ, ZX) and plots latent distances.
Unit-sphere normalization is always applied (PointMAE requirement).

Usage:
    python eval_rotation_sweep.py \
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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from pathlib import Path
from scipy.spatial.transform import Rotation as sciR
import argparse
import pandas as pd
import trimesh
import yaml

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
    """Normalize point cloud to unit sphere."""
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


def apply_joint_rotation(q_base, axis1, angle1_deg, axis2, angle2_deg):
    """Apply rotation around axis2 then axis1."""
    R_base = quaternion_to_rotation_matrix(q_base)
    
    def get_axis_vec(axis):
        if axis == 'x': return np.array([1, 0, 0])
        elif axis == 'y': return np.array([0, 1, 0])
        elif axis == 'z': return np.array([0, 0, 1])
    
    R1 = sciR.from_rotvec(np.deg2rad(angle1_deg) * get_axis_vec(axis1)).as_matrix().astype(np.float32)
    R2 = sciR.from_rotvec(np.deg2rad(angle2_deg) * get_axis_vec(axis2)).as_matrix().astype(np.float32)
    
    R_rotated = R2 @ R1 @ R_base
    return quaternion_from_rotation_matrix(R_rotated)


def apply_augmentations(points, noise_std=0.0, shuffle=False):
    """Apply augmentation pipeline (unit-sphere norm always applied)."""
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
    points_t = torch.from_numpy(points).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        # First divide into groups
        neighborhood, center = model.group_divider(points_t)
        
        # Then pass through encoder (noaug=True to disable masking)
        x_vis, mask = model.MAE_encoder(neighborhood, center, noaug=True)
        
        # Max pool across groups to get global feature
        features = x_vis.max(dim=1)[0]
    
    return features


def sweep_joint_rotations(model, points_base, q_base, t_base, device, angle_step=10, 
                          max_angle=180.0, noise_std=0.0, shuffle=False):
    """Sweep rotations jointly over pairs of axes."""
    pairs = [('x', 'y'), ('y', 'z'), ('z', 'x')]
    results = {}
    
    # Reference
    points_ref = transform_points(points_base.copy(), q_base, t_base)
    points_ref = apply_augmentations(points_ref, noise_std, shuffle)
    z_ref = extract_features(model, points_ref, device)
    
    angles = np.arange(0, max_angle + 1e-6, angle_step)
    
    for ax1, ax2 in pairs:
        print(f"  Sweeping {ax1}-{ax2} rotations...")
        pair_key = f"{ax1}{ax2}"
        grid_dists = np.zeros((len(angles), len(angles)))
        
        for i, a1 in enumerate(angles):
            for j, a2 in enumerate(angles):
                q_rot = apply_joint_rotation(q_base, ax1, a1, ax2, a2)
                points_rot = transform_points(points_base.copy(), q_rot, t_base)
                points_rot = apply_augmentations(points_rot, noise_std, shuffle)
                
                z_rot = extract_features(model, points_rot, device)
                dist = torch.norm(z_ref - z_rot, dim=1).item()
                grid_dists[i, j] = dist
        
        results[pair_key] = {
            'axis1': ax1, 'axis2': ax2,
            'angles': angles,
            'distances': grid_dists
        }
    
    return results


def plot_3d_results(results, output_path, title_prefix):
    """Plot 3D surfaces (paper-quality: no titles, larger labels)."""
    LABEL_SIZE = 18
    TICK_SIZE = 14
    
    fig = plt.figure(figsize=(22, 7))
    # No suptitle for paper figures
    
    pairs = ['xy', 'yz', 'zx']
    
    for idx, key in enumerate(pairs):
        if key not in results:
            continue
        
        res = results[key]
        ax = fig.add_subplot(1, 3, idx+1, projection='3d')
        
        X_vals = res['angles']
        Y_vals = X_vals
        
        X, Y = np.meshgrid(X_vals, Y_vals)
        Z = res['distances'].T
        
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        
        ax.set_xlabel(f'{res["axis1"].upper()} Angle (°)', fontsize=LABEL_SIZE, labelpad=12)
        ax.set_ylabel(f'{res["axis2"].upper()} Angle (°)', fontsize=LABEL_SIZE, labelpad=12)
        ax.set_zlabel('Latent Distance', fontsize=LABEL_SIZE, labelpad=25)
        ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
        ax.tick_params(axis='z', pad=10)
        # No subplot title for paper figures
    
    plt.subplots_adjust(left=0.02, right=0.98, wspace=0.25)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    output_path_svg = output_path.with_suffix('.svg')
    plt.savefig(output_path_svg, bbox_inches='tight')
    print(f"Saved plot to {output_path} and {output_path_svg}")
    plt.close()


def save_results_to_csv(results, output_path):
    """Save results to CSV."""
    rows = []
    
    for pair_key, res in results.items():
        angles = res['angles']
        distances = res['distances']
        ax1 = res['axis1']
        ax2 = res['axis2']
        
        for i, a1 in enumerate(angles):
            for j, a2 in enumerate(angles):
                rows.append({
                    'pair': pair_key,
                    'axis1': ax1,
                    'axis2': ax2,
                    f'{ax1}_angle': a1,
                    f'{ax2}_angle': a2,
                    'latent_distance': distances[i, j]
                })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved CSV to {output_path}")


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description='PointMAE 2D Rotation Sweep',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Using config file
  python eval_rotation_sweep.py --config configs/eval_pointmae.yaml
  
  # Legacy CLI arguments
  python eval_rotation_sweep.py --checkpoint ckpt.pth --objects 077_rubiks_cube \\
      --ycb-root ../E2PN/ycb --output-root results
        """)
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint (overrides config)')
    parser.add_argument('--objects', type=str, nargs='+', default=None, help='YCB objects (overrides config)')
    parser.add_argument('--ycb-root', type=str, default=None, help='YCB dataset path (overrides config)')
    parser.add_argument('--output-root', type=str, default=None, help='Output directory (overrides config)')
    parser.add_argument('--device', type=str, default=None, help='Device: cuda or cpu (overrides config)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (overrides config)')
    parser.add_argument('--input-num', type=int, default=None, help='Number of points (overrides config)')
    parser.add_argument('--angle-step', type=float, default=None, help='Angle step in degrees (overrides config)')
    parser.add_argument('--max-angle', type=float, default=None, help='Max angle in degrees (overrides config)')
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
        
        angle_step = args.angle_step if args.angle_step is not None else config['evaluation']['angle_step']
        max_angle = args.max_angle if args.max_angle is not None else config['evaluation']['max_angle']
        seed = args.seed if args.seed is not None else config['evaluation']['seed']
        device = args.device if args.device else config['evaluation']['device']
        
        output_root = args.output_root if args.output_root else config['output']['dir']
    else:
        # Legacy mode: all args must be provided via CLI
        if not args.checkpoint or not args.objects or not args.output_root:
            parser.error('--checkpoint, --objects, and --output-root are required when not using --config')
        
        ycb_root = args.ycb_root if args.ycb_root else '../E2PN/ycb'
        use_all = False
        objects_list = args.objects
        checkpoint = args.checkpoint
        input_num = args.input_num if args.input_num is not None else 4096
        noise_std = args.noise_std if args.noise_std is not None else 0.0
        shuffle = args.shuffle if args.shuffle is not None else False
        angle_step = args.angle_step if args.angle_step is not None else 10.0
        max_angle = args.max_angle if args.max_angle is not None else 180.0
        seed = args.seed if args.seed is not None else 42
        device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
        output_root = args.output_root
    
    # Path handling
    if not os.path.isabs(ycb_root):
        ycb_root = os.path.join(os.path.dirname(__file__), ycb_root)
    if not os.path.isabs(checkpoint):
        checkpoint = os.path.join(os.path.dirname(__file__), checkpoint)
    
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
    from utils.config import cfg_from_yaml_file
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
    
    # Process objects
    print(f"\nProcessing {len(objects_to_process)} object(s)...")
    
    for idx, obj_name in enumerate(objects_to_process):
        print(f"\n[{idx+1}/{len(objects_to_process)}] Processing {obj_name}...")
        
        try:
            points_base = load_ycb_object(ycb_root, obj_name, 
                                           input_num=input_num, seed=seed)
            
            rng = np.random.RandomState(seed)
            q_base, t_base = generate_random_pose(rng=rng)
            
            print(f"Sweeping rotations (max_angle={max_angle}, step={angle_step})...")
            print(f"Augmentations: noise_std={noise_std}, shuffle={shuffle}, unit_sphere_norm=True (always)")
            
            rot_results = sweep_joint_rotations(
                model, points_base, q_base, t_base, device,
                angle_step=angle_step, max_angle=max_angle,
                noise_std=noise_std, shuffle=shuffle
            )
            
            output_dir = Path(output_root) / f'rotation_sweep_{obj_name}'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            plot_path = output_dir / f"pointmae_{obj_name}.png"
            plot_3d_results(rot_results, plot_path, f"PointMAE 2D Rotation Sweep: {obj_name}")
            
            csv_path = output_dir / f"rotation_sweep_{obj_name}.csv"
            save_results_to_csv(rot_results, csv_path)
            
        except Exception as e:
            print(f"Error processing {obj_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("Done!")


if __name__ == '__main__':
    main()
