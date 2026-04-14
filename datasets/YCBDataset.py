import os
import torch
import numpy as np
import torch.utils.data as data
import trimesh
from .build import DATASETS
from utils.logger import *


@DATASETS.register_module()
class YCB(data.Dataset):
    """YCB dataset for Point-MAE masked reconstruction.
    
    Loads point clouds from YCB object meshes.
    """
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.npoints = config.N_POINTS
        self.sample_points_num = config.npoints
        self.subset = config.subset
        
        # Find all object directories
        self.object_dirs = []
        for obj_name in sorted(os.listdir(self.data_root)):
            obj_path = os.path.join(self.data_root, obj_name)
            if os.path.isdir(obj_path) and not obj_name.startswith('.'):
                # Check if mesh file exists
                mesh_path = os.path.join(obj_path, 'google_16k', 'nontextured.ply')
                if not os.path.exists(mesh_path):
                    mesh_path = os.path.join(obj_path, 'google_16k', 'textured.obj')
                if os.path.exists(mesh_path):
                    self.object_dirs.append({
                        'taxonomy_id': obj_name.split('_')[0],
                        'model_id': obj_name,
                        'mesh_path': mesh_path
                    })
        
        # Simple train/val split (80/20)
        n_objects = len(self.object_dirs)
        split_idx = int(0.8 * n_objects)
        
        if self.subset == 'train':
            self.object_dirs = self.object_dirs[:split_idx]
        else:  # test/val
            self.object_dirs = self.object_dirs[split_idx:]
        
        print_log(f'[DATASET] YCB {self.subset}: {len(self.object_dirs)} objects loaded', logger='YCB')
        print_log(f'[DATASET] Sample {self.sample_points_num} points per object', logger='YCB')
        
        self.permutation = np.arange(self.npoints)
        
        # Cache loaded meshes for efficiency
        self._mesh_cache = {}
    
    def _load_mesh(self, mesh_path):
        """Load mesh and cache it."""
        if mesh_path not in self._mesh_cache:
            mesh = trimesh.load(mesh_path, force='mesh')
            self._mesh_cache[mesh_path] = mesh
        return self._mesh_cache[mesh_path]
    
    def pc_norm(self, pc):
        """Normalize point cloud to unit sphere. pc: NxC, return NxC"""
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
    
    def random_sample(self, pc, num):
        """Randomly sample num points from pc."""
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc
    
    def __getitem__(self, idx):
        sample = self.object_dirs[idx]
        
        # Load mesh and sample points
        mesh = self._load_mesh(sample['mesh_path'])
        
        # Sample points from mesh surface
        points, _ = trimesh.sample.sample_surface(mesh, self.npoints)
        points = points.astype(np.float32)
        
        # Random sample to get exact number of points
        points = self.random_sample(points, self.sample_points_num)
        
        # Normalize to unit sphere
        points = self.pc_norm(points)
        
        points = torch.from_numpy(points).float()
        return sample['taxonomy_id'], sample['model_id'], points
    
    def __len__(self):
        return len(self.object_dirs)
