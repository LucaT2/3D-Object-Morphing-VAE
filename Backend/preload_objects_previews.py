import os
import json
import numpy as np
import imageio
import pyvista as pv
from collections import defaultdict

def preload_meshes_and_voxels(mesh_base_dir="Dataset_Storage/",
                              voxel_train_dir="Dataset_Storage/voxelized-modelnet10-trainset",
                                voxel_test_dir="Dataset_Storage/voxelized-modelnet10-testset",
                               files_from_train = 10,
                                 files_from_test = 10):
    """
    Scans a directory to find and pair mesh (.obj) and voxel (.npy) files.
    """
    temp_data = defaultdict(lambda: {'train': [], 'test': []})
    
    # Walk through the mesh directory to get categories and file names
    for root, _, files in os.walk(mesh_base_dir):
        for file in files:
            if file.endswith((".off", ".obj")): 
                base_name = os.path.splitext(file)[0]
                category = os.path.basename(os.path.dirname(root))
                mesh_path = os.path.join(root, file)
                
                is_train = 'train' in root
                is_test = 'test' in root
                
                if is_train:
                    voxel_path = os.path.join(voxel_train_dir, base_name + ".npy")
                    if os.path.exists(voxel_path):
                        temp_data[category]['train'].append((mesh_path, voxel_path))
                elif is_test:
                    voxel_path = os.path.join(voxel_test_dir, base_name + ".npy")
                    if os.path.exists(voxel_path):
                        temp_data[category]['test'].append((mesh_path, voxel_path))

    organized_data = defaultdict(list)
    for category, sets in temp_data.items():
        train_files = sets['train'][:files_from_train]
        test_files = sets['test'][:files_from_test]
        organized_data[category].extend(train_files)
        organized_data[category].extend(test_files)
        
    return dict(organized_data)


print("Starting mesh preview generation...")

CACHE_DIR = "Frontend/previews_cache"
METADATA_FILE = os.path.join(CACHE_DIR, "previews_metadata.json")
os.makedirs(CACHE_DIR, exist_ok=True)


preloaded_data = preload_meshes_and_voxels() 
category_metadata = {}

for category, items in preloaded_data.items():
    print(f"Processing category: {category}...")
    previews_metadata = []
    
    for mesh_path, voxel_path in items: 
        base_name = os.path.basename(voxel_path).replace('.npy', '.png')
        image_cache_path = os.path.join(CACHE_DIR, base_name)
        
        if not os.path.exists(image_cache_path):
            print(f"  -> Generating preview for {base_name}...")
            plotter = pv.Plotter(off_screen=True, window_size=[400, 400])
            mesh = pv.read(mesh_path)
            
            plotter.add_mesh(mesh, color='lightblue', smooth_shading=True, specular=0.5, ambient=0.3)
            plotter.view_isometric()
            plotter.enable_parallel_projection() 
            
            img_array = plotter.screenshot(transparent_background=True)
            plotter.close() 
            
            imageio.imwrite(image_cache_path, img_array)
        
        previews_metadata.append({
            "name": os.path.basename(voxel_path),
            "path": voxel_path, 
            "image_cache_path": image_cache_path
        })
        
    category_metadata[category] = previews_metadata

with open(METADATA_FILE, 'w') as f:
    json.dump(category_metadata, f, indent=4)

print(f"Mesh previews saved to '{CACHE_DIR}'.")