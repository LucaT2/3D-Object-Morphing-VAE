import os
import numpy as np
import trimesh

class PreprocessDataset():
    def __init__(self, input_dir, train_output_dir, test_output_dir):
        self.input_dir = input_dir
        self.train_output_dir = train_output_dir
        self.test_output_dir = test_output_dir
    @staticmethod
    def voxelize(mesh , grid_size):
            # Takes a trimesh and converts it into a voxel grid of size 64x64x64

        # Get the bounding box of the mesh
        bounds = mesh.bounds
        bounding_box_size = bounds[1]- bounds[0]

        max_dimension = max(bounding_box_size)

        scale_factor = 1.0 / max_dimension
        # Center the mesh at the origin and scale it
        mesh.apply_translation(-mesh.centroid)
        mesh.apply_scale(scale_factor)

        # Voxelize the normalized mesh
        pitch = 1.0 / grid_size
        voxel_grid = mesh.voxelized(pitch = pitch)

        # The voxelized method will return an object of type VoxelGrid. We have to convert it into a dense
        # matrix and then force it to have the correct shape
        matrix = voxel_grid.matrix.astype(np.float32)

        # Create an empty grid of the correct size
        final_grid = np.zeros((grid_size,grid_size,grid_size), dtype = np.float32)

        min_dimensions = np.minimum(matrix.shape, [grid_size,grid_size,grid_size])

        # Copy the voxel data into the correctly sized grid
        final_grid[:min_dimensions[0], :min_dimensions[1], :min_dimensions[2]] = \
                                matrix[:min_dimensions[0], :min_dimensions[1], :min_dimensions[2]]
        return final_grid

    def voxelize_train_dataset(self):
        data_categories = os.listdir(self.input_dir)
        dir_exists = False
        if not os.path.exists(self.train_output_dir):
            dir_exists = True
            os.makedirs(self.train_output_dir)
        if len(os.listdir(self.train_output_dir)) < 3000:
            for category in data_categories:
                print(f"\nProcessing category: {category} ")
                category_train_path = os.path.join(self.input_dir, category, 'train')
                category_files = os.listdir(category_train_path)
                
                for i, filename in enumerate(category_files):
                    if i % 50 == 0:
                        print(f"Processing file {i+1}/{len(category_files)}: {filename}")
                    filename = os.path.join(category_train_path, filename)
                    # Create a new filename for the .npy file
                    basename = os.path.basename(filename)
                    new_filename = os.path.join(self.train_output_dir, basename.replace(".off", ".npy"))
                    if os.path.exists(new_filename):
                        continue            
                    # Voxelize and save
                    mesh = trimesh.load(filename)
                    voxels = self.voxelize(mesh, 64)
                    np.save(new_filename, voxels)
                    
        else:
            print(f"The preprocessed objects are already in the {self.test_output_dir} directory")

    def voxelize_test_dataset(self):
        data_categories = os.listdir(self.input_dir)
        dir_exists = False
        if not os.path.exists(self.test_output_dir):
            dir_exists = True
            os.makedirs(self.test_output_dir)
        if len(os.listdir(self.test_output_dir)) < 3000:
            for category in data_categories:
                print(f"\nProcessing category: {category} ")
                category_test_path = os.path.join(self.input_dir, category, 'test')
                category_files = os.listdir(category_test_path)
                
                for i, filename in enumerate(category_files):
                    if i % 50 == 0:
                        print(f"Processing file {i+1}/{len(category_files)}: {filename}")
                    filename = os.path.join(category_test_path, filename)
                    # Create a new filename for the .npy file
                    basename = os.path.basename(filename)
                    new_filename = os.path.join(self.train_output_dir, basename.replace(".off", ".npy"))
                    if os.path.exists(new_filename):
                        continue            
                    # Voxelize and save
                    mesh = trimesh.load(filename)
                    voxels = self.voxelize(mesh, 64)
                    np.save(new_filename, voxels)
                    
        else:
            print(f"The preprocessed objects are already in the {self.test_output_dir} directory")