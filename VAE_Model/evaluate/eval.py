# Libraries imports
import tensorflow as tf
import numpy as np
import os

# This project's file imports
from VAE_Model.build_vae.build_model import VAE
from VAE_Model.Preprocess.VoxelizedDataset import VoxelizedDataset
from VAE_Model.evaluate.eval_utils import calculate_reconstruction_similarity, plot_voxels
import VAE_Model.Hyperparameters as hp

class Eval_VAE():
    def __init__(self, test_dir, weights_file):
        self.test_dir = test_dir
        self.weights_file = weights_file
        self.vae_test_model = None

        all_test_files = os.listdir(test_dir)
        self.full_file_paths = [os.path.join(test_dir, f) for f in all_test_files]
        self.test_generator = VoxelizedDataset(self.full_file_paths, hp.BATCH_SIZE, augment = False)

    def initialize_model(self):
        self.vae_test_model = VAE(hp.INPUT_DIM, hp.LATENT_DIM, hp.RESHAPE_DIM, hp.BETA, hp.L2_WEIGTH)
        self.vae_test_model.build(input_shape = hp.BUILD_INPUT_SHAPE)
        self.vae_test_model.load_weights(self.weights_file)
        self.vae_test_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-3))

    def evaluate_model(self):
        print("Evaluating model on the test dataset \n")
        test_metrics = self.vae_test_model.evaluate(self.test_generator, verbose = 1)
        print("Test Results: ")
        print(f"Test Total Loss: {test_metrics[0]}")
        print(f"Test Reconstruction Loss: {test_metrics[1]}")
        print(f"Test KL Loss: {test_metrics[2]}")
        original_voxels, _ = next(iter(self.test_generator))
        reconstructed_voxels = self.vae_test_model.predict(original_voxels)
        avg_iou = calculate_reconstruction_similarity(original_voxels, reconstructed_voxels)
        print(f"Average IOU is: {avg_iou}%")
    
    def reconstruct_one_object(self):
        random_number = np.random.randint(100) + 100
        print(self.full_file_paths[random_number])

        random_model = np.expand_dims(np.load(self.full_file_paths[random_number]), axis=0) 
        random_model = np.expand_dims(random_model, axis=-1)
        plot_voxels(np.squeeze(random_model), title="Original Random Model")

        #z_random_mean = recreated_vae.encoder.predict(random_model)[0]
        z_random = self.vae_test_model.encoder.predict(random_model)[2]



        reconstructed_voxel_grid = self.vae_test_model.decoder.predict(z_random)
        plot_voxels(np.squeeze(reconstructed_voxel_grid), title="Reconstruction from random latent sample")
