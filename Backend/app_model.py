import tensorflow as tf


import VAE_Model.Hyperparameters as hp
from VAE_Model.build_vae.build_model import VAE
import numpy as np

class trained_model():
    def __init__(self, weights_file):
        self.weights_file = weights_file
        self.vae_test_model = None
    
    def initialize_model(self):
        self.vae_test_model = VAE(hp.INPUT_DIM, hp.LATENT_DIM, hp.RESHAPE_DIM, hp.BETA, hp.L2_WEIGTH)
        self.vae_test_model.build(input_shape = hp.BUILD_INPUT_SHAPE)
        #self.vae_test_model.summary()
        self.vae_test_model.load_weights(self.weights_file)
        self.vae_test_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-3))
    
    def reconstruct_given_object(self, object):
        z_random = self.vae_test_model.encoder.predict(object)[2]
        reconstructed_voxel_grid = self.vae_test_model.decoder.predict(z_random)
        return reconstructed_voxel_grid
    
    def get_interpolation(self, file_path_1, file_path_2, num_steps = 10):
        """
        Generates a sequence of morphed voxel grids between two objects.
        Returns: A list of voxel grids, starting with model 1,
                            ending with model 2, including all interpolated steps.
        """
        model_a = np.expand_dims(np.load(file_path_1), axis=(0, -1))
        model_b = np.expand_dims(np.load(file_path_2), axis=(0, -1))

        # Get the mean of the latent vectors
        z_a = self.vae_test_model.encoder.predict(model_a, verbose=0)[0]
        z_b = self.vae_test_model.encoder.predict(model_b, verbose=0)[0]

        # Linearly interpolate between the two latent vectors
        interpolated_vectors = np.array([z_a + (z_b - z_a) * t for t in np.linspace(0, 1, num_steps)])
        interpolated_vectors = np.squeeze(interpolated_vectors, axis=1)

        # Generate the predictions
        morphed_voxels = self.vae_test_model.decoder.predict(interpolated_vectors, verbose=0)
        
        all_frames = [np.squeeze(model_a)] + list(morphed_voxels) + [np.squeeze(model_b)]
        
        return all_frames