import tensorflow as tf


import VAE_Model.Hyperparameters as hp
from VAE_Model.build_vae.build_model import VAE

class trained_model():
    def __init__(self, weights_file):
        self.weights_file = weights_file
        self.model = None
    
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