import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2

from VAE_Model.build_vae.build_utils import Sampling, scaled_sigmoid_activation, weighted_bce_loss
from .. import Hyperparameters as hp

class VAE(Model):
    def __init__(self, input_dim, latent_dim, reshape_dim, beta, l2_weight, **kwargs):
        super().__init__(**kwargs)
        # Hyperparameters
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.reshape_dim = reshape_dim
        self.beta = beta
        self.l2_weight = l2_weight

        # Building the encoder + decoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        # Tracking losses
        self.total_loss_tracker = tf.keras.metrics.Mean(name = 'total_loss')
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name = 'reconstruction_loss')
        self.kl_loss_tracker = tf.keras.metrics.Mean(name = 'kl_loss')

    def call(self, inputs):
        """Defines the forward pass of the model."""
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction
    
    

    def build_encoder(self):
        encoder_inputs = layers.Input(shape = self.input_dim)
        # Downsampling by using strides = 2
        x = layers.Conv3D(32, 3, activation = 'elu', strides = 2, padding = 'same', kernel_regularizer = l2(self.l2_weight))(encoder_inputs)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv3D(64, 3, activation = 'elu', strides = 2, padding = 'same', kernel_regularizer = l2(self.l2_weight))(x)
        x = layers.BatchNormalization()(x)
        
        # Residual Block
        residual = x
        x = layers.Conv3D(128, 3, activation='elu', strides=2, padding='same',kernel_regularizer=l2(self.l2_weight))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv3D(128, 3, activation='elu', padding='same',kernel_regularizer=l2(self.l2_weight))(x)
        x = layers.BatchNormalization()(x)
        
        # Shortcut connection for the resnet architecture
        residual_downsampled = layers.Conv3D(128, 1, strides=2, padding='same')(residual)
        x = layers.add([x,residual_downsampled])
        
        # Flatten and map to the latent space
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation = 'elu')(x)

        # Obtaining the mean and log variance
        z_mean = layers.Dense(self.latent_dim, name = 'z_mean', kernel_regularizer = l2(self.l2_weight))(x)
        z_log_var = layers.Dense(self.latent_dim, name = 'z_log_var', kernel_regularizer = l2(self.l2_weight))(x)

        z = Sampling()([z_mean,z_log_var])

        encoder = tf.keras.Model(encoder_inputs, [z_mean,z_log_var,z], name = 'encoder')
        return encoder
    def build_decoder(self):
        latent_inputs = tf.keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(256, activation='elu')(latent_inputs)

        #Upsample from the latent vector to the small grid necessary for the decoder
        x = layers.Dense(self.reshape_dim[0] * self.reshape_dim[1] * self.reshape_dim[2] * 128,\
                          activation="elu")(x)
        x = layers.Reshape(self.reshape_dim)(x)

        
        # The transposed residual block
        residual = x
        x = layers.Conv3DTranspose(128, 3, activation='elu', padding='same',
                                kernel_regularizer=l2(self.l2_weight))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv3DTranspose(128, 3, activation='elu', strides=2, padding='same',
                                kernel_regularizer=l2(self.l2_weight))(x)
        x = layers.BatchNormalization()(x)

        # Shortcut connection
        residual = layers.Conv3DTranspose(128, 1, strides=2, padding='same')(residual)
        x = layers.add([x, residual])

        # Continue the upsampling
        x = layers.Conv3DTranspose(64, 3, activation = 'elu', strides = 2, padding = 'same', 
                                   kernel_regularizer = l2(self.l2_weight))(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv3DTranspose(32, 3, activation = 'elu', strides = 2, padding = 'same', 
                                   kernel_regularizer = l2(self.l2_weight))(x)
        x = layers.BatchNormalization()(x)

        # This is the final layer that reconstructs the voxel grid
        decoder_outputs = layers.Conv3DTranspose(1, 3, activation=scaled_sigmoid_activation, padding="same", 
                                                 kernel_regularizer = l2(self.l2_weight))(x)
        
        decoder = tf.keras.Model(latent_inputs, decoder_outputs, name = 'decoder')
        return decoder


    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            x, targets = data
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    weighted_bce_loss(targets, reconstruction, gamma = hp.GAMMA),
                    axis=(1, 2,3),
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + self.beta * kl_loss + tf.reduce_sum(self.losses)
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    

    def test_step(self, data):
        """Defines the logic for one evaluation step."""
        x, targets = data
        
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                    weighted_bce_loss(targets, reconstruction, gamma = hp.GAMMA),
                axis=(1, 2, 3),
            )
        )
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        
        total_loss = reconstruction_loss + self.beta * kl_loss + tf.reduce_sum(self.losses)
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {m.name: m.result() for m in self.metrics}