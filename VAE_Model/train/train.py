# Library imports
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

# Files from this project imports
from VAE_Model.Preprocess.VoxelizedDataset import VoxelizedDataset
from VAE_Model.train.train_utils import WarmupCosineDecay, PrintLR
from VAE_Model.build_vae.build_model import VAE
import VAE_Model.Hyperparameters as hp

class Train_VAE():
    def __init__(self,num_epochs, num_warmup_epochs, batch_size, train_dir, weights_file):
        self.num_epochs = num_epochs
        self.num_warmup_epochs = num_warmup_epochs
        self.batch_size = batch_size
        self.train_dir = train_dir
        self.train_generator = None
        self.val_generator = None
        self.weights_file = weights_file
        self.history = None
        self.vae_model = None

    def configure_loaders(self):
        all_processed_files = os.listdir(self.train_dir)

        categories = [f.split('_')[0] for f in all_processed_files]

        train_processed, val_processed = train_test_split(
            all_processed_files,
            test_size=0.2,
            random_state=42,
            stratify = categories
        )
        train_full_paths = [os.path.join(self.train_dir, f) for f in train_processed]
        val_full_paths = [os.path.join(self.train_dir, f) for f in val_processed]
        
        self.train_generator = VoxelizedDataset(train_full_paths, batch_size=hp.BATCH_SIZE, augment=True)
        self.val_generator = VoxelizedDataset(val_full_paths, batch_size=hp.BATCH_SIZE, augment=False)

    def define_steps(self):
        steps_per_epoch = len(self.train_generator)
        total_steps = self.num_epochs * steps_per_epoch
        warmup_steps = self.num_warmup_epochs * steps_per_epoch
        decay_steps = total_steps - warmup_steps
        return warmup_steps, decay_steps

    def configure_optimizer(self):
        warmup_steps, decay_steps = self.define_steps()
        lr_schedule = WarmupCosineDecay(
            initial_learning_rate=0.001,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            alpha=0.01
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        return optimizer

    def configure_training_environment(self):
        """
        Initialize the necessary checkpoints and callbacks for the model to train properly
        and for you to be able to visualize what is happening
        """


        model_checkpoint_callback = ModelCheckpoint(
            filepath= self.weights_file,  
            save_weights_only=True,              
            monitor='val_total_loss',                 
            mode='min',                           
            save_best_only=True,                 
            verbose=1                           
        )
        early_stopping_callback = EarlyStopping(
            monitor='val_total_loss', 
            patience=10,              
            mode='min',               
            verbose=1,               
            restore_best_weights=True
        )

        print_lr_callback = PrintLR()

        return model_checkpoint_callback, early_stopping_callback, print_lr_callback
    
    def configure_vae(self):
        self.vae_model = VAE(hp.INPUT_DIM, hp.LATENT_DIM, hp.RESHAPE_DIM, hp.BETA, hp.L2_WEIGTH)
        self.vae_model.build(input_shape = hp.BUILD_INPUT_SHAPE)

    def train(self):
        self.configure_loaders()
        self.configure_vae()
        optimizer = self.configure_optimizer()
        model_checkpoint_callback,\
            early_stopping_callback, print_lr_callback = self.configure_training_environment()
        self.vae_model.compile(optimizer = optimizer)

        print("Starting the training phase...")

        self.history = self.vae_model.fit(
            self.train_generator,
            epochs=self.num_epochs,
            validation_data=self.val_generator,
            callbacks=[model_checkpoint_callback,
                    early_stopping_callback,
                    print_lr_callback]
        )

        print("Training finished!")
    
    def plot_history(self):

        print("Plotting training history...")

        loss = self.history.history['loss']
        val_loss =self.history.history['val_total_loss']
        epochs = range(1, len(loss) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, loss, 'bo-', label='Training Loss')
        plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

        print("Plot displayed.")

