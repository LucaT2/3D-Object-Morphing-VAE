import numpy as np
import tensorflow as tf
import os
import glob
import random
from Backend.app_model import trained_model
model_instance = None

def initialize_model():
    global model_instance
    if model_instance is None:
        model_instance = trained_model(weights_file = r'VAE_Model/Weights/best_vae_model.weights.h5')
        model_instance.initialize_model()
    return model_instance

def warm_up_model(model):
    """
    Runs a dummy prediction with the default batch size to
    trigger compilation for the model in order for the interpolation
    and reconstruction to have a lower compute time.
    """
    print("Warming up the model for interpolation...")
    
    # Default value of your Gradio slider.
    warmup_batch_size = 15 
    latent_dim = 128 
    
    dummy_latent_vectors = np.zeros((warmup_batch_size, latent_dim), dtype=np.float32)
    reconstruction_dummy = np.zeros((1,latent_dim), dtype=np.float32)
    # Running two dummy predictions through the decoder.
    _ = model.vae_test_model.decoder.predict(dummy_latent_vectors, verbose=0)
    _ = model.vae_test_model.decoder.predict(reconstruction_dummy, verbose=0)
    
    print("Model is warmed up and ready!")

def preload_objects(data_dir = r"Dataset_Storage/voxelized-modelnet10-trainset"):
    objects = {}
    categories = ['bathtub','chair', 'sofa', 'table','desk',
                  'toilet','dresser','bed','night_stand', 'monitor']
    for category in categories:
        pattern = os.path.join(data_dir, f"{category}_*.npy")
        files = glob.glob(pattern)
        selected_files = random.sample(files, min(10, len(files)))
        objects[category] = []
        for file_path in selected_files:
            voxel_data = np.load(file_path)
            voxel_data = np.expand_dims(voxel_data, axis=(0, -1))
            objects[category].append((file_path, voxel_data))
    return objects