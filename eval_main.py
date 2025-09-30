import tensorflow as tf
import os

import VAE_Model.Hyperparameters as hp
from VAE_Model.evaluate.eval import Eval_VAE


def test_eval():
    weights = r'VAE_Model\Weights\best_vae_model.weights.h5'

    eval = Eval_VAE(r'Dataset_Storage\voxelized-modelnet10-testset', weights_file = weights)
    eval.initialize_model()
    eval.evaluate_model()

if __name__ == "__main__":
    test_eval()
