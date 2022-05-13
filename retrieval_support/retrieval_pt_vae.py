from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import numpy as np
import torch
import time

# Define a wrapper class to load the PyTorch model and perform the
# linear interpolation to the target grid

class VAE_PT_Model_old:
    
    def __init__(self, file_path):
        self.model = torch.jit.load(file_path)
    
    @property
    def min_t(self):
        return float(min(self.model.pressure_grid))
    
    @property
    def max_t(self):
        return float(max(self.model.pressure_grid))


    def get_temperatures(self, z: np.ndarray, p: np.ndarray):
        
        temperature_grid = self.model(torch.Tensor(z))
        
        interpolator = interp1d(
            x=self.model.pressure_grid.numpy(),
            y=temperature_grid.numpy(),
        )
        
        return interpolator(p)

class VAE_PT_Model:
    
    def __init__(self, file_path):
        self.model = torch.jit.load(file_path)
        self.latent_size = self.model.latent_size

    def get_temperatures(self,z: np.ndarray,log_p: np.ndarray,) -> np.ndarray:
        """
        Take a latent code `z` that represents a PT profile, and
        evaluate at the (log)-pressure values given in `log_p`.
        """
        # Set model to test model -- this is important!
        self.model.eval()
        with torch.no_grad():
            # Compute the forward pass through the decoder
            #print(z)
            #print(log_p)
            t_pred = self.model(z=torch.Tensor(z), log_p_grid=torch.Tensor(log_p),) 

            # Re-apply normalization
            t_pred *= self.model.train_std
            t_pred += self.model.train_mean
        
        return t_pred.numpy().T[0]