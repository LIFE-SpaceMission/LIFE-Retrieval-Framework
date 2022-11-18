from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import numpy as np
import torch
import normflows
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

#class VAE_PT_Model:
#    
#    def __init__(self, file_path):
#        self.model = torch.jit.load(file_path)
#        self.latent_size = self.model.latent_size
#
#    def get_temperatures(self,z: np.ndarray,log_p: np.ndarray,) -> np.ndarray:
#        """
#        Take a latent code `z` that represents a PT profile, and
#        evaluate at the (log)-pressure values given in `log_p`.
#        """
#        # Set model to test model -- this is important!
#        self.model.eval()
#        with torch.no_grad():
#            # Compute the forward pass through the decoder
#            #print(z)
#            #print(log_p)
#            t_pred = self.model(z=torch.Tensor(z), log_p_grid=torch.Tensor(log_p),) 
#
#            # Re-apply normalization
#            t_pred *= self.model.train_std
#            t_pred += self.model.train_mean
#        
#        return t_pred.numpy().T[0]


class VAE_PT_Model_prev:
    """
    Wrapper class for using trained (decoder) models which represent
    a (parameterized) pressure-temperature profile.
    """

    def __init__(self, file_path):

        # Load the (decoder) model from the given file path
        self.model = torch.jit.load(file_path)  # type: ignore

        # Make some other properties available
        self.latent_size = self.model.latent_size
        self.T_offset = self.model.T_offset
        self.T_factor = self.model.T_factor

    def get_temperatures(self,z: np.ndarray,log_p: np.ndarray,) -> np.ndarray:

        # Construct inputs to model
        log_P_in = torch.from_numpy(log_p.reshape(-1, 1)).float()
        z_in = torch.from_numpy(np.tile(A=z, reps=(log_p.shape[0], 1))).float()

        # Send through the model
        with torch.no_grad():
            T = self.model.forward(z=z_in, log_P=log_P_in).numpy()

        return np.asarray(np.atleast_1d(T.squeeze()))


class VAE_PT_Model:
    def __init__(self,file_path,flow_file_path = None):

        # Load the (decoder) model from the given file path
        self.model = torch.jit.load(file_path)  # type: ignore

        # If a flow file path is given, load the flow
        self.flow = None
        if flow_file_path is not None:
            print(flow_file_path)
            self.flow = torch.load(flow_file_path)  # type: ignore
            print('flow')
        else:
            print('no flow')

        # Make some other properties available
        self.latent_size = self.model.latent_size
        self.T_offset = self.model.T_offset
        self.T_factor = self.model.T_factor

    def get_temperatures(self,z,log_p):

        # Make sure inputs have the right shape
        if not z.shape == (self.model.latent_size,):
            raise ValueError(f'z must be {self.model.latent_size}D!')
        if log_p.ndim != 1:
            raise ValueError('log_P must be 1D!')

        # Apply flow, if needed
        z_in = torch.from_numpy(z).float().unsqueeze(0)
        if self.flow is not None:
            with torch.no_grad():
                for layer in self.flow.flows:
                    z_in, _ = layer(z_in)
            
        # Construct inputs to model
        log_P_in = torch.from_numpy(log_p.reshape(-1, 1)).float()
        z_in = torch.tile(z_in, (log_p.shape[0], 1))

        # Send through the model
        with torch.no_grad():
            T = self.model.forward(z=z_in, log_P=log_P_in).numpy()

        return np.asarray(np.atleast_1d(T.squeeze()))