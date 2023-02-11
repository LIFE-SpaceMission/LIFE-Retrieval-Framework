# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import normflows as nf
import numpy as np
import onnxruntime
import torch


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------


def load_flow_from_state_dict(file_path):
    """
    Auxiliary function to create a normalizing flow with a particular
    architecture and then load the pre-trained weights.
    """

    # Define hyperparameters of the flow
    latent_size = 2
    hidden_units = 32
    hidden_layers = 2
    num_layers = 8

    # Construct the flow
    flows = []
    for i in range(num_layers):
        flows += [
            nf.flows.AutoregressiveRationalQuadraticSpline(
                latent_size, hidden_layers, hidden_units
            )
        ]
        flows += [nf.flows.LULinearPermute(latent_size)]

    # Define Gaussian base distribution
    base = nf.distributions.DiagGaussian(latent_size, trainable=False)

    # Construct flow
    flow = nf.NormalizingFlow(base, flows)

    # Load weights from given state dict file
    flow.load_state_dict(
        torch.load(
            f=file_path,
            map_location=torch.device("cpu"),
        )
    )

    return flow


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

    def get_temperatures(
        self,
        z: np.ndarray,
        log_p: np.ndarray,
    ) -> np.ndarray:
        # Construct inputs to model
        log_P_in = torch.from_numpy(log_p.reshape(-1, 1)).float()
        z_in = torch.from_numpy(np.tile(A=z, reps=(log_p.shape[0], 1))).float()

        # Send through the model
        with torch.no_grad():
            T = self.model.forward(z=z_in, log_P=log_P_in).numpy()

        return np.asarray(np.atleast_1d(T.squeeze()))


class VAE_PT_Model:
    """
    Wrapper class for loading a decoder model from a (ONNX) file that
    provides an intuitive interface for running the model.
    """

    def __init__(self, path_or_bytes):
        # Load the (decoder) model from the given file path
        self.model = ONNXDecoder(path_or_bytes)

        # Get the latent size from the model
        self.latent_size = self.model.session.get_inputs()[0].shape[1]

    def get_temperatures(self, z, log_p):
        # Ensure that the input arrays have the correct shape
        z = np.atleast_2d(z)
        log_p = np.atleast_2d(log_p)

        # Run some sanity checks on the shapes
        if not z.shape[1] == self.latent_size:
            raise ValueError(f"z must be {self.latent_size}-dimensional!")
        if not z.shape[0] == log_p.shape[0]:
            raise ValueError("Batch size of z and log_P must match!")

        # Send through the model
        T = self.model(z=z, log_P=log_p)

        return np.asarray(np.atleast_1d(T.squeeze()))


class VAE_PT_Model_Flow:
    def __init__(
        self,
        path_or_bytes,
        flow_path=None,
        log_P_min=None,
        log_P_max=None,
        T_min=0,
        T_max=None,
    ):
        # Store constructor arguments
        self.log_P_min = log_P_min
        self.log_P_max = log_P_max
        self.T_min = T_min
        self.T_max = T_max

        # Load the (decoder) model from the given file path
        self.model = ONNXDecoder(path_or_bytes)

        # Get the latent size from the model
        self.latent_size = self.model.session.get_inputs()[0].shape[1]

        # Load the normalizing flow
        if flow_path is not None:
            self.flow = load_flow_from_state_dict(file_path=flow_path)
        else:
            self.flow = None

    def transform_z(self, z):
        """
        Auxiliary method to apply the flow to a given z (which is
        assumed to be normally distributed).
        """

        # If no flow is defined, simply return z as is
        if self.flow is None:
            return z

        # Otherwise, send z through all the layers of the flow
        with torch.no_grad():
            z = torch.from_numpy(z).float()
            for layer in self.flow.flows:
                z, _ = layer(z)
            z = z.numpy()

        return z

    def get_temperatures(self, z, log_p):
        # Ensure that the input arrays have the correct shape
        z = np.atleast_2d(z)
        log_P = np.atleast_2d(log_p)

        # Run some sanity checks on the shapes
        if not z.shape[1] == self.latent_size:
            raise ValueError(f"z must be {self.latent_size}-dimensional!")
        if not z.shape[0] == log_P.shape[0]:
            raise ValueError("Batch size of z and log_P must match!")

        # Clip the log_P values if necessary
        if self.log_P_min is not None:
            log_P = np.clip(log_P, self.log_P_min, None)
        if self.log_P_max is not None:
            log_P = np.clip(log_P, None, self.log_P_max)

        # Transform z
        z = self.transform_z(z=z)

        # Send through the model
        T = self.model(z=z, log_P=log_P)

        # Clip the temperature values if necessary
        if self.T_min is not None:
            T = np.clip(T, self.T_min, None)
        if self.T_max is not None:
            T = np.clip(T, None, self.T_max)

        return np.asarray(np.atleast_1d(T.squeeze()))


class ONNXDecoder:
    """
    A thin wrapper around ``onnxruntime.InferenceSession`` that can
    load an ONNX model from a file path or a byte string and provides
    a simple ``__call__`` method that can be used to run the model.
    """

    def __init__(self, path_or_bytes, n_threads=1):
        # Define the session options
        sess_options = onnxruntime.SessionOptions()
        sess_options.intra_op_num_threads = n_threads

        # Cast the path to string if necessary
        if isinstance(path_or_bytes, Path):
            path_or_bytes = path_or_bytes.as_posix()

        # Create a session for the ONNX model
        self.session = onnxruntime.InferenceSession(
            path_or_bytes=path_or_bytes,
            sess_options=sess_options,
        )

    def __call__(self, z, log_P):
        # Run the model
        inputs = {
            "z": z.astype(np.float32),
            "log_P": log_P.astype(np.float32),
        }
        outputs = self.session.run(None, inputs)

        return np.asarray(outputs[0])
