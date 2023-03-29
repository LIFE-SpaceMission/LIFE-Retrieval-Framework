import numpy as np
from numpy import ndarray


def validate_pt_profile(settings: dict, temp_vars: dict, phys_vars: dict) -> bool:
    # Check: Return -inf if pressures for madhuseager models are not
    # monotonically increasing.
    result = False
    if settings["parameterization"] == "madhuseager":
        if (
                not settings['log_top_pressure']
                    < temp_vars["log_P1"]
                    < temp_vars["log_P2"]
                    < temp_vars["log_P3"]
                    < phys_vars["log_P0"]
        ):
            result = True
    if settings["parameterization"] == "mod_madhuseager":
        if (
                not settings['log_top_pressure']
                    < temp_vars["log_P1"]
                    < temp_vars["log_P2"]
                    < phys_vars["log_P0"]
        ):
            result = True

    # Check: Return -inf if parameters for the Guillot model are bad
    if settings["parameterization"] == "guillot":
        if temp_vars['alpha'] < -1:
            result = True

    return result

def validate_sum_of_cube(cube:list) -> bool:

    if np.isfinite(cube).all() == True:
        return True

def validate_positive_temperatures(T:ndarray)-> bool:
    if any((T < 0).tolist()):
        return True

def validate_sum_of_abundances(chem_vars) -> bool:
    if sum(chem_vars.values()) > 1:
        return True
# def log_likelihood(self, cube):
#
#
#     # Check: Return -inf if the abundances exceed 1
#
#
#
#     # Calculate the forward model; this returns the wavelengths in cm
#     # and the flux F_nu in erg/cm^2/s/Hz.
#     self.retrieval_model_plain()
#
#     # Check: Return -inf if forward model returns NaN values.
#     if np.sum(np.isnan(self.rt_object.flux)) > 0:
#         print("NaN spectrum encountered")
#         return -1e32
#
#     # Calculate total log-likelihood (sum over instruments)
#     log_likelihood = 0.0
#     for inst in self.instrument.keys():
#         # Rebin the spectrum according to the input spectrum if wavelengths
#         # differ strongly
#         if not np.array(
#                 [
#                     (
#                             np.round(self.instrument[inst]["wl"], 10)
#                             == np.round(self.nc.c / self.rt_object.freq * 1e4, 10)
#                     )
#                 ]
#         ).all():
#             flux_temp = spectres.spectres(
#                 self.instrument[inst]["wl"],
#                 self.nc.c / self.rt_object.freq * 1e4,
#                 self.rt_object.flux,
#             )
#
#             # Rebin and add the moon Flux if present
#             if self.settings["moon"] == "True":
#                 flux_temp = flux_temp + spectres.spectres(
#                     self.instrument[inst]["wl"],
#                     self.nc.c / self.rt_object.freq * 1e4,
#                     self.moon_flux,
#                 )
#
#         # If no rebin is required
#         else:
#             flux_temp = self.rt_object.flux
#
#             # Rebin and add the moon Flux if present
#             if self.settings["moon"] == "True":
#                 flux_temp = flux_temp + self.moon_flux
#
#         # Calculate log-likelihood
#         log_likelihood += -0.5 * np.sum(
#             (
#                     (flux_temp - self.instrument[inst]["flux"])
#                     / self.instrument[inst]["error"]
#             )
#             ** 2.0
#         )
#
#     return log_likelihood
