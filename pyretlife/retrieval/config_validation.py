def validate_config(config_file: dict):
    raise NotImplementedError


### OLD FUNCTION TO BE SPLIT UP
#
# def check_temperature_parameters(config: dict) -> None:
#     """
#     This function checks if all temperature variables necessary
#     for the given parametrization are provided by the user. If not,
#     it stops the run.
#     """
#
#     # TODO: I would recommend to restructure the configuration of the
#     #   temperature parameters in a more general and structured way,
#     #   for example something like this:
#     #   ```
#     #   TEMPERATURE PARAMETERS:
#     #     parametrization: polynomial
#     #     parameters:
#     #       a_4 = U 2 5 T 3.67756393
#     #       a_3 = U 0 100 T 40.08733884
#     #       a_2 = U 0 300 T 136.42147966
#     #       a_1 = U 0 500 T 182.6557084
#     #       a_0 = U 0 600 T 292.92802205
#     #     extra_parameters:
#     #       dim_z: dimensionality of the latent space
#     #       file_path: /path/to/learned/PT/model
#     #       ...
#     #       (other parameters that are specific to the parametrization)
#     #  ```
#     #  This would make it easier to check if all the necessary parameters
#     #  for the given parametrization are provided, and it would also make
#     #  it easier to add new parametrizations in the future.
#
#     input_pt = list(config["TEMPERATURE PARAMETERS"].keys())
#
#     # check if all parameters are there:
#     if (
#         config["TEMPERATURE PARAMETERS"]["settings_parametrization"]
#         == "polynomial"
#     ):
#         required_params = ["a_" + str(i) for i in range(len(input_pt) - 1)]
#     elif "vae_pt" in config["TEMPERATURE PARAMETERS"]["parametrization"]:
#         required_params = [
#             "z_" + str(i + 1)
#             for i in range(
#                 len(
#                     [
#                         input_pt[i]
#                         for i in range(len(input_pt))
#                         if "settings" not in input_pt[i]
#                     ]
#                 )
#                 - 2
#             )
#         ]
#     elif config["TEMPERATURE PARAMETERS"]["parametrization"] == "guillot":
#         required_params = [
#             "log_delta",
#             "log_gamma",
#             "t_int",
#             "t_equ",
#             "log_p_trans",
#             "alpha",
#         ]
#     elif config["TEMPERATURE PARAMETERS"]["parametrization"] == "madhuseager":
#         required_params = [
#             "T0",
#             "log_P1",
#             "log_P2",
#             "log_P3",
#             "alpha1",
#             "alpha2",
#         ]
#     elif (
#         config["TEMPERATURE PARAMETERS"]["parametrization"]
#         == "mod_madhuseager"
#     ):
#         required_params = ["T0", "log_P1", "log_P2", "alpha1", "alpha2"]
#     elif config["TEMPERATURE PARAMETERS"]["parametrization"] == "isothermal":
#         required_params = ["T_eq"]
#     elif config["TEMPERATURE PARAMETERS"]["parametrization"] == "input":
#         required_params = ["input_path"]
#     else:
#         raise RuntimeError("Unknown PT parametrization.")
#
#     if not all(elem in input_pt for elem in required_params):
#         missing_params = [_ for _ in required_params if _ not in input_pt]
#         raise RuntimeError(
#             "Missing one or more PT parameters/knowns. "
#             "Make sure these exist:" + str(missing_params)
#         )
