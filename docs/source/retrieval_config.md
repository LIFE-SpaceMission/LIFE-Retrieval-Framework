
# Create a Configuration File

The retrieval reads a YAML config file to set up the run. This config file contains all the settings, the known values, and the parameters that the retrieval will need to use, or retrieve for, in a run. If some keywords are not specified, the retrieval falls back into reading the default config file `configs/config_default.yaml` which contains some standard values generally used in our previous retrievals. In the remainder of the text, we will show in red all the settings/keywords/parameters that must be specified in the config file; in blue the settings/keywords/parameters that have a default value in the `config_default.yaml` file; in green the settings/keywords/parameters that are optional and/or necessary only if some specific settings are turned on. 

These three categories need to follow specific rules to be recognized correctly by the script when reading the configuration file.

Any parameter that we want to retrieve within the retrieval needs to have the following structure:


## Parameters

Every parameter must have a specific parameter_name that is recognizable by the code. 
These parameter names are often determined by specific settings (i.e. scattering, or the p-t profile parameterization), see below for details.
The parameter unit is a nested dictionary that looks like this:  

```
  parameter_name:
    prior:
      kind: uniform/gaussian/log-uniform/log-gaussian/custom
      prior_specs:
        ...
    truth: [float]
    unit: [str]
```

The necessary keyword that allows the parameter unit to be identified as a parameter is prior. 
This is itself a dictionary that has the following keys:
- **kind [str]:** a string that specifies the type of priors to consider. 
The currently implemented priors are the ones specified in `pyretlife/priors.py`. These are:


  |Kind| Description|
  |:---| :----------|
  |uniform|A boxcar prior between a lower and upper limit.|
  |gaussian|A gaussian prior of known mean and sigma.|
  |log-uniform|A uniform prior on the logarithm (base 10) of the parameter.|
  |log-gaussian|A gaussian prior on the logarithm (base 10) of the parameter.|
  |custom|A user-defined prior.|


- **prior_specs [dict]**: the specific keywords required by the prior kind used. These are:


  |Kind| Prior_specs keywords|
  |:---| :-------------------|
  |uniform| **lower** [float]: the lower limit of the boxcar prior <br> **upper** [float]: the upper limit of the boxcar prior|
  |gaussian| **mean** [float]: the mean of the gaussian prior <br> **sigma** [float]: the sigma of the gaussian prior|
  |log-uniform|**log_lower** [float]: the lower limit of the boxcar prior (in logarithm space) <br> **log_upper** [float]: the upper limit of the boxcar prior (in logarithm space)|
  |log-gaussian| **log_mean** [float]: the mean of the gaussian prior (in logarithm space) <br> **log_sigma** [float]: the sigma of the gaussian prior (in logarithm space)|
  |custom|**prior_path** [str]: the path to a file containing the data used to produce a custom prior distribution (e.g. the posterior of a previous run)|



- **truth [float]**: If desired, it is possible to specify the truth value of the parameter. This will be used by the plotting routine.
- **unit [string]**: if the prior_specs and/or the truth values are not in the default units of the parameter, the unit needs to be specified as a string. 
If not specified, the retrieval routine will assume the default units.


## Known Values

Every known value must have a specific known_name that is recognizable by the code. 
These known names are often determined by specific settings (i.e. scattering, or the p-t profile parameterization), see below for details. 
The parameter unit is a nested dictionary that looks like this: 
 
```
  known_name:
    truth: [float]
    unit: [str]
```

The necessary keyword that allows the parameter unit to be identified as a parameter is **truth [float]**. 
This specifies the truth value of the parameter which will be used both during the retrieval run and during the plotting.
The optional keyword is **unit [string]**: if the truth value is not in the default units of the parameter, the unit needs to be specified as a string. If not specified, the retrieval routine will assume the default units.

## Settings

The retrieval considers every keyword that does not contain a nested “prior” dictionary nor a “truth” keyword as a setting. 
These can be both found as simple keywords associated to booleans or strings, e.g.:

```
include_CIA: True
parameterization: polynomial
```

## Sections

The config file needs to have specific sections for the run to work properly. 

- **RUN SETTINGS**: This section must contain all the parameters that are related to the settings of the specific run. It includes:

    - **wavelength_range [list] (Default: [3,20]):** a list of the boundary wavelength values in micron between which to calculate the spectrum with petitRADTRANS. It must be equal or larger than the input data wavelength range. If it is shorter, the rebinning of the theoretical spectrum done in the calculation of the likelihood (NAME THE FUNCTION) by the module spectres will throw an error (NAME THE ERROR). 
    - **output_folder [str]**: a string showing the path to the desired output folder.
    - **include_scattering [dict]**: a dictionary of sub-keywords that concern the scattering calculation. It is split into these keywords:
        
        - **Rayleigh [bool] (Default: False)**: a boolean to turn on Rayleigh scattering calculation for each theoretical model calculated during the run
        
        - **thermal [bool] (Default: False)**: a boolean to turn on thermal scattering calculation for each theoretical model calculated during the run
        
        - **direct_light [bool] (Default: False)**: a boolean to turn on direct light scattering calculation for each theoretical model calculated during the run
        
        - **clouds [bool] (Default: False)**: a boolean to turn on Rayleigh scattering calculation for each theoretical model calculated during the run
        
        - **include_CIA [bool] (Default: False)**: a boolean to turn on the collision-induced-absorption (CIA) calculation for each theoretical model calculated during the run.
        
        - **include_moon [bool] (Default: False)**: a boolean to turn on the calculation of the moon spectrum (blackbody) to be added to the planetary spectrum.
    - **resolution [int] (Default: 1000)**: an integer to specify the spectral resolution to use for the calculation for each theoretical model calculated during the run. If not specified, it will default to R=1000 which is the default resolution of petitRADTRANS. However, we strongly recommend using a lower resolution setting if you only require R=50-200 (LIFE scenario), since it will result in a significant increase in computational speed. It is also recommended to make sure that precomputed correlated-k tables for every species of interest are included in the opacity folder (i.e. the path of the PYRETLIFE_OPACITY_PATH environment variable).
  ```{warning}
  Documentation work in progress. (rebinning of corr-k at any resolution in retrievals)
  ```

    - **n_layers [int] (Default: 100)**: an integer to specify the number of layers in which to split the atmosphere.
    - **log_top_pressure [float] (Default: -6)**: a float to specify the pressure (in log scale) of the top layer of the atmosphere.
    - **live_points [int] (Default: 600)**: an integer to specify the number of live points used by MultiNest.

  ```
  RUN SETTINGS:
    wavelength_range: [2,20]
    output_folder: template_retrieval/
    include_scattering:
      Rayleigh: True
      thermal: False
      direct_light: False
      clouds: False
    include_CIA: True
    include_moon: False
    resolution: 50
    n_layers: 100
    log_top_pressure: -4
    live_points: 600
  ```

- **GROUND TRUTH DATA**: In this section, all the paths to the input data must be specified. Units can be specified if different from the default units.
    - **input_profile [str]**: the path to a file containing the input profile. It is optional and, if specified, it is only used during plotting. The file must be made of two columns (P,T) with P being the pressure in bars and T the temperature in K, with no headers. If not provided, the plotting routine will use the true values specified in TEMPERATURE PARAMETERS (see below) to produce the true P-T profile.
    - **data_files**: a dictionary of sub-keywords that concern the files containing the observed data. It is a nested dictionary that contains at least the information for one spectrum (and potentially more, see example). The keyword for the inner dictionary is free to choose by the user and we recommend using an easily human-readable keyword (e.g. data_short, data_long, life, luvoir…). For each observed data dictionary we then require the following keywords:

        - **path [str]**: the path to the data file. The file must be made of three columns: wavelengths, flux, and noise. Signal-to-noise ratio is not acceptable as a representation of the noise. The default units assumed by the retrieval are micron for the wavelengths,  erg s-1 Hz-1 m-2 for the flux and noise. If the data file contains different units, these will need to be specified with an additional keyword.
        - **unit [str]**: This is a string containing the non-default units for the wavelength and the flux. The two units need to be separated by a comma. The first unit needs to be the one associated with the wavelength, while the second one is the one associated with the flux and the noise. If not specified, the retrieval will assume that the wavelength and the flux (noise) have the default units.

  ```
  GROUND TRUTH DATA:
  input_profile: mf_clear.txt
  data_files:
      data_short:
        path: EqC17JAN_LIFEsim_SN10_short.txt
        unit:  micron, erg s-1 Hz-1 m-2
      data_long:
        path: EqC17JAN_LIFEsim_SN10_long.txt
        unit: micron, erg s-1 Hz-1 m-2
  ```

- **TEMPERATURE PARAMETERS**: In this section, all settings/knowns/parameters related to the pressure-temperature profile must be included. This section is composed of these keywords:

    - **parameterization [str]**: this setting specifies the parameterization of the pressure-temperature profile. The currently implemented parameterizations are:

      |Parameterization| Description|
      |:---| :-------------------|
      |isothermal| An isothermal atmospheric profile.|
      |polynomial |A polynomial atmospheric profile of degree n (>0). |
      |guillot |A Guillot atmospheric profile (used in the petitRADTRANS documentation). (eq.29 [Guillot (2010)](http://adsabs.harvard.edu/abs/2010A%26A...520A..27G))|
      |madhuseager|A Madhusudhan-Seager profile ([arXiv:0910.1347](https://arxiv.org/abs/0910.1347)) |
      |mod_madhuseager|A modified Madhusudhan-Seager profile (which excludes the isothermal deep layers)|
      |spline| A P-T profile defined by a set of points and a spline interpolation between these points|

    - **parameterization-specific parameters/known values**: this setting specifies the parameterization of the pressure-temperature profile. These are:

      |Parameterization| Parameterization-specific parameters/known values|
      |:---| :-------------------|
      |isothermal| **T_eq** [float]: the temperature of the isothermal profile|
      |polynomial|**a_i** (with i ranging from 0 to n-1) [float]: the coefficients of the polynomial function from the highest degree to the constant term. The retrieval automatically recognizes the degree of the polynomial by reading the **a_i** coefficients. All **a_i** coefficients need to be present either as parameters or known values for the polynomial fit to work. For example: if we want to assume a special polynomial function f(x)=**a_2***x^2, we would need to keep **a_2** as a parameter to be retrieved, but also set **a_1** and **a_0** as known values to 0.  |
      |guillot| **log_delta** [float]: ratio of $\\kappa_{\\rm IR}$ - the atmospheric opacity in the IR wavelengths (i.e. the cross-section per unit mass) and $g$ - the atmospheric surface gravity <br> **log_gamma** [float]: is the ratio between the optical and IR opacity, <br> **t_int** [float]: planetary internal temperature <br> **t_equ** [float]: the atmospheric equilibrium temperature <br> **log_p_trans** [float]: <br> **alpha** [float]:|
      |madhuseager| **log_P1** [float]: first pressure point (highest) <br> **log_P2** second pressure point (medium) [float]: <br> **log_P3** [float]: third pressure point (lowest) <br> **T0** [float]: temperature at lowest pressure point <br> **alpha1** [float]: first alpha coefficient <br> **alpha2** [float]: second alpha coefficient|
      |mod_madhuseager| **log_P1** [float]: first pressure point <br> **log_P2** [float]: second pressure point<br> **T0** [float]:temperature at lowest pressure point <br> **alpha1** [float]: first alpha coefficient <br> **alpha2** [float]:second alpha coefficient|
      |spline| **spline_degree_k** [int]: degree of the spline interpolation (1: linear; 2:quadratic;...)<br> **spline_points**: Number of points in the atmosphere n. Minimum: 2 (1 at top and 1 at bottom of the atmosphere)<br> **spline_smooting** [float]: option to add a Gaussian filter over the spline profile to smooth it out<br> **Ti** [float]: (i lies between 0 and n-1) Temperature at the ith point in the atmosphere. i=0 is the surface and i=n-1 is the temperature at the top of the atmosphere <br> **Position_Pi** [float]: (i lies between 1 and n-2) lies between 0 and. Position of the ith pressure point in the atmosphere.<br>|
  ```
  TEMPERATURE PARAMETERS:
    parameterization: polynomial
    a_2:
      prior:
        kind: uniform
        prior_specs:
          lower: 0.
          upper: 500.
      truth: 99.70
    a_1:
      prior:
        kind: uniform
        prior_specs:
          lower: 0.
          upper: 500.
      truth: 146.63
    a_0:
      prior:
        kind: uniform
        prior_specs:
          lower: 0.
          upper: 1000.
      truth: 285.01
  ```

- **PHYSICAL PARAMETERS**: In this section, we must list all the parameters or known values concerning the physics of the planet/system must be included. These are:
    - **P0**: the ground pressure of the atmosphere. In the case of opaque clouds (see [CLOUDS]), it must not be specified as a parameter, but it can be specified as a known. In all other cases, it must be specified either as a parameter or known. The default unit is bar.
    - **d_syst**: the distance of the system from the observer. The default unit is parsec.
    - **R_pl**: the radius of the planet. The default unit is R_earth (astropy constant).
    - **M_pl**: the mass of the planet. The default unit is M_earth (astropy constant).
	Here is an example:

  ```
  PHYSICAL PARAMETERS:
    P0:
      prior:
        kind: log-uniform
        prior_specs:
          log_lower: -4
          log_upper: 2
      truth: 1.0294
    d_syst:
      truth: 10.
    R_pl:
      prior:
        kind: gaussian
        prior_specs:
          mean: 1
          sigma: 0.2
      truth: 1.
    M_pl:
      prior:
        kind: log-gaussian
        prior_specs:
          log_mean: 0
          log_sigma: 0.4
      truth: 1.
  ```


- **CHEMICAL COMPOSITION PARAMETERS**: In this section, we must list all the parameters or known values concerning the atmospheric composition of the atmosphere. These are:
  - **mmw_inert [float]**: if specified, it allows the retrieval to consider an inert mass of that specified mean molecular weight that is spectrally inactive but that contributes to fill the atmosphere to reach the total pressure at each layer. It can be interpreted as an inactive filling gas. If not present, the retrieval won’t consider any filling gas.
  - **condensation [bool]**: if True, water is condensed out in the Atmosphere, yielding more realistic H2O profiles in retrievals. If not specified to be True, the condensation is turned off
  - **abundance_untis [str]**: specifies whether abundances in the config file are specified in Mass fractions (‘MMR’) or in volume mixing ratios (‘VMR’). If the parameter is not specified ‘MMR’ is assumed
  - **chemical_species_name**: the retrieval requires one keyword per chemical species to be considered as a known value/parameter in the atmosphere. If it is a parameter to be retrieved, the prior dictionary needs to be specified and the truth can be specified or not. If it is a known value, the truth keyword must be specified (see above). Additional optional keywords are:
      - **lines [str, list]**: the name of the specific opacity folder for each species. It can be a string or a list of strings (see example). If not specified, the chemical species can still be considered for CIA and Rayleigh scattering, but it won’t be considered as a regular absorber (e.g. N2).

  ```
  CHEMICAL COMPOSITION PARAMETERS:
    mmw_inert: 28
    condensation: False
    abundance_units: MMR
    N2:
      prior:
        kind: log-uniform
        prior_specs:
          log_lower: -10
          log_upper: 0
      truth: 0.79
    O2:
      lines: O2_main_HN20_air_C25
      prior:
        kind: log-uniform
        prior_specs:
          log_lower: -10
          log_upper: 0
      truth: 0.20
    CO2:
      lines:
        - CO2_main_HN20_air_C25
        - CO2_UV
      prior:
        kind: log-uniform
        prior_specs:
          log_lower: -10
          log_upper: 0
      truth: 0.00041
  ```



- **CLOUD PARAMETERS**:
```{warning}
Documentation work in progress.
```


- **SCATTERING PARAMETERS**: In this section, we must list all the parameters or known values concerning the scattering treatment. This section is optional and relevant only if any of the scattering settings are True (see above). The keywords that could be added are:

  - **reflectance:** needed when direct_light and/or thermal are True. It is the reflectance of the surface.
  ```{warning}
Documentation work in progress. (implement wavelength-dependent reflectance)
```
-**emissivity**: needed when direct_light and/or thermal are True. It is the emissivity of the surface. 
  - **geometry [str]**: It is a setting needed when direct_light is True. It represents the geometry of the star-planet system (see [petitRADTRANS docs](https://petitradtrans.readthedocs.io/en/latest/content/notebooks/emis_scat.html#Scattering-of-stellar-light)). Currently implemented (and validated) geometries are planetary_ave and dayside_ave. 
  ```{warning}
  Documentation work in progress. (add quadrature)
  ```
  - **stellar_temperature [float]**: it is a known value needed when direct_light is True. It is the stellar temperature, used to calculate the incident stellar spectrum (see [petitRADTRANS docs](https://petitradtrans.readthedocs.io/en/latest/content/notebooks/emis_scat.html#Scattering-of-stellar-light) ). The default unit is K.
  - **stellar_radius [float]**: it is a known value needed when direct_light is True. It is the stellar radius, used to calculate the incident stellar spectrum (see [petitRADTRANS docs](https://petitradtrans.readthedocs.io/en/latest/content/notebooks/emis_scat.html#Scattering-of-stellar-light) ). The default unit is R_sun.
  - **semimajor_axis [float]**: it is a known value needed when direct_light is True. It is the semimajor axis of the stellar-planet system, used to calculate the incident stellar spectrum (see [petitRADTRANS docs](https://petitradtrans.readthedocs.io/en/latest/content/notebooks/emis_scat.html#Scattering-of-stellar-light) ). The default unit is AU.

  ```
  SCATTERING PARAMETERS:
    reflectance:
      prior:
        kind: uniform
        prior_specs:
          lower: 0.
          upper: 1
      truth: 0.1
    emissivity:
      prior:
        kind: uniform
        prior_specs:
          lower: 0.
          upper: 1
      truth: 1
    geometry: planetary_ave/dayside_ave
    stellar_temperature:
      truth: 5778.
    stellar_radius:
      truth: 1.
    semimajor_axis:
      truth: 1.

  ```

- **USER-DEFINED UNITS**: This is an optional section that contains all user-defined units, if any (see [Units](units.md)). 

