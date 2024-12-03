# Units

The default units assumed when ingesting the config file (unless specified, see below) are:


| Parameter | Default Input Unit | 
|:-----------|:-------------------| 
| wavelength ranges | micron | 
| P0 | bar | 
| d_syst | parsec |
| R_pl | R_Earth | 
| M_pl | M_Earth | 
| stellar_temperature | K | 
| stellar_radius | R_sun | 
| semimajor_axis | au | 
| T_m | K | 
| R_m | R_Earth | 
| cloud_species_top_pressure | bar | 
| cloud_species_thickness | bar | 
| cloud_species_particle_radius | cm | 
| wavelength | micron | 
| flux | erg s-1 Hz-1 m-2 |

These are all associated with astropy units and constants (https://docs.astropy.org/en/stable/units/index.html#module-astropy.units.si) and can be indexed with any aliases associated with this module.
All other quantities (e.g.mass fractions, reflectance, emissivity) are assumed to be unitless.

If you need a custom unit, you can define a custom unit in the USER-DEFINED UNITS section. This needs to be expressed in terms of astropy constants in the config file (e.g. custom_meter: 1.01 m).

Within the retrieval, these units will be converted if necessary into the standard Output Units, which are the ones used also when saving the output (unless otherwise specified in the plotting routines).

| Parameter | Default Output Unit | 
|-----------|---------------------| 
| wavelength ranges | micron | 
| P0 | bar | 
| d_syst | m | 
| R_pl | cm | 
| M_pl | g | 
| stellar_temperature | K | 
| stellar_radius | cm | 
| semimajor_axis | cm | 
| T_m | K | | R_m | cm | 
| cloud_species_top_pressure | bar | 
| cloud_species_thickness | bar | 
| cloud_species_particle_radius | cm | 
| wavelength | micron | 
| flux | erg s-1 Hz-1 m-2 |