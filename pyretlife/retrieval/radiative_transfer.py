import os
import sys
from pathlib import Path
from typing import Union, Tuple, Set

import numpy as np
import spectres
from numpy import ndarray


def define_linelists(
    config: dict, settings: dict, input_opacity_path: Union[str, Path]
) -> Tuple[list, list, list, list]:

    used_rayleigh_species = []
    used_cia_species = []
    used_cloud_species = []

    species = list(config["CHEMICAL COMPOSITION PARAMETERS"].keys())

    used_line_species = ingest_opacity_linelists(settings, input_opacity_path)

    tot_mols = [s.split("_")[0] for s in used_line_species]

    # RAYLEIGH
    if settings["include_scattering"]["Rayleigh"]:
        used_rayleigh_species = ingest_rayleigh_linelists(species)
        tot_mols.extend(used_rayleigh_species)
    # CIA
    if settings["include_CIA"]:
        used_cia_species = ingest_cia_linelists(species, input_opacity_path)
        tot_mols.extend(
            [
                item
                for sublist in [key.split("-") for key in used_cia_species]
                for item in sublist
            ]
        )
    # CLOUDS
    if settings["include_scattering"]["clouds"]:
        cloud_species = set(
            [
                "_".join(key.split("_")[:2])
                for key in config["CLOUD PARAMETERS"].keys()
                if key != "settings_clouds"
            ]
        )
        used_cloud_species = ingest_cloud_linelists(
            cloud_species, input_opacity_path
        )
        tot_mols.extend([cloud.split("_", 1)[0] for cloud in cloud_species])

    tot_mols = list(set(tot_mols))

    print("Used line species:\t\t" + str(used_line_species))
    print("Used rayleigh species:\t\t" + str(used_rayleigh_species))
    print("Used continuum opacities:\t" + str(used_cia_species))
    print("Used cloud species:\t\t" + str(used_cloud_species))
    print("Used species *in general*:\t" + str(tot_mols))
    print()
    return (
        used_line_species,
        used_rayleigh_species,
        used_cia_species,
        used_cloud_species,
    )


def ingest_opacity_linelists(
    settings: dict, input_opacity_path: Union[str, Path]
) -> list:
    string = ""
    if settings["resolution"] != "1000":
        string = "_R_" + str(settings["resolution"])
    species_at_resolution = []
    for line in settings["opacity_linelist"]:
        species_at_resolution.append(line + string)

    line_species = os.listdir(
        Path(input_opacity_path) / "opacities" / "lines" / "corr_k"
    )
    return list(set(species_at_resolution) & set(line_species))


def ingest_rayleigh_linelists(species: list) -> list:
    rayleigh_species = [
        "H2",
        "He",
        "H2O",
        "CO2",
        "O2",
        "N2",
        "CO",
        "CH4",
        "N2",
    ]
    return list(set(species) & set(rayleigh_species))


def ingest_cia_linelists(
    species: list, input_opacity_path: Union[str, Path]
) -> list:
    used_cia_species = []
    continuum_opacities = os.listdir(
        Path(input_opacity_path) / "opacities" / "continuum" / "CIA"
    )
    for cia in continuum_opacities:
        cia: str
        cia_components = cia.split("-")
        if len(cia_components) > 1:
            if (
                species.count(cia_components[0])
                + species.count(cia_components[1])
                == 2
            ):
                used_cia_species.append(cia)
    return used_cia_species


def ingest_cloud_linelists(
    cloud_species: Set[str], input_opacity_path: Union[str, Path]
) -> list:
    used_cloud_species = []
    cloud_dict = {
        "a": "/amorphous",
        "c": "/crystalline",
        "m": "/mie",
        "d": "/DHS",
    }
    for cloud in cloud_species:
        cloud_dir = (
            str(input_opacity_path)
            + "/opacities/continuum/clouds/"
            + cloud.split("_")[0].replace("(c)", "_c")
            + cloud_dict[cloud[-2]]
            + cloud_dict[cloud[-1]]
        )

        if not os.path.exists(cloud_dir):
            sys.exit(
                "ERROR: No opacities found for the cloud species "
                + str(cloud)
                + "."
            )
        used_cloud_species.append(cloud)
    return used_cloud_species


def calculate_moon_flux(frequency: ndarray, prt_instance, moon_vars: dict):
    exponent = prt_instance.nat_cst.h * frequency / (prt_instance.nat_cst.kB * moon_vars["T_m"])
    blackbody_nu = (
            2
            * prt_instance.nat_cst.h
            * frequency ** 3
            / prt_instance.nat_cst.c ** 2
            / (np.exp(exponent) - 1)
    )  # in erg/cm^2/s/Hz/sr
    return np.pi * blackbody_nu  # in erg/cm^2/s/Hz


def assign_reflectance_emissivity(
    scat_vars: dict, frequency: ndarray
) -> Tuple[ndarray, ndarray]:
    reflectance = scat_vars["reflectance"] * np.ones_like(frequency)
    emissivity = scat_vars["emissivity"] * np.ones_like(frequency)
    return reflectance, emissivity


def calculate_emission_flux(
    rt_object,
    settings: dict,
    temp: ndarray,
    abundances: dict,
    gravity: float,
    mmw: ndarray,
    cloud_radii: dict,
    cloud_lnorm: int,
    scat_vars: dict,
    em_contr=False,
) -> Tuple[ndarray, ndarray]:
    """
    Creates the pressure-temperature profile for the current atmosphere
    and calculates the corresponding emitted flux.
    """

    # Calculate the Spectrum of the planet, prevent unnecessary printing
    # TODO implement better logging
    # old_stdout = sys.stdout
    # sys.stdout = open(os.devnull, "w")
    if not settings["include_scattering"]["direct_light"]:
        rt_object.calc_flux(
            temp,
            abundances,
            gravity,
            mmw,
            radius=cloud_radii,
            sigma_lnorm=cloud_lnorm,
            add_cloud_scat_as_abs=settings["include_scattering"]["clouds"],
            contribution=em_contr,
        )
    else:
        rt_object.calc_flux(
            temp,
            abundances,
            gravity,
            mmw,
            radius=cloud_radii,
            sigma_lnorm=cloud_lnorm,
            geometry=settings["geometry"],
            Tstar=scat_vars["stellar_temperature"],
            Rstar=scat_vars["stellar_radius"],
            semimajoraxis=scat_vars["semimajor_axis"],
            add_cloud_scat_as_abs=settings["include_scattering"]["clouds"],
            contribution=em_contr,
        )
    # sys.stdout = old_stdout
    return rt_object.freq, rt_object.flux


def scale_flux_to_distance(
    flux: ndarray, radius: float, distance: float
) -> ndarray:
    # Scale the fluxes to the desired separation
    flux = flux * radius**2 / distance**2
    return flux


def rebin_spectrum(instrument, wavelength, flux):
    # differ strongly
    if not np.array(
        [(np.round(instrument["wavelength"], 10) == np.round(wavelength, 10))]
    ).all():
        return spectres.spectres(
            instrument["wavelength"],
            wavelength,
            flux,
        )
    else:
        return flux
