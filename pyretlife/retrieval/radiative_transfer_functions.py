from pathlib import Path
from typing import Union, Tuple
import os, sys


def define_linelists(config:dict, settings:dict, input_opacity_path:Union[str,Path]) -> Tuple[list,list,list,list]:
        used_line_species=[]
        used_rayleigh_species=[]
        used_cia_species = []
        used_cloud_species = []

        species=list(set(config['CHEMICAL COMPOSITION PARAMETERS'].keys()))

        used_line_species = ingest_opacity_linelists(settings,input_opacity_path)

        tot_mols = [s.split("_")[0] for s in used_line_species]

        #RAYLEIGH
        if settings['include_scattering']['Rayleigh']:
            used_rayleigh_species = ingest_rayleigh_linelists(species)
            tot_mols.extend(used_rayleigh_species)
        #CIA
        if settings['include_CIA']:
            used_cia_species = ingest_cia_linelists(species,input_opacity_path)
            tot_mols.extend( [item for sublist in  [key.split('-') for key in used_cia_species] for item in sublist])
        #CLOUDS
        if settings['include_scattering']['clouds']:
            cloud_species = set(["_".join(key.split('_')[:2]) for key in config['CLOUD PARAMETERS'].keys() if key !='settings_clouds'])
            used_cloud_species=ingest_cloud_linelists(cloud_species,input_opacity_path)
            tot_mols.extend([cloud.split("_", 1)[0] for cloud in cloud_species])


        tot_mols = list(set(tot_mols))

        print("Used line species:\t\t" + str(used_line_species))
        print("Used rayleigh species:\t\t" + str(used_rayleigh_species))
        print("Used continuum opacities:\t" + str(used_cia_species))
        print("Used cloud species:\t\t" + str(used_cloud_species))
        print("Used species *in general*:\t" + str(tot_mols))
        print()
        return used_line_species,used_rayleigh_species,used_cia_species,used_cloud_species

def ingest_opacity_linelists(settings: dict, input_opacity_path: Union[str, Path]) -> list:
    string = ""
    if settings["resolution"] != "1000":
        string = "_R_" + str(settings["resolution"])
    species_at_resolution = []
    for line in settings['opacity_linelist']:
        species_at_resolution.append(line + string)

    line_species = os.listdir(
        Path(input_opacity_path)/ "opacities"/"lines"/"corr_k"
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


def ingest_cia_linelists(species: list, input_opacity_path: Union[str, Path]) -> list:
    used_cia_species = []
    continuum_opacities = os.listdir(
        Path(input_opacity_path) / "opacities" / "continuum" / "CIA"
    )
    for cia in continuum_opacities:
        cia_components = cia.split("-")
        if len(cia_components) > 1:
            if (
                    species.count(cia_components[0])
                    + species.count(cia_components[1])
                    == 2
            ):
                used_cia_species.append(cia)
    return used_cia_species

def ingest_cloud_linelists(cloud_species:list, input_opacity_path:Union[str, Path]) -> list:
    used_cloud_species=[]
    cloud_dict = {
        "a": "/amorphous",
        "c": "/crystalline",
        "m": "/mie",
        "d": "/DHS",
    }
    for cloud in cloud_species:
        cloud_dir = str(
            input_opacity_path) + "/opacities/continuum/clouds/" +\
                    cloud.split('_')[0].replace('(c)', '_c') + \
                    cloud_dict[cloud[-2]] + cloud_dict[cloud[-1]]

        if not os.path.exists(cloud_dir):
            sys.exit(
                "ERROR: No opacities found for the cloud species "
                + str(cloud)
                + "."
            )
        used_cloud_species.append(cloud)
    return used_cloud_species



    def retrieval_model_plain(self, em_contr=True):
        """
        Creates the pressure-temperature profile for the current atmosphere
        and calculates the corresponding emitted flux.
        """

        add_cloud_scat_as_abs = False
        self.abundances = {}
        self.cloud_radii = {}
        self.cloud_lnorm = 0
        for name in self.chem_vars.keys():
            if "(c)" in name:
                add_cloud_scat_as_abs = True
                self.abundances[name] = np.zeros_like(self.press)
                for cloud in self.cloud_vars.keys():
                    # Calculate bottom pressure from the thickness parameter
                    self.cloud_vars[cloud]["bottom_pressure"] = (
                        self.cloud_vars[cloud]["top_pressure"]
                        + self.cloud_vars[cloud]["thickness"]
                    )
                    if name in cloud:
                        self.abundances[name][
                            np.where(
                                (
                                    self.press
                                    < self.cloud_vars[cloud]["bottom_pressure"]
                                )
                                & (
                                    self.press
                                    > self.cloud_vars[cloud]["top_pressure"]
                                )
                            )
                        ] = self.chem_vars[name]
                        self.cloud_radii[name] = self.cloud_vars[cloud][
                            "particle_radius"
                        ]
                        self.cloud_lnorm = self.cloud_vars[cloud][
                            "sigma_lnorm"
                        ]
            else:
                self.abundances[name.split("_")[0]] = (
                    np.ones_like(self.press) * self.chem_vars[name]
                )
        self.calc_MMW()

        # Calculate the moon flux
        if self.settings["moon"] == "True":
            nu = self.rt_object.freq
            exponent = self.nc.h * nu / (self.nc.kB * self.moon_vars["T_m"])
            B_nu = (
                2
                * self.nc.h
                * nu**3
                / self.nc.c**2
                / (np.exp(exponent) - 1)
            )  # in erg/cm^2/s/Hz/sr
            self.moon_flux = np.pi * B_nu  # in erg/cm^2/s/Hz

        # Setting the scattering parameters for the surface
        if self.settings["scattering"] == "True":
            # Try adding the reflectance
            try:
                self.rt_object.reflectance = self.scat_vars[
                    "reflectance"
                ] * np.ones_like(self.rt_object.freq)
            except:
                pass
            # Try adding the emissivity
            try:
                self.rt_object.emissivity = self.scat_vars[
                    "emissivity"
                ] * np.ones_like(self.rt_object.freq)
            except:
                pass

        # Calculate the Spectrum of the planet, prevent unnecessary printing
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        if not self.settings["directlight"]:
            self.rt_object.calc_flux(
                self.temp,
                self.abundances,
                self.phys_vars["g"],
                self.MMW,
                radius=self.cloud_radii,
                sigma_lnorm=self.cloud_lnorm,
                add_cloud_scat_as_abs=add_cloud_scat_as_abs,
                contribution=em_contr,
            )
        else:
            self.rt_object.calc_flux(
                self.temp,
                self.abundances,
                self.phys_vars["g"],
                self.MMW,
                radius=self.cloud_radii,
                sigma_lnorm=self.cloud_lnorm,
                geometry=self.settings["geometry"],
                Tstar=self.scat_vars["stellar_temp"],
                Rstar=self.scat_vars["stellar_radius"],
                semimajoraxis=self.scat_vars[
                    "semimajoraxis"
                ],  # *self.nc.r_sun*self.nc.AU
                add_cloud_scat_as_abs=add_cloud_scat_as_abs,
                contribution=em_contr,
            )
        sys.stdout = old_stdout

        # Scale the fluxes to the desired separation
        if self.phys_vars["d_syst"] is not None:
            self.rt_object.flux *= (
                self.phys_vars["R_pl"] ** 2 / self.phys_vars["d_syst"] ** 2
            )
            if self.settings["moon"] == "True":
                self.moon_flux *= (
                    self.moon_vars["R_m"] ** 2 / self.phys_vars["d_syst"] ** 2
                )
