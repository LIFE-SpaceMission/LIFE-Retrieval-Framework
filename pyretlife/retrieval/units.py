import astropy.units as u
import numpy as np


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------


class UnitsUtil:
    def __init__(self, pRT_const):
        # Standard units for input of the retrieval. All parameters not
        # mentioned here are assumed to be dimensionless

        self.std_input_units = {
            # LAMBDA RANGE
            "WMIN": u.micron,
            "WMAX": u.micron,
            # PHYSICAL PARAMETERS
            "P0": u.bar,
            "d_syst": self.custom_unit("pRT_pc", pRT_const.pc * u.cm),
            "R_pl": self.custom_unit("pRT_R_earth", pRT_const.r_earth * u.cm),
            "M_pl": self.custom_unit("pRT_M_earth", pRT_const.m_earth * u.g),
            # SCATTERING PARAMETERS
            "stellar_temperature": u.K,
            "stellar_radius": self.custom_unit(
                "pRT_R_sun", pRT_const.r_sun * u.cm
            ),
            "semimajor_axis": self.custom_unit("pRT_AU", pRT_const.AU * u.cm),
            # MOON PARAMETERS
            "T_m": u.K,
            "R_m": u.pRT_R_earth,
            # Input Files
            "wavelength": u.micron,
            "flux": u.erg / u.s / u.Hz / u.m**2,
        }

        # Units for the retrieval (mostly cgs). All parameters not
        # mentioned here are assumed to be dimensionless
        self.retrieval_units = {
            # LAMBDA RANGE
            "WMIN": u.micron,
            "WMAX": u.micron,
            # PHYSICAL PARAMETERS
            "P0": u.bar,
            "d_syst": u.m,
            "R_pl": u.cm,
            "M_pl": u.g,
            # SCATTERING PARAMETERS
            "stellar_temperature": u.K,
            "stellar_radius": u.cm,
            "semimajor_axis": u.cm,
            # MOON PARAMETERS
            "T_m": u.K,
            "R_m": u.cm,
            # Input Files
            "wavelength": u.micron,
            "flux": u.erg / u.s / u.Hz / u.m**2,
        }

    @staticmethod
    def custom_unit(name, astropy_conv):
        temp_unit = u.def_unit(name, astropy_conv)
        setattr(u, name, temp_unit)
        u.add_enabled_units([getattr(u, name)])
        return getattr(u, name)

    @staticmethod
    def return_units(key, units):
        try:
            return units[key]
        except KeyError:
            return u.dimensionless_unscaled


    @staticmethod
    def prior_unit_conversion(
        key, input_unit, target_unit, prior, printing=True
    ) -> dict:
        converted_prior = {}

        # Conversion of LogUniform priors
        if prior["kind"] == "log-uniform":
            for spec in prior["prior_specs"]:
                converted_prior[spec] = np.log10(
                    (10 ** prior["prior_specs"][spec] * input_unit)
                    .to(target_unit)
                    .value
                )
        elif prior["kind"] == "log-gaussian":
            # only translate the mean but leave the sigma the same (in log space)
            converted_prior["log_mean"] = np.log10(
                (10 ** prior["prior_specs"]["log_mean"] * input_unit)
                .to(target_unit)
                .value
            )
            converted_prior["log_sigma"] = prior["prior_specs"]["log_sigma"]
        else:
            for spec in prior["prior_specs"]:
                converted_prior[spec] = (
                    (prior["prior_specs"][spec] * input_unit)
                    .to(target_unit)
                    .value
                )

        # G and U priors are transformed easily as they do not contain a
        # logarithm they are just scaled
        # ULU and FU (only used for abundances and PT parameters in the
        # polynomial profile. no special treatment)

        # If a conversion was performed print it as a check
        if (target_unit != input_unit) and printing:
            print(
                "Conversion performed for prior of "
                + key
                + ". Prior kind: "
                + prior["kind"]
            )
            print("Input values:", prior["prior_specs"], input_unit)
            print("Converted value:", converted_prior, target_unit)
            print()

        return converted_prior

    @staticmethod
    def truth_unit_conversion(
        key,
        input_unit,
        target_unit,
        input_truth,
        printing=True,
    ):
        converted_truth = (input_truth * input_unit).to(target_unit)

        # If a conversion was performed print it as a check
        if (target_unit != input_unit) and printing:
            print("Conversion performed for truth value of " + key + ".")
            print("Input value:", input_truth, input_unit)
            print("Converted value:", converted_truth, target_unit)
            print()

        return converted_truth

    @staticmethod
    def unit_spectrum_conversion(
        key, input_unit, target_unit, input_data, printing=True
    ):
        conv_data = np.zeros_like(input_data)
        conv_data[:, 0] = (
            (input_data[:, 0] * input_unit[0])
            .to(target_unit[0], equivalencies=u.spectral())
            .value
        )
        conv_data[:, 1] = (
            (input_data[:, 1] * input_unit[1])
            .to(
                target_unit[1],
                equivalencies=u.spectral_density(
                    input_data[:, 0] * input_unit[0]
                ),
            )
            .value
        )
        conv_data[:, 2] = (
            (input_data[:, 2] * input_unit[1])
            .to(
                target_unit[1],
                equivalencies=u.spectral_density(
                    input_data[:, 0] * input_unit[0]
                ),
            )
            .value
        )

        if (target_unit != input_unit) and printing:
            print("Conversion performed for input spectrum " + key + ".")
            print(
                "Input wavelength unit:",
                input_unit[0],
                "\tConverted wavelength unit:",
                target_unit[0],
            )
            print(
                "Input flux unit:",
                input_unit[1],
                "\tConverted flux unit:",
                target_unit[1],
            )
            print()
        return conv_data

    @staticmethod
    def unit_spectrum_cube(input_unit, target_unit, input_wl, input_flux):
        conv_wl = (
            (input_wl * input_unit[0])
            .to(target_unit[0], equivalencies=u.spectral())
            .value
        )
        conv_flux = (
            (input_flux * input_unit[1])
            .to(
                target_unit[1],
                equivalencies=u.spectral_density(input_wl * input_unit[0]),
            )
            .value
        )
        return conv_wl, conv_flux

def convert_spectrum(instrument: dict, units: UnitsUtil) -> dict:
    for data_file in instrument.keys():
        converted_unit_wavelength = units.return_units(
            "wavelength", units.retrieval_units
        )
        converted_unit_flux = units.return_units("flux", units.retrieval_units)
        converted_data = units.unit_spectrum_conversion(
            data_file,
            [
                instrument[data_file]["input_unit_wavelength"],
                instrument[data_file]["input_unit_flux"],
            ],
            [converted_unit_wavelength, converted_unit_flux],
            instrument[data_file]["input_data"],
        )

        instrument[data_file] = {
            "wavelength": converted_data[:, 0],
            "flux": converted_data[:, 1],
            "error": converted_data[:, 2],
            "unit_wavelength": converted_unit_wavelength,
            "unit_flux": converted_unit_flux,
            "input_wavelength": instrument[data_file]["input_data"][:, 0],
            "input_flux": instrument[data_file]["input_data"][:, 1],
            "input_error": instrument[data_file]["input_data"][:, 2],
            "input_unit_wavelength": instrument[data_file][
                "input_unit_wavelength"
            ],
            "input_unit_flux": instrument[data_file]["input_unit_flux"],
        }
    return instrument


def convert_knowns_and_parameters(dictionary: dict, units: UnitsUtil) -> dict:
    for section in dictionary.keys():
        refined_dict = {}
        # Convert the input to retrieval units
        converted_unit = units.return_units(section, units.retrieval_units)
        refined_dict["input_unit"] = dictionary[section]["unit"]
        refined_dict["unit"] = converted_unit

        if "truth" in dictionary[section].keys():
            converted_truth = units.truth_unit_conversion(
                section,
                dictionary[section]["unit"],
                converted_unit,
                dictionary[section]["truth"],
            )
            refined_dict["input_truth"] = dictionary[section]["truth"]
            refined_dict["truth"] = converted_truth

        if "prior" in dictionary[section].keys():
            converted_prior = units.prior_unit_conversion(
                section,
                dictionary[section]["unit"],
                converted_unit,
                dictionary[section]["prior"],
            )
            refined_dict["prior"] = dictionary[section]["prior"]
            refined_dict["prior"]["input_prior_specs"] = dictionary[section][
                "prior"
            ]["prior_specs"]
            refined_dict["prior"]["prior_specs"] = converted_prior
        refined_dict["type"] = dictionary[section]["type"]
        dictionary[section] = refined_dict
    return dictionary

