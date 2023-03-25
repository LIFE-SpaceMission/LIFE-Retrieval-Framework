# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

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
            "stellar_temp": u.K,
            "stellar_radius": self.custom_unit(
                "pRT_R_sun", pRT_const.r_sun * u.cm
            ),
            "semimajoraxis": self.custom_unit("pRT_AU", pRT_const.AU * u.cm),
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
            "stellar_temp": u.K,
            "stellar_radius": u.cm,
            "semimajoraxis": u.cm,
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
        except IndexError:
            return u.dimensionless_unscaled

    def unit_extract(self, key, input_string):
        # Split the string into components ignoring whitespaces
        splitted = [
            string for string in input_string.split(" ") if string != ""
        ]

        # Read the provided units. If none are provided use standard input.
        ind = [i for i in range(len(splitted)) if splitted[i] == "Unit"]
        if not ind:  # This checks if `ind` is empty!
            input_unit = self.return_units(key, self.std_input_units)
        else:
            input_unit = u.Unit(" ".join(splitted[(ind[0] + 1) :]))
            splitted = splitted[: ind[0]]

        return input_unit, splitted

    @staticmethod
    def unit_conv(
        key,
        input_unit,
        target_unit,
        input_truth,
        prior_type=None,
        input_prior=None,
        printing=True,
    ):
        # Convert the truth if it is provided
        conv_truth = (
            (input_truth * input_unit).to(target_unit).value
            if input_truth is not None
            else None
        )

        # If provided convert the prior
        if (prior_type is not None) and (input_prior is not None):
            # Conversion of LogUniform priors
            if prior_type in ["LU"]:
                conv_prior = [
                    (10**i * input_unit).to(target_unit) for i in input_prior
                ]
                conv_prior = [np.log10(i.value) for i in conv_prior]

            # Conversion of LogGaussian priors
            elif prior_type in ["LG"]:
                conv_prior = [
                    (10 ** input_prior[0] * input_unit).to(target_unit),
                    input_prior[1],
                ]
                conv_prior = [np.log10(conv_prior[0].value), conv_prior[1]]

            # G and U priors are transformed easily as they do not contain a
            # logarithm they are just scaled
            # ULU and FU (only used for abundances and PT parameters in the
            # polynomial profile. no special treatment)
            else:
                conv_prior = [
                    (i * input_unit).to(target_unit).value for i in input_prior
                ]

            # If a conversion was performed print it as a check
            if (target_unit != input_unit) and printing:
                print(
                    "Conversion performed for retrieved parameter " + key + "."
                )
                print(
                    "Input value:",
                    input_truth,
                    input_unit,
                    "\tInput prior:",
                    prior_type,
                    input_prior,
                )
                print(
                    "Converted value:",
                    conv_truth,
                    target_unit,
                    "\tConverted prior:",
                    prior_type,
                    conv_prior,
                )
                print()

            return conv_truth, conv_prior

        # If a conversion was performed print it as a check
        if (target_unit != input_unit) and printing:
            print("Conversion performed for known parameter " + key + ".")
            print("Input value:", input_truth, input_unit)
            print("Converted value:", conv_truth, target_unit)
            print()

        return conv_truth

    def unit_spectrum_extract(self, input_string):
        # Split the string into components ignoring whitespaces
        splitted = [
            string for string in input_string.split(" ") if string != ""
        ]

        # Read the provided units. If none are provided use standard input.
        ind = [i for i in range(len(splitted)) if splitted[i] == "Unit"]
        if not ind:  # This checks if `ind` is empty!
            input_unit = [
                self.return_units("wavelength", self.std_input_units),
                self.return_units("flux", self.std_input_units),
            ]
        else:
            input_unit = [
                u.Unit(i)
                for i in (" ".join(splitted[(ind[0] + 1) :])).split(",")
            ]

        return input_unit[0], input_unit[1], splitted[0]

    @staticmethod
    def unit_spectrum_conv(
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
