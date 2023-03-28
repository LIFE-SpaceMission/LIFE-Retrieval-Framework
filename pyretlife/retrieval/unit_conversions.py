from pyretlife.retrieval.UnitsUtil import UnitsUtil


def convert_spectrum(Instrument: dict, Units: UnitsUtil) -> dict:
    for data_file in Instrument.keys():
        converted_unit_wavelength = Units.return_units(
            "wavelength", Units.retrieval_units
        )
        converted_unit_flux = Units.return_units("flux", Units.retrieval_units)
        converted_data = Units.unit_spectrum_conversion(
            data_file,
            [
                Instrument[data_file]["input_unit_wavelength"],
                Instrument[data_file]["input_unit_flux"],
            ],
            [converted_unit_wavelength, converted_unit_flux],
            Instrument[data_file]["input_data"],
        )

        Instrument[data_file] = {
            "wavelength": converted_data[:, 0],
            "flux": converted_data[:, 1],
            "error": converted_data[:, 2],
            "unit_wavelength": converted_unit_wavelength,
            "unit_flux": converted_unit_flux,
            "input_wavelength": Instrument[data_file]["input_data"][:, 0],
            "input_flux": Instrument[data_file]["input_data"][:, 1],
            "input_error": Instrument[data_file]["input_data"][:, 2],
            "input_unit_wavelength": Instrument[data_file][
                "input_unit_wavelength"
            ],
            "input_unit_flux": Instrument[data_file]["input_unit_flux"],
        }
    return Instrument


def convert_knowns_and_parameters(Dictionary: dict, Units: UnitsUtil) -> dict:
    for section in Dictionary.keys():
            refined_dict = {}
            # Convert the input to retrieval units
            converted_unit = Units.return_units(
                section, Units.retrieval_units
            )
            refined_dict["input_unit"] = Dictionary[section]["unit"]
            refined_dict["unit"] = converted_unit

            if "truth" in Dictionary[section].keys():
                converted_truth = Units.truth_unit_conversion(
                    section,
                    Dictionary[section]["unit"],
                    converted_unit,
                    Dictionary[section]["truth"],
                )
                refined_dict["input_truth"] = Dictionary[section][
                    "truth"
                ]
                refined_dict["truth"] = converted_truth

            if "prior" in Dictionary[section].keys():
                converted_prior = Units.prior_unit_conversion(
                    section,
                    Dictionary[section]["unit"],
                    converted_unit,
                    Dictionary[section]["prior"],
                )
                refined_dict["prior"] = Dictionary[section]["prior"]
                refined_dict["prior"]["converted_prior_specs"] = converted_prior
            refined_dict['type']=Dictionary[section]["type"]
            Dictionary[section] = refined_dict
    return Dictionary
