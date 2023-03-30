from pyretlife.retrieval.UnitsUtil import UnitsUtil


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
            refined_dict['prior']['input_prior_specs']= dictionary[section]["prior"]['prior_specs']
            refined_dict["prior"]["prior_specs"] = converted_prior
        refined_dict["type"] = dictionary[section]["type"]
        dictionary[section] = refined_dict
    return dictionary
