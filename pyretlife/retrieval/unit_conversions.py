from pyretlife.retrieval.UnitsUtil import UnitsUtil

def convert_spectrum(Instrument:dict, Units: UnitsUtil) -> dict:
    for data_file in Instrument.keys():

        converted_unit_wavelength = Units.return_units(
        "wavelength", Units.retrieval_units)
        converted_unit_flux = Units.return_units(
            "flux", Units.retrieval_units
        )
        converted_data = Units.unit_spectrum_conversion(
            data_file,
            [Instrument[data_file]['input_unit_wavelength'],
             Instrument[data_file]['input_unit_flux']],
            [converted_unit_wavelength, converted_unit_flux],
            Instrument[data_file]['input_data'],
        )

        Instrument[data_file] = {'wavelength': converted_data[:, 0],
                                 'flux': converted_data[:, 1],
                                 'error': converted_data[:, 2],
                                 "unit_wavelength": converted_unit_wavelength,
                                 "unit_flux": converted_unit_flux,
                                 'input_wavelength': Instrument[data_file]['input_data'][:, 0],
                                 'input_flux': Instrument[data_file]['input_data'][:, 1],
                                 'input_error': Instrument[data_file]['input_data'][:, 2],
                                 "input_unit_wavelength": Instrument[data_file]['input_unit_wavelength'],
                                 "input_unit_flux": Instrument[data_file]['input_unit_flux'],
                             }
    return Instrument

