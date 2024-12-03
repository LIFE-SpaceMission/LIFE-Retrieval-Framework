# Structure of the GitHub Repo

```
Terrestrial_Retrieval
├── README.md
├── configs
│   ├── config_default.yaml
│   ├── config_new.yaml
│   ├── config_testing.yaml
│   ├── config_units.ini
│   └── config_vae.ini
├── multijob_sbatch.sh
├── pyproject.toml
├── pyretlife
│   ├── VERSION
│   ├── __init__.py
│   ├── legacy_src
│   │   ├──….
│   └── retrieval_plotting
│       ├── __init__.py
│       ├── color_handling.py
│       ├── custom_matplotlib_handles.py
│       ├── inlay_plot.py
│       ├── parallel_computation.py
│       └── run_plotting.py
├── requirements.txt
├── sbatch.sh
├── scripts
│   ├── main.py
│   └── main_plotting.py
├── setup.cfg
├── setup.py
├── template_retrieval
│   └── inputs
│       ├── Modern_Earth_pt_massfractions.csv
│       └── Rugheimer_ModernEarth_cloudfree_R50_SNR10.txt
└── tests
    ├── test_config.py
    └── test_priors.py

```