# Installation


## Pyenv Installation (if necessary)

```
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
cd ~/.pyenv && src/configure && make -C src && cd ..
~/.pyenv/bin/pyenv install --list
~/.pyenv/bin/pyenv install <version>
```

## Virtual Environment
Create directory with virtual environment (mkdir retrieval_dir)

```
python -m venv venvname
source venvname/bin/activate
```

## Install petitRADTRANS from Gitlab
Install prerequisites
```
pip install numpy meson-python ninja
```

Installation of petitRADTRANS

```
git clone https://gitlab.com/mauricemolli/petitRADTRANS.git
cd petitRADTRANS
pip install . --no-build-isolation
```

Check installation of petitRADTRANS (and set path to opacities)
```
python
from petitRADTRANS.config import petitradtrans_config_parser
petitradtrans_config_parser.set_input_data_path(<path to opacity database>)
```


## Install PyMultiNest from Github
Install and build MultiNest
```
git clone https://github.com/JohannesBuchner/MultiNest
cd MultciNest/build
cmake ..
make
```

Add path for LD library to .bashrc (or .bash_profile) file (source .bashrc)
```
export LD_LIBRARY_PATH=<path to MultiNest>/MultiNest/lib/:$LD_LIBRARY_PATH
```

Don’t forget to `source .bashrc`


## Install PyRetLIFE Package from Github
Install PyRetLIFE Package
```
git clone https://github.com/LIFE-SpaceMission/LIFE-Retrieval-Framework
-
pip install -e .
pip install -r ./requirements.txt
```
