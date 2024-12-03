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

##  Numpy needs to be installed

```
pip install numpy
```

## Download petitRADTRANS from Github
Installation of petitRADTRANS

```
git clone https://gitlab.com/mauricemolli/petitRADTRANS.git
```

Make (in /petitRADTRANS/petitRADTRANS dir)
```
python -m numpy.f2py -c --opt='-O3 -funroll-loops -ftree-vectorize -ftree-loop-optimize -msse -msse2 -m3dnow' -m fort_input fort_input.f90
python -m numpy.f2py -c --opt='-O3 -funroll-loops -ftree-vectorize -ftree-loop-optimize -msse -msse2 -m3dnow' -m fort_spec fort_spec.f90
python -m numpy.f2py -c --opt='-O3 -funroll-loops -ftree-vectorize -ftree-loop-optimize -msse -msse2 -m3dnow' -m fort_rebin fort_rebin.f90
```

Add paths for opacity database and LD library to .bashrc  (or .bash_profile) file (source .bashrc)
```
export pRT_input_data_path="/home/ipa/quanz/shared/opacity_database"
export LD_LIBRARY_PATH=/home/ipa/quanz/user_accounts/username/retrieval_dir/MultiNest/lib/:$LD_LIBRARY_PATH
```

Don’t forget to `source .bashrc`

Check installation of petitRADTRANS
```
pip install -r ./requirements.txt
cd /retrieval_dir/petitRADTRANS/
python
import petitRADTRANS

```


## Download PyMultiNest from Github
Install and build MultiNest
```
git clone https://github.com/JohannesBuchner/MultiNest
cd MultiNest/build
cmake ..
make
```


## Download PyRetLIFE Package from Github
Install PyRetLIFE Package
```
git clone - - branch suggestions https://github.com/konradbjorn/Terrestrial_Retrieval
cd Terrestrial_Retrieval/
pip install -e .
pip install -r ./requirements.txt (from suggestions branch)
```

