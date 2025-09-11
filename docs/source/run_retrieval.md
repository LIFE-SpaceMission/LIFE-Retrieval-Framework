
# Running a Retrieval

On any local or remote machine, you can launch the retrieval by using

```
python path/to/main.py -config path/to/config_file.yaml
```
Where you can specify the absolute path to the main.py file and the absolute path to the config file. 
When running on a shared machine, it is recommended to use nice -n 19 to reduce the priority of your jobs and to allow simultaneous computation by other people.

If you want to run the retrieval on multiple cores (recommended as it speeds up the calculation) you can use `mpiexec -n NUM_CORES` where you can specify the number of cores to use. 

The command then becomes:

```
nice -n 19 mpiexec -n NUM_CORES python /path/to/main.py --config /path/to/config_file.yaml
```
