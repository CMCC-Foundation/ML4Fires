# [&larr;](../README.md) Run the code on a LSF cluster (with GPUs)

<p align="justify">  In order to execute the code, the working directory <b> must be set </b> to <code>src</code> and a <code>*.sh</code> file must be prepared with the following template: </p>

```bash
#!/bin/sh
#BSUB -n total_n_jobs
#BSUB -q queue_name
#BSUB -P project_code
#BSUB -J exec_name
#BSUB -R "span[ptile=n_gpu_per_node]"
#BSUB -o ./out_file.out
#BSUB -e ./err_file.err
#BSUB -gpu "num=n_gpus_per_node"

python main.py --arguments **cli_arguments
```
Where:

- `total_n_jobs` is the total number of processes that must be launched and executed in parallel

- `queue_name` is the name of the queue, there are several options:
	
	- `s_short`, `s_medium` and `s_long` for code that must be executed sequentially	 

	- `p_short`, `p_medium`and `p_long` for code that must be parallelized

	- `g_short`, `g_medium` and `g_long`for code that needs the GPUs computational power to succeed

	- `g_devel` a special GPU queue with higher computational time resources

	
- `project_code` is the name of the project, for general research projcets it must be set to `R000`

- `exec_name` is the job name displayed once the script is launched and executed

- `n_gpu_per_node` is the number of GPU that must be used for each node


In the second part of the script, all the necessary variables must be defined:

- `DO_EXPS`: boolean flag used to specify if the experiments TOML configuration files must be created or not

- `EXPERIMENTS`: string with the python file name `experiments.py` that must be run to create the experiments TOML configuration files

- `TOML_16`, `TOML_32`, `TOML_64`, and `TOML_ALL`: strings with paths to the experiments TOML configuration files

- `MAIN`: string with the python file name `main.py` that must be run to execute the workflow

- `MULTIPLE_EXPS`: boolean flag used to choose if multiple experiments must be done or not

- `FROM` and `TO`: experiment start and end numbers, if they are the same number it means that only that experiment will be performed

- `from_value` and `to_value`: experiment start and end numbers, are used to create the range of values used to loop through the experiments.

```bash

echo Start the program " "
echo ""

set -e

# Create experiments that must be performed
DO_EXPS=false
EXPERIMENTS=experiments.py

if $DO_EXPS
then
	echo Creating experiments... " "
	echo ""
	python $EXPERIMENTS
fi

# Define TOML configuration filepaths
TOML_16="$PWD/config/experiments/UPP_16.toml"
TOML_32="$PWD/config/experiments/UPP_32.toml"
TOML_64="$PWD/config/experiments/UPP_64.toml"
TOML_ALL="$PWD/config/experiments/UPP_all.toml"

# Define main python file that must be executed
MAIN=main.py

# Flag to choose if multiple experiments must be done
MULTIPLE_EXPS=true

# Starting and ending experiment numbers
FROM=4
TO=4

from_value="$FROM"
to_value="$TO"
```

The last part of the script consists in looping through the list of experiments and lauching the python command:

`mpirun python $MAIN -c $TOML_ALL -nexp $exp_num`

where the `MAIN` variable contains the name of the file with the workflow code, `TOML_ALL` contains the path to the TOML configuration file with all the experiments and `exp_num` is the number of the experiment that must be performed:

```bash
# Command to execute Python program and to use MPI to parallelize model and data
echo Start training... " "
echo ""

if $MULTIPLE_EXPS
then
	if [[ $from_value -le $to_value ]]; then
	  order="Ascending"
	  exp_range=($(seq "$from_value" "$to_value"))
	else
	  order="Descending"
	  exp_range=($(seq "$to_value" -1 "$from_value"))
	fi

	echo "Performing experiment in $order order"

	for exp_num in "${exp_range[@]}"; do
	  echo "Experiment: $exp_num"
	  echo "Running experiment... $exp_num"
	  mpirun python $MAIN -c $TOML_ALL -nexp $exp_num
	done
else
	mpirun python $MAIN
fi

# End of script
echo "Program ended"

```

<p align="justify">  For our experiment, we run the code on the Juno LSF cluster, using the following command: </p>

```bash
# locate the current working directory to src
cd src

# e.g. training
bsub < launch.sh
```
