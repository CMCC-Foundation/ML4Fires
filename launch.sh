#!/bin/sh

#BSUB -n 8							# n processes
#BSUB -q g_devel					# queue
#BSUB -P R000						# project
#BSUB -R "span[ptile=2]"			# n gpu per node
#BSUB -J wildfires					# jobname
#BSUB -o job-%J.out
#BSUB -e job-%J.err
#BSUB -gpu "num=2"


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
