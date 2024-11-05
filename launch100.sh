#!/bin/bash

echo Start the program " "
echo ""

set -e

BASE=2
FROM=2
TO=2 #7
from_value="$FROM"
to_value="$TO"

exponents=($(seq "$from_value" "$to_value"))

echo "Performing experiments"

MODEL=unetpp
TRAINING_FILE=phase_training_100.py
for exp in "${exponents[@]}"; do
	BASE_FILTER_DIM=$(($BASE ** $exp))
	echo "Base filter dimension: "$BASE_FILTER_DIM
	# mpirun python $TRAINING_FILE -bfd $BASE_FILTER_DIM -mdl $MODEL
	python $TRAINING_FILE -bfd $BASE_FILTER_DIM -mdl $MODEL
done

# End of script
echo "Program ended"
