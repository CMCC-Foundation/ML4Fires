#!/bin/sh

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

MODEL=unet
TRAINING_FILE=phase_training_100.py
for exp in "${exponents[@]}"; do
  BASE_FILTER_DIM=$(($BASE ** $exp))
  echo $BASE_FILTER_DIM
  python $TRAINING_FILE -bfd $BASE_FILTER_DIM -mdl $MODEL
done

# End of script
echo "Program ended"
