#!/bin/sh

#BSUB -n 1 							# n_nodes
#BSUB -P R000						# project
#BSUB -q g_medium					# queue
#BSUB -J wildfires					# jobname
#BSUB -o job-%J.out
#BSUB -e job-%J.err
#BSUB -x 
#BSUB -gpu "num=2"


#Command to execute Python program
python3 main.py

#End of script