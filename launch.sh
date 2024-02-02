#!/bin/sh

#BSUB -n 4 							# n processes
#BSUB -q g_medium					# queue
#BSUB -P R000						# project
#BSUB -R "span[ptile=2]"			# n gpu per node
#BSUB -J wildfires					# jobname
#BSUB -o job-%J.out
#BSUB -e job-%J.err
#BSUB -gpu "num=2"
#BSUB -x 

#Command to execute Python program
python3 main.py

#End of script
