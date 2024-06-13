## Run the code on a LSF cluster (with GPUs)

<p align="justify">  In order to execute the code, the working directory <b> must be set </b> to <code>src</code> and a <code>*.sh</code> file must be prepared with the following template: </p>

```bash
#!/bin/sh
#BSUB -n total_n_jobs
#BSUB -q queue_name
#BSUB -P project_code
#BSUB -J exec_name
#BSUB -o ./out_file.out
#BSUB -e ./err_file.err
#BSUB -gpu "num=n_gpus_per_node"

python main.py --arguments **cli_arguments
```

<p align="justify">  For our experiment, we run the code on the Juno LSF cluster, using the following command: </p>

```bash
# locate the current working directory to src
cd src

# e.g. training
bsub < launch.sh
```
