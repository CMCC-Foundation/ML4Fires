# Workflows based on ML4Fires

<p align="justify"> This repository includes a workflow useful to test the inference model for Wildfire Burned Areas Prediction and Projection. </p>

## Ophidia workflow

<p align="justify"> [Ophidia](https://ophidia.cmcc.it) is an open-source HPC framework for data-intensive analysis, exploiting advanced parallel computing techniques and smart data distribution methods. Ophidia exploits a client-server approach; the user can interact by means a Python module, called [PyOphidia](https://pyophidia.readthedocs.io/en/latest/), using a JSON interface. </p>
<p align="justify"> A sample workflow is coded in "fires.py" and a Python notebook 'fires.ipynb' is also provided. It allows to evaluate burned areas results across an ensemble of different climate models. </p>
<p align="justify"> The workflow includes a number of tasks for data preparation, ML model execution and data post-processing. In particular, given a set of input parameters (e.g. "time domain" can be set the command line), the workflow: </p>

1. selects the variables from input NetCDF files,
1. aggregates them accordingly (different operations can be applied to the various variables),
1. regrids the variables to a common grid,
1. executes a pre-trained ML model on such data,
1. masks the results, and
1. evaluates the statistics from the ensemble of results on multiple CMIP6 models.

<p align="justify"> The workflow can be started by the following Python command. </p>

```
$ fires(time_range = "2030-01-01_2031-01-01")
```
<p align="justify"> CWL and JSON implemetations are also provided. Provenance information can be also produced by PyOphidia, once the execution is completed. </p>

