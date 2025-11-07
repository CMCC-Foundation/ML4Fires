## EXAMPLE NOTEBOOKS

This folder containes a set of demonstration Jupyter Notebooks for testing and running an ML model based on ML4fires.

### demo_evaluation.ipynb

This notebook allow to evaluate the results of pre-trained ML model for burned area maps generation on a test set based on historical data.

Workflow  

1. Parameter definition
    First the user specifies:  
        - `run_name`: name of the pre-trained model on MLflow to be used for inference.  
        - `data_path`: path to the FCCI BA data.  

2. Model and dataset loading
    The ML model is downloaded from MLflow together with other artifacts
    Provenance of the training process associated with the model can be inspected
    Data is loaded in the data loader

3. Inference
    The model is executed on the historical data for the generation of burned area maps  

4. Visualization
    The notebook provides several visualization functions to display:
    - average burned area maps on the test (both values from the dataset (FCCI BA) and predicted ones)
    - a map with the difference between predicted values and historical values
    Morover, a set of performance metrics (RMSE, MAE, SSIM, etc.) can be also computed.

### cmip6_forecast.ipynb

This notebook allows to run a pre-trained ML model from ML4Fires on climate data.

Workflow

1. Parameter definition & input selection
    The user specifies:
        - `config`: path to the configuration file used for CMIP6 data pre-processing
        - `searfire_ds_path`: path to the seasfirecube data 
        - `sea_poles_mask`: path to the mask file

    then it can select via widgets the climate data to be used by selecting:
        - the Shared Socioeconomic Pathways among a pre-defined list
        - the climate model among a pre-defined list
        - the range of years to consider

    finally, the `run_name` of pre-trained ML model from MLflow can be specified.


2. Data loading & prediction
    - Climate data from selected climate model, scenario and year range is loaded and prepared for the prediction 
    - The pre-trained model selected is run on the data for generating the prediction of burned areas

3. Post processing of predictions
    - Different aggregation functions can be selected from widgets for a monthly, yearly and decadal basis

4. Results visualization
    - Finally an average map with the burned area estimation is shown