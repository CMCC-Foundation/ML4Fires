## configuration.toml 
This configuration file defines the essential settings for wildfire prediction, such as features, data paths, and experiment settings.
| **Key**        | **Description**                                                                                                          |
| -------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `drivers`      | List of **input features** for the model (e.g., LAI, temperature, humidity). Users can **add or remove variables here**. |
| `targets`      | The **target variable** (default is `'fcci_ba'` for burned area).                                                        |
| `landsea_mask` | Land-sea mask variable (`'lsm'`) used to mask ocean areas.                                                               |

| Key              | Description                       |
| ---------------- | --------------------------------- |
| `trn_years_list` | Years for **training**.           |
| `val_years_list` | Years for **validation**.         |
| `tst_years_list` | Years for **testing/prediction**. |

| Key              | Description                       |
| ---------------- | --------------------------------- |
| `trn_years_list` | Years for **training**.           |
| `val_years_list` | Years for **validation**.         |
| `tst_years_list` | Years for **testing/prediction**. |


| Key        | Description                         |
| ---------- | ----------------------------------- |
| `LOGS_DIR` | Directory for saving logs.          |
| `DATA_DIR` | Directory containing your datasets. |

| Key               | Description                     |
| ----------------- | ------------------------------- |
| `TRACKING_URI`    | MLFlow tracking server URL.     |
| `EXPERIMENT_NAME` | Name of your MLFlow experiment. |


### models.toml
This file contains the model structure, hyperparameters, and training configurations for wildfire burned area prediction using TensorFlow/Keras.

#### [model]

| **Key**         | **Description**                                                       |
| --------------- | --------------------------------------------------------------------- |
| `devices`       | Auto-detects available **GPU devices** using TensorFlow API.          |
| `strategy`      | Distributed strategy (`MirroredStrategy`) for **multi-GPU training**. |
| `loss`          | Loss function used during training (e.g., **Mean Absolute Error**).   |
| `metrics`       | List of **evaluation metrics** (MSE, RMSE).                           |
| `optimizer`     | Optimizer used (**Adam** optimizer with specified learning rate).     |
| `learning_rate` | Learning rate for the optimizer (default `1e-4`).                     |
| `ba_hectares`   | Whether to compute **burned area in hectares** (`True` or `False`).   |

#### [layers]
| **Key**               | **Description**                             |
| --------------------- | ------------------------------------------- |
| `drop_rate`           | Dropout rate (`0.25`) for regularization.   |
| `kernel`              | Kernel size for convolution layers (`3x3`). |
| `activation`          | Activation function (`relu`).               |
| `initializer`         | Kernel initializer (`he_normal`).           |
| `padding`             | Padding strategy (`same`).                  |
| `regularizer`         | Regularization applied (`L2`).              |
| `maxpool_size`        | Size of max pooling window (`2x2`).         |
| `maxpool_strides`     | Strides for max pooling (`2x2`).            |
| `maxpool_data_format` | Data format (`channels_last`).              |

#### [config_1]
| **Key**              | **Description**                                 |
| -------------------- | ----------------------------------------------- |
| `bsize`              | Batch size (`4`).                               |
| `base_shape`         | Input shape (e.g., `720x1440` for the grid).    |
| `shuffle`            | Whether to shuffle data (`True` or `False`).    |
| `shard_size`         | Number of dataset shards (`1`).                 |
| `epochs`             | Number of training epochs (`10`).               |
| `shift_list`         | Time-shift steps list (`[0]` for no shift).     |
| `scaler_type_minmax` | Scaling type for **MinMax scaling**.            |
| `scaler_type_zscore` | Scaling type for **Z-score normalization**.     |
| `scaler_name`        | Filename for saving the scaler (`scaler.dump`). |
| `target_source`      | Source for target data (e.g., `FCCI`).          |


### torch.toml
This file configures the PyTorch Lightning training, model setup, and optimizer settings for the wildfire burned area prediction task using UNet++.

#### [base]

| **Key**             | **Description**                                         |
| ------------------- | ------------------------------------------------------- |
| `cuda_availability` | Checks if **CUDA GPU is available** for acceleration.   |
| `accelerator`       | Defines the accelerator (`cuda` for GPU usage).         |
| `matmul_precision`  | Precision setting for matrix multiplication (`medium`). |


#### [model]

| **Key**    | **Description**                                              |
| ---------- | ------------------------------------------------------------ |
| `strategy` | Distributed strategy (`DDPStrategy` for multi-GPU training). |
| `loss`     | Loss function used during training (`WeightedBCE_L1Loss`).   |

#### [model.loss_args]

| **Key**      | **Description**                            |
| ------------ | ------------------------------------------ |
| `weight_bce` | Weight for **BCE Loss component** (`0.3`). |
| `weight_l1`  | Weight for **L1 Loss component** (`0.7`).  |


#### [model.dir]

| **Key**             | **Description**                         |
| ------------------- | --------------------------------------- |
| `RUN_DIR`           | Directory for experiment run outputs.   |
| `CHECKPOINTS_DIR`   | Checkpoints directory inside `RUN_DIR`. |
| `SKIP_DAYS_DIRNAME` | Directory name for skip-days data.      |


#### [model.unetpp]

| **Key** | **Description**                         |
| ------- | --------------------------------------- |
| `cls`   | Class path for **UNet++ architecture**. |


#### [model.unetpp.args]

| **Key**            | **Description**                    |
| ------------------ | ---------------------------------- |
| `input_shape`      | Input shape (`[720, 1440, 8]`).    |
| `num_classes`      | Number of output classes (`1`).    |
| `depth`            | Depth of UNet++ (`4`).             |
| `base_filter_dim`  | Base filter size (`64`).           |
| `deep_supervision` | Enable deep supervision (`false`). |


#### [trainer]

| **Key**                   | **Description**                               |
| ------------------------- | --------------------------------------------- |
| `accumulation_steps`      | Gradient accumulation steps (`1`).            |
| `devices`                 | Number of devices (`1`).                      |
| `workers`                 | Number of workers (`1`).                      |
| `epochs`                  | Number of training epochs (`75`).             |
| `num_nodes`               | Number of nodes (`1`).                        |
| `precision`               | Precision (`32-true`).                        |
| `batch_size`              | Batch size (`2`).                             |
| `drop_reminder`           | Drops last incomplete batch (`true`).         |
| `plugins`                 | Plugins (`[MPIEnvironment()]`).               |
| `use_distributed_sampler` | Whether to use distributed sampler (`False`). |


#### [trainer.optim]

| **Key** | **Description**                       |
| ------- | ------------------------------------- |
| `cls`   | Optimizer class (`torch.optim.Adam`). |
| `args`  | Optimizer arguments (`lr=1e-4`).      |


#### [trainer.scheduler]

| **Key** | **Description**               |
| ------- | ----------------------------- |
| `cls`   | Scheduler class (`None`).     |
| `args`  | Scheduler arguments (`None`). |

#### [trainer.checkpoint]

| **Key** | **Description**           |
| ------- | ------------------------- |
| `ckpt`  | Checkpoint file (`None`). |


#### Notes:

    The loss function is fully configurable via loss_args, allowing for custom ratios of BCE and L1 Loss.

    The model architecture is based on UNet++, but you can replace it easily in model.unetpp.cls.

    Optimizer, scheduler, and checkpoint settings are also modular.



### train.toml

This file controls the experiment directory structure and output management for different models.

#### [run]

| **Key**        | **Description**                                                                               |
| -------------- | --------------------------------------------------------------------------------------------- |
| `curr_run_dir` | Lambda function to create the **current run directory** by combining base path and timestamp. |

#### [run.dir]

| **Key**       | **Description**                                                         |
| ------------- | ----------------------------------------------------------------------- |
| `transformer` | Directory for experiments using the **ViT model** (Vision Transformer). |
| `unetpp`      | Directory for experiments using the **UNet++ model**.                   |
| `unet`        | Directory for experiments using the **UNet model**.                     |

 Notes:

    This configuration helps organize training outputs and keep results separated per model type.

    By adjusting these directories, users can easily manage experiments for different model architectures.


    
### Description of the flow:

#### Configuration Files Block:

    configuration.toml → Data paths, variables, preprocessing config.

    models.toml → TensorFlow-based model settings.

    torch.toml → PyTorch model, loss, optimizer, trainer config.

    train.toml → Run directories, experiment naming.

#### Pipeline Flow Block:

    Data Preparation → Reads and preprocesses data.

    Model Setup → Loads model architecture and weights.

    Trainer Setup → Configures optimizer, scheduler, distributed training.

    Training Execution → Runs the model training.

    Evaluation & Metrics → Calculates and logs metrics.

    Baseline Comparison → Compares model with persistence/climatology.