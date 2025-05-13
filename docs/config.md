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

