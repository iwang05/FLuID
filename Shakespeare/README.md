# Invariant Dropout (Shakespeare)

This project demonstrates an invariant Dropout setup with Android Clients with the Shakespeare dataset.
The Shakespeare dataset is publicly available from the benchmarking framework [LEAF](https://github.com/TalwalkarLab/leaf)

Currently, the provided code can be run with 5 mobile clients (users with `ids` 1 to 5) without any additional data processing or `.tflite` model generation.

To run the experiment with with more clients follow the instructions in the [Processing the Dataset](#processing-the-dataset) section.

## Sample Data and Model definition
  - Inside `Shakespeare/client/app/src/main/assets/data` contains 5 sample Shakespeare user datasets that can be used for training. 
  - Inside `Shakespeare/client/app/src/main/assets/model` contains sample `.tflite` model definitions for all sub-model sizes `p=[1.0, 0.95, 0.85, 0.75, 0.65, 0.5]`.
    -  **No additional** model definition is needed to be generated to run the current code

## Processing the Dataset
1. Clone the [LEAF](https://github.com/TalwalkarLab/leaf) repo and obtain the Shakespeare data following the download instructions there.
2. The Shakespeare data processed by LEAF under `data\test` and `data\train` will generally contain lots of users combined
3. Use the `split_json_data.py` in this directory for general single-user test and training datasets
    ```shell
    python split_json_data.py --save_root <directory to save split json data> --leaf_train_json <path to the tain json file to split> --leaf_test_json <path to the test    json file to split> --val_frac <by default 0>
    ```
4. Now add the `train.json` and `test.json` from each user's dataset that you wish to run to `Shakespeare/client/app/src/main/assets/data`
5. Rename the file to `<id>_train.json` and `<id>_test.json`. The <id> refers to the (client id +1) which you'll enter on the mobile device to load data. 

## TensorFlow Lite models
Please follow the instructions in the [general README](../README.md#client-application-setup) to create the .tflite models

## Run Federated Dropout on Android Clients
Please follow the instructions in the [general README](../README.md#run-federated-dropout-on-android-clients)
