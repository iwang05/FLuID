# Invariant Dropout (FEMNIST)

This project demonstrates an invariant Dropout setup with Android Clients with the FEMNIST dataset.
The FEMNIST dataset is from the benchmarking framework [LEAF](https://github.com/TalwalkarLab/leaf)

Currently, the provided code can be run with 5 mobile clients (users with `ids` 1 to 5) without any additional data processing or `.tflite` model generation. 

To run the experiment with with more clients follow the instructions in the [Processing the Dataset](#processing-the-dataset) section.

## Sample Data and Model definition
  - Inside `FEMNIST/client/app/src/main/assets/data` contains 5 sample FEMNIST user datasets that can be used for training. 
  - Inside `FEMNIST/client/app/src/main/assets/model` contains sample `.tflite` model definitions for all sub-model sizes `p=[1.0, 0.95, 0.85, 0.75, 0.65, 0.5]`.
    -  **No additional** model definition is needed to be generated to run the current code

## Processing the Dataset
1. Clone the [LEAF](https://github.com/TalwalkarLab/leaf) repo and obtain the FEMNIST following the download instructions there
2. Directly add the FEMNIST data (`.json` files) processed by LEAF under `data\test` and `data\train` to `FEMNIST/client/app/src/main/assets/data`
5. Rename the file to `<id>_train.json` and `<id>_test.json`. The <id> refers to the (client id +1) which you'll enter on the mobile device to load data. 

## TensorFlow Lite models
Follow the instructions in the [general README](../README.md) to create the .tflite models

## Run Federated Dropout on Android Clients
Please follow the instructions in the [general README](../README.md#run-federated-dropout-on-android-clients)

