# Invariant Dropout (CIFAR10)

This project demonstrates an invariant Dropout setup with Android Clients with the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

To run the experiment, please follow the instructions below to download the dataset and build the mobile application.
Currently, the provided code can be run without any additional  `.tflite` model generation.

NOTE: to partition more data on each client, the application is currently configured to only support 5 users. 
Each user with client id `<id>` loads the data from `partition_<id-1>` and `partition<id-1+5>`

## Sample Data and Model definition
  - Inside `CIFAR10/client/app/src/main/assets/model` contains sample `.tflite` model definitions for all sub-model sizes `p=[1.0, 0.95, 0.85, 0.75, 0.65, 0.5]`.
    -  **No additional** model definition is needed to be generated to run the current code
  - No sample data is included in the repo due to the large size of the file

## Processing the Dataset
1. Download the dataset from this [link](https://www.dropbox.com/s/coeixr4kh8ljw6o/cifar10.zip?dl=1)
2. Unzip and add the contents to `CIFAR10/client/app/src/main/assets/data`

## TensorFlow Lite Models
Follow the instructions in the [general README](../README.md) to create the .tflite models

## Run Federated Dropout on Android Clients
Please follow the instructions in the [general README](../README.md#run-federated-dropout-on-android-clients)
