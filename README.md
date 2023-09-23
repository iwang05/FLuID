# Federated Learning using Invariant Dropout (FLuID)

This repository contains the source code for the NeurIPS 2023 paper "FLuID: Mitigating Stragglers in Federated Learning using Invariant Dropout" by Irene Wang, Prashant J. Nair, and Divya Mahajan.
[[pdf]](https://arxiv.org/pdf/2307.02623)

This project is implemented using the [Flower framework v0.18.0](https://github.com/adap/flower). Also uses Flower's [Android example](https://flower.dev/blog/2021-12-15-federated-learning-on-android-devices-with-flower/) as an implementation basis.

## Citation
If you use FLuID in your research, please cite our paper:

```
@inproceedings{fluid,
    author={Irene Wang and Prashant J. Nair and Divya Mahajan},
    booktitle={Advances in Neural Information Processing Systems},
    title={FLuID: Mitigating Stragglers in Federated Learning using Invariant Dropout}, 
    year={2023}
}
```
## Installation

The project requires **Python >=3.7, <=3.9**, and uses **Gradle** to build the client mobile app. 
The mobile application has been verified to be working for devices with at least Android 9 (sdk 28)

Install FLuID and other project dependencies

```shell
pip install flwr==0.18.0
pip install tensorflow
git clone https://github.com/iwang05/FLuID.git
cd FLuID
```

## Quick Start
**A reminder that the technique requires a minimum of 2 clients for training**

###  Building the Client Application

1. To Install the application on an Android Device, first do the following:
    1. Enable `Developer Mode` and `USB debugging` on the Android Device
    2. Connect a mobile device wirelessly or using a USB cord

2. There are two options for building the client application:
    1. Using Android Studio
        1. In Android Studio open the project at `<Dataset>/client/`
        2. Use the `Run App` function on Android Studio to build and install the application
    2. Using Gradlew
        ```shell
            cd <Dataset>/client/
            gradlew installDebug
        ```
        - If on Max or Linux run `./gradlew installDebug`
        
### Server setup 
Configure `<dataset>/server.py`
  - Specify the number of clients to run with `min_fit_clients`, `min_eval_clients`, and `min_available_clients`,
  - Specify the server's IP address, and the number of rounds to run:
  ```shell 
  fl.server.start_server("192.168.1.7:1999", config={"num_rounds": 10}, server=server, strategy=strategy)
  ```         
### Run FLuID on Android Clients

1. To start the server, in `FLuID/<dataset>` run
```shell
python server.py
```
2. Open the Client app corresponding on your phone
3. Add the client ID (between 1-10), the IP and port of your server, and press `Load Dataset`. This will load the local dataset in memory.
4. Then press `Setup Connection Channel` which will establish a connection with the server.
5. Finally, press `Train Federated!` which will start the federated training. 

## Custom Configurations

### 1. Change Dropout method
Change the dropout method in `fedDrop<dataset>_android.py`
  - This is the actual implementation of the dropout methods
  - In the method `configure_fit` Select the desired dropout method
  ```shell 
  fit_ins_drop = FitIns(self.drop_rand(parameters, self.p_val, [0,3], 10, client.cid), config_drop)
  ```
### 2. Generating additional sub-model sizes
**NOTE**:  **No additional** model definition needed to be generated to run the current code.
  - All 6 required model definitions to run the current code `p=[1.0, 0.95, 0.85, 0.75, 0.65, 0.5]` are included in `<Dataset>/client/app/src/main/assets/model`
  - To modify the model, or add new sub-model sizes, you would need to generate `.tflite` files for all sub-model sizes that you wish to run, plus the full model size `p=1.0`
     
To generate more model definitions for sub-model sizes: 
1. Define the modle in `<Dataset>/tflite_convertor/convert_to_tflite.py`.  
2. Vary the `p` variable to create models of different sizes (1.0 = full model, 0.5 = model with half the size)
3. Execute the script, and models will be created in the `tflite_convertor/tflite_model` folder
  ```shell
  python convert_to_tflite.py
  ```
4. Rename each file as `<p>_<original file name>.tflite`. 
   - For example, a the `train_head.tflite` file create with `p=0.75` would be renamed as `0.75_train_head.tflite`
6. Add files to `<Dataset>/client/app/src/main/assets/model` directory

### 3. Adding additional data files
   - Follow the `Processing the Dataset` instructions in the README files for each dataset subfolder
   - Add the downloaded datasets to `<Dataset>/client/app/src/main/assets/data` directory

## Datasets
This project contains the implementation of the FLuID framework for CIFAR10, FEMNIST, and Shakespeare and also implements baseline techniques for Random and Ordered Dropout.

These datasets are all publicly available for download.
The Shakespeare and FEMNIST datasets can be obtained from the [official LEAF Repo](https://github.com/TalwalkarLab/leaf). 
Please follow the instructions there to download the LEAF datasets there
   - 5 example user datasets are included for each dataset in `<Dataset>/client/app/src/main/assets/data`

The [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) Implementation uses the dataset partitioned by and provided by the Flower Framework and can be downloaded from this [link](https://www.dropbox.com/s/coeixr4kh8ljw6o/cifar10.zip?dl=1)

**Quick start instructions above outline the general steps to run FLuID, for dataset-specific processing instructions please refer to the README in each dataset subfolder.**

## Troubleshooting

1. In case of an `SDK location not found` Error when building the Client Application, create a file `local.properties` file in`<Dataset>/client/` with the following line:
```shell
sdk.dir=<sdk dir path>
```
where `<sdk.dir path>` is the path of where your Android SDK is installed.

## License 
This source code is licensed under the Apache License, Version 2.0 found in the LICENSE file in the root directory.
  




