# Python Backend

This guide will focus on how to deploy HuggingFace models with NVIDIA Triton Python Backend. For this example, the `Llama2` model will be used.

## Model setup

For making use of Triton's python backend, the first step is to wrap the model using the TritonPythonModel class and include the following functions:

- `initialize()` - This function is executed when Triton loads the model. It is generally used for loading any model or data needed. The use of this function is optional.

- `execute()` - This function is executed upon every request. It usually contains the complete pipeline logic. 

The second step is to create a configuration file for the model. The purpose of this file is for Triton to understand how to process the model. It usually includes specifications for the inputs and outputs of the models, the runtime environment and the necessary hardware resources.

Finally, the files should follow the following structure:

```
model_repository
|-- 1
|  |-- model.py
|-- config.pbtxt
````

## Server setup

1. Open a new terminal and run the following command to setup the docker container for Triton Server

```
docker run --gpus=all -it --shm-size=1g --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:23.08-py3 bash
```

2. Install the following dependencies:
```
pip install app
pip install torch
pip install transformers
pip install huggingface_hub
pip install accelerate
pip install bitsandbytes
pip install scipy
```

3. Start up Triton Server
```
tritonserver --model-repository=/models --log-verbose=2
```

## Client setup
1. Open a new terminal and run docker container for Triton Client
```
docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:23.08-py3-sdk bash
```
2. Run your inference script
```
python3 client.py --model_name "llamav2"
```