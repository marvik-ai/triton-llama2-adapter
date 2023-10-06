# Ensemble model

This approach requires first converting the model into a serialized representation, such as ONNx, before deploying it on the Triton server. Once converted, there are two ways of deploy the model onto Triton server:

- Client-side tokenizer: Only the model is deployed onto the Triton server, while the tokenization is handled entirely on the client side. 
- Server-side tokenizer: Both the tokenizer and the model are deployed on the server. 

## Model setup

The model repository should contain three different folders with the following structure:

```
model_repository/
|-- ensemble_model
|   |-- 1
|   |-- config.pbtxt
|-- model
|    |-- 1
|       |-- llamav2.onnx
|  |-- config.pbtxt
|-- tokenizer
|  |-- 1
|  | |-- config.json
|  | |-- model.py
|  | |-- special_tokens_map.json
|  | |-- tokenizer.json
|  |-- config.pbtxt
```


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