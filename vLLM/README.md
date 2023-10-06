# vLLM 

## Model setup

```
model_repository/
|-- vllm
    |-- 1
    |   |-- model.py
    |-- config.pbtxt
    |-- vllm_engine_args.json
```

The `vllm_engine_args.json` file should contain the following:
```
{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "disable_log_requests": "true"
}
```

The `model.py` file intends to define the model using the TritonPythonModel class as in the Python Backend approach. Here you can find an example on how to set up a model using vLLM.

## Server setup

1. Open a new terminal and build a new docker container image derived from `tritonserver:23.09-py3`
```
docker build -t tritonserver_vllm .
```
2. Start the Triton server
```
docker run --gpus all -it --rm -p 8001:8001 --shm-size=1G --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/work -w /work tritonserver_vllm tritonserver --model-store ./model_repository
```
## Client setup

1. Open a new terminal and Start the Triton client 
```
docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:23.08-py3-sdk bash
```
2. Run your inference script
```
python3 client.py
```