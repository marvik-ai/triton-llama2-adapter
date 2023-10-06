import os
from typing import Dict, List

import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, PreTrainedTokenizer, TensorType

class TritonPythonModel:
    tokenizer: PreTrainedTokenizer

    def initialize(self, args: Dict[str, str]) -> None:
        """
        Initialize the tokenization process
        :param args: arguments from Triton config file
        """
        # more variables in https://github.com/triton-inference-server/python_backend/blob/main/src/python.cc
        path: str = os.path.join(args["model_repository"], args["model_version"])
        self.tokenizer = AutoTokenizer.from_pretrained(path)

    def execute(self, requests) -> "List[List[pb_utils.Tensor]]":
        """
        Parse and tokenize each request
        :param requests: 1 or more requests received by Triton server.
        :return: text as input tensors
        """
        responses = []
        # for loop for batch requests (disabled in our case)
        for request in requests:
            # binary data typed back to string
            query = [
                t.decode("UTF-8")
                for t in pb_utils.get_input_tensor_by_name(request, "TEXT")
                .as_numpy()
                .tolist()
            ]
            tokens: Dict[str, np.ndarray] = self.tokenizer(
                text=query, return_tensors=TensorType.NUMPY
            )
            # tensorrt uses int32 as input type, ort uses int64
            tokens = {k: v.astype(np.int64) for k, v in tokens.items()}
            # communicate the tokenization results to Triton server
            outputs = list()
            for input_name in self.tokenizer.model_input_names:
                tensor_input = pb_utils.Tensor(input_name, tokens[input_name])
                outputs.append(tensor_input)

            inference_response = pb_utils.InferenceResponse(output_tensors=outputs)
            responses.append(inference_response)

        return responses
