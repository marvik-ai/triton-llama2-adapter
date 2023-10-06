from tritonclient.utils import *
import tritonclient.http as httpclient
import time
import numpy as np

tm1 = time.perf_counter()
with httpclient.InferenceServerClient(url="localhost:8000", verbose=False, concurrency=32) as client:
    
    # Define input config
    input_text =[["Where is Uruguay?"],
                 ["Who is George Washington?"],
                 ["Who is Lionel Messi?"],
                 ]
                  
    text_obj = np.array(input_text, dtype="object")

    inputs = [
        httpclient.InferInput("prompt", text_obj.shape, np_to_triton_dtype(text_obj.dtype)).set_data_from_numpy(text_obj),
    ]

    # Define output config
    outputs = [
        httpclient.InferRequestedOutput("generated_text"),
    ]
    
    
    # Hit triton server
    n_requests = 2
    responses = []
    for i in range(n_requests):
        responses.append(client.async_infer('llamav2', model_version='1', inputs=inputs, outputs=outputs))
        

for r in responses: 
    result = r.get_result()
    content = result.as_numpy('generated_text')
    print(content)
    
tm2 = time.perf_counter()
print(f'Total time elapsed: {tm2-tm1:0.2f} seconds')