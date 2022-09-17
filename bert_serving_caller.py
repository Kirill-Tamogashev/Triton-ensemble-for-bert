import tritonclient.http as triton_client
import tritonclient
import tensorflow as tf
import time
import random

from tritonclient.utils import *
import tritonclient.http as httpclient
import sys

import numpy as np


with httpclient.InferenceServerClient("localhost:8000") as client:
    N = 123
    input_string = [''.join(random.choices("абвгвжлппрашыфл", k=10)) for _ in range(N)]
    model_in = triton_client.InferInput("TEXT", [N], "BYTES")
    model_in.set_data_from_numpy(np.asarray(input_string, dtype="object").reshape(N))
    inputs =[model_in]
    model_name = "ensemble_model"
    outputs = [
        httpclient.InferRequestedOutput("output_0"),
    ]

    response = client.infer(model_name,
                            inputs,
                            request_id=str(1),
                            outputs=outputs)

    result = response.get_response()
    output_data = response.as_numpy("output_0")
    print(output_data.shape)


    print('PASS: add_sub')
    sys.exit(0)









