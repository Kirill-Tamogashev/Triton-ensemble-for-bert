from pathlib import Path
from typing import Dict, List

import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import BertTokenizer, PreTrainedTokenizer


class TritonPythonModel:
    tokenizer: PreTrainedTokenizer

    def initialize(self, args: Dict[str, str]) -> None:
        """
        Initialize the tokenization process
        :param args: arguments from Triton config file
        """
        vocab_path: str = Path(args["model_repository"]).joinpath(
            args["model_version"],
            "vocab.txt"
            )
        self.tokenizer = BertTokenizer(vocab_path.as_posix())

    def execute(self, requests) -> "List[List[pb_utils.Tensor]]":
        """
        Parse and tokenize each request
        :param requests: 1 or more requests received by Triton server.
        :return: text as input tensors
        """
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
            query: List[List[int]] = [
                t.decode("UTF-8")
                for t in input_tensor.as_numpy().tolist()
            ]
            tokens: Dict[str, list] = self.tokenizer(
                text=query
            )
            outputs: List[pb_utils.Tensor]= [
                pb_utils.Tensor(input_name, np.array(token).astype(np.int64))
                for input_name, token in tokens.items()
            ]
 
            inference_response = pb_utils.InferenceResponse(output_tensors=outputs)
            responses.append(inference_response)

        return responses
