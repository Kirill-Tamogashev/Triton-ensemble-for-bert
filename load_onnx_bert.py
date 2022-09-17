from pathlib import Path
from transformers.convert_graph_to_onnx import convert

convert(framework="pt", model="DeepPavlov/rubert-base-cased", output=Path("model_repository/model/1/rubert-base-cased.onnx"), opset=11)
