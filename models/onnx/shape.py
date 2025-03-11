import onnx
model = onnx.load("model.onnx")
input_info = [(inp.name, [dim.dim_value if dim.dim_value else -1 for dim in inp.type.tensor_type.shape.dim]) for inp in model.graph.input]
output_info = [(out.name, [dim.dim_value if dim.dim_value else -1 for dim in out.type.tensor_type.shape.dim]) for out in model.graph.output]

print("Input Shape:", input_info)
print("Output Shape:", output_info)
