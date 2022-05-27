import time
from deepsparse import compile_model
from deepsparse.utils import generate_random_inputs

onnx_filepath = "arcfacer50.onnx"
batch_size = 1

# Generate random sample input
inputs = generate_random_inputs(onnx_filepath, batch_size)

# Compile and run
engine = compile_model(onnx_filepath, batch_size)
start = time.time()
outputs = engine.run(inputs)
end = time.time()
print(f'Time: {round((end-start)*1000, 2)}ms.')