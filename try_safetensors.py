# Let's load safetensors


import safetensors

with safetensors.safe_open("../gpt2/model.safetensors", framework="pt", device=0) as file:
    print(file.keys())
    print(type(file.get_tensor(file.keys()[0])))
    # OK, it's a torch.Tensor

    tensor = file.get_tensor(file.keys()[0])
    print(f"{tensor.dtype=} {tensor.shape=}")

    tensors = {key: file.get_tensor(key) for key in file.keys()}

print("-" * 80)

for key, tensor in tensors.items():
    print(f"{key:32}{tensor.dtype!r:32}{tensor.shape!r:32}")
# All torch.float32! I don't have to de-quantize data unlike GGUF.
