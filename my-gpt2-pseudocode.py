# I learned a lot about the GPT-2 architecture from this blog post from January 2023.
# https://jaykmody.com/blog/gpt-from-scratch/

# I did not use generative AI for this file.

# 2026-07-07


"""
class GPT:
    def __init__(self):
        # Imagine we load the GPT-2 model here.
        self.tensors = get_tensors()

        # https://docs.pytorch.org/docs/2.12/generated/torch.nn.GELU.html
        self.gelu = torch.nn.GELU()

        # https://docs.pytorch.org/docs/2.12/generated/torch.nn.Softmax.html
        self.softmax = torch.nn.Softmax(2)

    def gpt(self, ids):
        x = self.tensors["wte"][ids]
        x += self.tensors["wpe"][:len(ids), :]

        # Is num_blocks stored on metadata? I have to check later.
        for i in range(self.num_blocks):
            # Self attention
            y = self.layer_norm(
                x, self.tensors[f"h{i}/ln_1/b"], self.tensors[f"h{i}/ln_1/g"]
            )
            y @= self.tensors[f"h{i}/attn/c_attn/w"]
            y += self.tensors[f"h{i}/attn/c_attn/b"]
            # query, key, and value
            q, k, v = torch.split(y, 3)  # I'm not sure whether this syntax is correct...
            z = q @ k.transpose(0, 1)
            z /= torch.sqrt(self.num_embedding)
            z = self.softmax(z)
            z @= v
            z @= self.tensors[f"h{i}/mlp/c_proj/w"]
            z += self.tensors[f"h{i}/mlp/c_proj/b"]
            x += z

            # Feed forward
            y = self.layer_norm(
                x, self.tensors[f"h{i}/ln_2/b"], self.tensors[f"h{i}/ln_2/g"]
            )
            y @= self.tensors[f"h{i}/mlp/c_fc/w"]
            y += self.tensors[f"h{i}/mlp/c_fc/b"]
            y = self.gelu(y)
            y @= self.tensors[f"h{i}/mlp/c_proj/w"]
            y += self.tensors[f"h{i}/mlp/c_proj/b"]
            x += y

        x = self.LayerNorm(x, self.tensors["ln_f/b"], self.tensors["ln_f/g"])
        x @= self.tensors["wte"].transpose(0, 1)

        return x

        # It's not super complicated than I imagined before

    def LayerNorm(self, x, b, g):
"""

import safetensors
import torch


class MyGPT2:
    def __init__(self) -> None:
        # This is where I put the Safetensors file now.
        path = "../gpt2/model.safetensors"

        # "pt" stands for PyTorch.
        with safetensors.safe_open(path, framework="pt") as file:
            self._tensors = {key: file.get_tensor(key) for key in file.keys()}

        print(f"\x1b[32mModel loaded!\x1b[39m")
        for key, tensor in self._tensors.items():
            print(f"{key:32}{tensor.shape}")

    def gpt(self, ids: list[int]) -> torch.Tensor:
        print(f"\x1b[32mInput\x1b[39m {ids=}")

        #### Embedding ####

        x = self._tensors["wte.weight"][ids]
        print("A", x.shape)

        """
        assert(
            repr(self._tensors["wpe.weight"][:len(ids), :])
            == 
            repr(self._tensors["wpe.weight"][list(range(len(ids)))])
        )
        """

        x += self._tensors["wpe.weight"][: len(ids), :]
        print("B", x.shape)


if __name__ == "__main__":
    my_gpt2 = MyGPT2()
    my_gpt2.gpt([1234, 5678])
