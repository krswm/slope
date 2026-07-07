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

import json
import math
import pprint

import safetensors
import torch


def tprint(checkpoint_name: str, tensor: torch.Tensor):
    """Tensor print"""
    print(f"\x1b[36m{checkpoint_name} {tensor.shape}\x1b[39m")
    print(tensor)


class MyGPT2:
    def __init__(self) -> None:
        # This is where I put the model files for GPT-2 now.
        safetensors_path = "../gpt2/model.safetensors"
        config_path = "../gpt2/config.json"

        # "pt" stands for PyTorch.
        with safetensors.safe_open(safetensors_path, framework="pt") as file:
            self._tensors = {key: file.get_tensor(key) for key in file.keys()}

        print(f"\x1b[32mTensors loaded!\x1b[39m")
        for key, tensor in self._tensors.items():
            print(f"{key:32}{tensor.shape}")

        with open(config_path) as file:
            self._config = json.load(file)

        print(f"\x1b[32mConfig loaded!\x1b[39m")
        pprint.pprint(self._config)

        # Maybe this is causal mask?
        # print(self._tensors["h.0.attn.bias"])
        # Yes as I suspected!
        # But this matrix is 1 for unmasked and 0 for masked
        # not 0 for unmasked and -inf for masked
        # Maybe I have to use to multiply instead of adding?
        # At least I can ignore h.i.attn.bias
        # because it is just a simple triangular matrix
        # that doesn't contain any model information.

    def gpt(self, ids: list[int]) -> torch.Tensor:
        print(f"\x1b[32mInput\x1b[39m {ids=}")

        #### Input Embedding ####

        print(f"\x1b[32mInput embedding\x1b[39m")

        x = self._tensors["wte.weight"][ids]
        tprint("A", x)

        """
        assert(
            repr(self._tensors["wpe.weight"][:len(ids), :])
            == 
            repr(self._tensors["wpe.weight"][list(range(len(ids)))])
        )
        """

        x += self._tensors["wpe.weight"][: len(ids), :]
        tprint("B", x)

        #### Layers ####

        for i in range(self._config["n_layer"]):
            print(f"\x1b[32mLayer #{i}\x1b[39m")

            #### Attention ####

            ln_1 = torch.nn.LayerNorm(self._config["n_embd"])
            ln_1.weight = torch.nn.Parameter(
                self._tensors[f"h.{i}.ln_1.weight"]
            )
            ln_1.bias = torch.nn.Parameter(
                self._tensors[f"h.{i}.ln_1.bias"]
            )
            print("ln_1", ln_1)

            y = ln_1(x)
            tprint("C", y)

            y @= self._tensors[f"h.{i}.attn.c_attn.weight"]
            tprint("D", y)

            y += self._tensors[f"h.{i}.attn.c_attn.bias"]
            tprint("E", y)

            """
            print(type(y.split(3)))
            print([z.shape for z in y.split(3)])
            # print([z.shape for z in y.split(3, dim=1)])
            print([z.shape for z in y.split(self._config["n_embd"], dim=1)])
            """

            q, k, v = y.split(self._config["n_embd"], dim=1)
            print("F", q.shape, k.shape, v.shape)

            y = q
            tprint("G", y)

            y @= k.T
            tprint("H", y)

            y /= math.sqrt(len(ids))
            tprint("I", y)

            y += (torch.ones(len(ids), len(ids)) * -1e-12).triu(diagonal=1)

            y = y.softmax(0)
            tprint("J", y)

            y @= v
            tprint("K", y)

            y @= self._tensors[f"h.{i}.attn.c_proj.weight"]
            tprint("L", y)

            y += self._tensors[f"h.{i}.attn.c_proj.bias"]
            tprint("M", y)

            x += y
            tprint("N", x)

            #### Feed Forward ####

            ln_2 = torch.nn.LayerNorm(self._config["n_embd"])
            ln_2.weight = torch.nn.Parameter(
                self._tensors[f"h.{i}.ln_2.weight"]
            )
            ln_2.bias = torch.nn.Parameter(
                self._tensors[f"h.{i}.ln_2.bias"]
            )
            print("ln_2", ln_2)

            y = ln_2(x)
            tprint("O", y)

            y @= self._tensors[f"h.{i}.mlp.c_fc.weight"]
            tprint("P", y)

            y += self._tensors[f"h.{i}.mlp.c_fc.bias"]
            tprint("Q", y)

            gelu = torch.nn.GELU()
            y = gelu(y)
            tprint("R", y)

            y @= self._tensors[f"h.{i}.mlp.c_proj.weight"]
            tprint("S", y)

            y += self._tensors[f"h.{i}.mlp.c_proj.bias"]
            tprint("T", y)

            x += y

        #### Output embedding ####

        print(f"\x1b[32mOutput embedding\x1b[39m")

        ln_f = torch.nn.LayerNorm(self._config["n_embd"])
        ln_f.weight = torch.nn.Parameter(self._tensors["ln_f.weight"])
        ln_f.bias = torch.nn.Parameter(self._tensors["ln_f.bias"])
        print("ln_f", ln_f)

        x = ln_f(x)
        tprint("U", x)

        x @= self._tensors["wte.weight"].T
        tprint("V", x)

        return x

        # Done. Kind of.


if __name__ == "__main__":
    my_gpt2 = MyGPT2()
    my_gpt2.gpt([1234, 5678])
