# GPT-2 Inference with tenferro

I built a GPT-2 inference engine from scratch in Rust.

## Quickstart

I made this project just for educational purpose. Use at your own risk.

### Step 1: Clone this repository.

```
git clone https://github.com/krswm/slope
```

### Step 2: Download the pretrained GPT-2 model.

We will install the model from [Hugging Face](https://huggingface.co/openai-community/gpt2).

First, install [the Git Xet extention](https://huggingface.co/docs/hub/main/en/xet/using-xet-storage#git).
If you are using macOS and Homebrew:

```
brew install git-xet
git xet install
```

Next, clone the model repository. It may take a while to complete.

```
git clone https://huggingface.co/openai-community/gpt2
```

### Step 3: Start generating text.

The GPT-2 model is not for chat conversation, but for text continuation.
Watch the model continues your prompt.

```
cd slope
cargo run ../gpt2 'Artificial intelligence is'
```

## Source Files

- <src/loader.rs> loads a [Safetensors](https://github.com/safetensors/safetensors) file and convert the tensors into tenferro’s `TypedTensor`.
- <src/tokenizer.rs> tokenizes your prompt with the BPE algorithm.
- <src/transformer.rs> is the heart of the GPT-2 inferenece. It receives tokens (your prompt + already generated text) and predicts the next token.
- <src/main.rs> loads files from the GPT-2 repository and generates text.

## My Future Plans

- Make the text generation not deterministic. Apply stochasticity for sampling.
- Optimize the inference. Use techniques such as KV-cache.
- Use GPU backend. tenferro provides CUDA and WebGPU backend. (I’m not sure whether I have an access for such hardware though…)
- Support variety of LLM architectures beyond GPT-2 such as Llama 2.
- Ultimately, implement LLM training from scratch. (I’m interested on [TinyStories](https://arxiv.org/pdf/2305.07759).)

## Credits

- [GPT-2](https://huggingface.co/openai-community/gpt2) for devising an influental LLM architecture.
- [*GPT in 60 Lines of NumPy*](https://jaykmody.com/blog/gpt-from-scratch/) (a blog post) for teaching me how to implement a GPT-2 inference engine from scratch.
- [*Implementing A Byte Pair Encoding (BPE) Tokenizer From Scratch*](https://sebastianraschka.com/blog/2025/bpe-from-scratch.html) (a blog post) for teaching me how to implement a BPE tokenizer from scratch.
- [tenferro](https://github.com/tensor4all/tenferro-rs) for providing me an amazing tensor library for Rust.

## Development

This is a hobby project of mine I started from scratch.

I started this project on 2026-07-03 and finished my first implementation on 2026-07-14.

I used open source LLM inference engines (Ollama, etc.) and open source LLM models (TinyLlama, GPT-2, etc.) only for the purpose to observe their behavior as LLM architecture.
Except for that, I did **not** use generative AI for this project at all.
