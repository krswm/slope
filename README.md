# GPT-2 Inference with tenferro

I built a GPT-2 inference engine from scratch in Rust.

## Quickstart

I made this project just for educational purpose. Use at your own risk!

### Clone this repository.

```
git clone https://github.com/krswm/slope
```

### Download the pretrained GPT-2 model.

We’ll install the model from [Hugging Face](https://huggingface.co/openai-community/gpt2).

Install [the Git Xet extention](https://huggingface.co/docs/hub/main/en/xet/using-xet-storage#git).
If you're using macOS and Homebrew:

```
brew install git-xet
git xet install
```

Clone the model repository. It may take a while.

```
git clone https://huggingface.co/openai-community/gpt2
```

### Start generating text!

```
cd slope
cargo run ../gpt2 'Artificial intelligence is'
```

### Credits

- [GPT-2](https://huggingface.co/openai-community/gpt2) for devising an influental LLM architecture.
- [*GPT in 60 Lines of NumPy*](https://jaykmody.com/blog/gpt-from-scratch/) (a blog post) for teaching me how to implement a GPT-2 inference engine from scratch.
- [*Implementing A Byte Pair Encoding (BPE) Tokenizer From Scratch*](https://sebastianraschka.com/blog/2025/bpe-from-scratch.html) (a blog post) for teaching me how to implement a BPE tokenizer from scratch.
- [tenferro](https://github.com/tensor4all/tenferro-rs) for providing me an amazing tensor library for Rust!

## My Attempt

I want to build a local LLM software
that loads pretrained models
and generates text with it.
Such thing is called an inference engine.
I want to achieve it from scratch.

I used `ollama` to observe its behavior as an LLM inference engine.
Except for this, I did *not* use generative AI for this project.
