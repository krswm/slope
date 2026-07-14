# GPT-2 Inference with tenferro

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

Clone the model repository.

```
git clone https://huggingface.co/openai-community/gpt2
```

### Start generating text!

```
cd slope
cargo run ../gpt2 'Artificial intelligence is'
```

## My Attempt

I want to build a local LLM software
that loads pretrained models
and generates text with it.
Such thing is called an inference engine.
I want to achieve it from scratch.

I used `ollama` to observe its behavior as an LLM inference engine.
Except for this, I did *not* use generative AI for this project.
