# Tokei rikigaku...


from slope import GGUF


class Tokenizer:
    def __init__(self, gguf: GGUF) -> None:
        self._gguf = gguf

    def number_to_token(self, number: int) -> None:
        return self._gguf.metadata_kv["tokenizer.ggml.tokens"][number]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", help="GGUF file name")
    args = parser.parse_args()

    gguf = GGUF(args.file_name)
    tokenizer = Tokenizer(gguf)

    print(tokenizer.number_to_token(51367))
