# Tokei rikigaku...


from slope import GGUF


class Tokenizer:
    def __init__(self, gguf: GGUF) -> None:
        self._table = gguf.metadata_kv["tokenizer.ggml.tokens"]

    def number_to_token(self, number: int) -> str:
        return self._table[number]

    def _string_to_replaced_symbols(self, string: str) -> str:
        return string.replace(" ", "Ġ")
        # How to represent languages with literal "Ġ"?
        # I need more research later...

    def text_to_tokens(self, text: str) -> None:
        text = self._string_to_replaced_symbols(text)

        tokens = []
        start = 0
        stop = 1
        stop_of_last_match = 1
        while stop <= len(text):
            segment = text[start:stop]

            if segment in self._table:
                stop_of_last_match = stop

            tokens_starting_with_segment = [
                token for token in self._table
                if token.startswith(segment)
            ]
            print(f"{segment:32}{tokens_starting_with_segment}")

            if len(tokens_starting_with_segment) == 0:
                tokens.append(text[start:stop_of_last_match])
                start = stop_of_last_match
                stop = start + 1
                stop_of_last_match = start + 1
            else:
                stop += 1
        tokens.append(text[start:])

        return tokens


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", help="GGUF file name")
    args = parser.parse_args()

    gguf = GGUF(args.file_name)
    tokenizer = Tokenizer(gguf)

    print(tokenizer.text_to_tokens("Hello, world. konnichiwa"))
