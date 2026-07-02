# The GGUF specification:
# https://github.com/ggml-org/ggml/blob/master/docs/gguf.md


import struct


class GGUF:
    def __init__(self, file_name: str) -> None:
        self._file_name = file_name
        self._parse()

    def _parse(self) -> None:
        with open(self._file_name, "rb") as file:
            (
                self.magic_number,
                self.version,
                self.tensor_count,
                self.metadata_kv_count,
            ) = struct.unpack("<4sLQQ", file.read(24))

    def __repr__(self) -> str:
        return " ".join(
            f"{attr}={getattr(self, attr)}"
            for attr in [
                "magic_number", "version", "tensor_count", "metadata_kv_count"
            ]
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", help="GGUF file name")
    args = parser.parse_args()

    gguf = GGUF(args.file_name)
    print(gguf)
