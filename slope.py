# The GGUF specification:
# https://github.com/ggml-org/ggml/blob/master/docs/gguf.md


import io
import struct


class ArrayEncountered(Exception):
    pass


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

            self.metadata_kv = {}
            for _ in range(self.metadata_kv_count):
                try:
                    key, value = self._parse_metadata_kv(file)
                    self.metadata_kv[key] = value
                except ArrayEncountered:
                    print("array encountered")
                    break

    def _parse_metadata_kv(self, file: io.BufferedReader) -> None:
        key_len, = struct.unpack("<Q", file.read(8))
        key = file.read(key_len).decode()

        value_type, = struct.unpack("<L", file.read(4))
        match value_type:
            case 0:  # UINT8
                value, = struct.unpack("<B", file.read(1))
            case 1:  # INT8
                value, = struct.unpack("<b", file.read(1))
            case 2:  # UINT16
                value, = struct.unpack("<H", file.read(2))
            case 3:  # INT16
                value, = struct.unpack("<h", file.read(2))
            case 4:  # UINT32
                value, = struct.unpack("<L", file.read(4))
            case 5:  # INT32
                value, = struct.unpack("<l", file.read(4))
            case 6:  # FLOAT32
                value, = struct.unpack("<f", file.read(4))
            case 7:  # BOOL
                value, = struct.unpack("<?", file.read(1))
            case 8:  # STRING
                value_len, = struct.unpack("<Q", file.read(8))
                value = file.read(value_len).decode()
            case 9:  # ARRAY
                # Hmmm...
                # The spec says array can be nested.
                # I need to rework the parsing algorithm later...
                # I'll stop parsing the KV just for now.
                raise ArrayEncountered
            case 10:  # UINT64
                value, = struct.unpack("<Q", file.read(8))
            case 11:  # INT64
                value, = struct.unpack("<q", file.read(8))
            case 12:  # FLOAT64
                value, = struct.unpack("<d", file.read(8))

        return key, value
        

    def __repr__(self) -> str:
        return " ".join(
            f"{attr}={getattr(self, attr)}"
            for attr in [
                "magic_number",
                "version",
                "tensor_count",
                "metadata_kv_count",
                "metadata_kv",
            ]
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", help="GGUF file name")
    args = parser.parse_args()

    gguf = GGUF(args.file_name)
    print(gguf)
