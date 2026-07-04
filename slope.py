# The GGUF specification:
# https://github.com/ggml-org/ggml/blob/master/docs/gguf.md


import io
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

            self.metadata_kv = {}
            for _ in range(self.metadata_kv_count):
                key, value = self._parse_metadata_kv(file)
                self.metadata_kv[key] = value

            self.tensor_infos = [
                self._parse_tensor_info(file) for _ in range(self.tensor_count)
            ]

    def _parse_metadata_kv(self, file: io.BufferedReader) -> None:
        key_len, = struct.unpack("<Q", file.read(8))
        key = file.read(key_len).decode()

        value_type, = struct.unpack("<L", file.read(4))
        value = self._parse_metadata_value(file, value_type)

        return key, value

    def _parse_metadata_value(
        self, file: io.BufferedReader, value_type: int
    ) -> None:
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
                # Recursion!
                array_type, array_len = struct.unpack("<LQ", file.read(12))
                value = [
	            self._parse_metadata_value(file, array_type)
                    for _ in range(array_len)
                ]
            case 10:  # UINT64
                value, = struct.unpack("<Q", file.read(8))
            case 11:  # INT64
                value, = struct.unpack("<q", file.read(8))
            case 12:  # FLOAT64
                value, = struct.unpack("<d", file.read(8))
        return value

    def _parse_tensor_info(self, file: io.BufferedReader) -> None:
        string_len, = struct.unpack("<Q", file.read(8))
        string = file.read(string_len).decode()

        n_dimensions, = struct.unpack("<L", file.read(4))
        dimensions = list(
            struct.unpack(f"<{n_dimensions}Q", file.read(8 * n_dimensions))
        )

        tensor_type, offset = struct.unpack("<LQ", file.read(12))

        return (string, dimensions, tensor_type, offset)

    def __repr__(self) -> str:
        return " ".join(
            f"{attr}={getattr(self, attr)}"
            for attr in [
                "magic_number",
                "version",
                "tensor_count",
                "metadata_kv_count",
                "metadata_kv",
                "tensor_infos",
            ]
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", help="GGUF file name")
    parser.add_argument(
        "--action",
        choices=["print-contents", "print-metadata", "print-tensor-infos"],
        default="print-contents",
    )
    parser.add_argument("--metadata-key")
    args = parser.parse_args()

    gguf = GGUF(args.file_name)
    if args.metadata_key:
        print(gguf.metadata_kv[args.metadata_key])
    else:
        match args.action:
            case "print-contents":
                print(gguf)
            case "print-metadata":
                for key, value in gguf.metadata_kv.items():
                    if isinstance(value, list):
                        print(f"{key:64}<list>")
                    else:
                        print(f"{key:64}{value!r}")
            case "print-tensor-infos":
                for string, dimensions, tensor_type, offset in gguf.tensor_infos:
                    print(f"{string:32}{dimensions!r:32}{tensor_type:4}{offset:32}")
