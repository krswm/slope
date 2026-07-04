import unittest

from slope import GGUF
from tokei import Tokenizer


class Test(unittest.TestCase):
    def setUp(self) -> None:
        gguf = GGUF("../tmp-model/llama3:8b.gguf")  # Outside of the project direcory!
        self.tokenizer = Tokenizer(gguf)

    # I obtained the expected result from Tiktokenizer:
    # https://tiktokenizer.vercel.app/?model=meta-llama%2FMeta-Llama-3-8B

    def test_number_to_token(self) -> None:
        self.assertEqual(self.tokenizer.number_to_token(51367), "foobar")

    # I was surprized that a nonsense word "foobar" has its own token!
