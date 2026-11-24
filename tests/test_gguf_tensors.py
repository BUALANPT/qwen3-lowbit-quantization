import struct
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from gguf_tensors import read_tensor_headers


def string(value):
    return struct.pack('<Q', len(value)) + value


class GgufTensorTests(unittest.TestCase):
    def test_reads_tensor_offsets_through_variable_metadata_and_alignment(self):
        header = b'GGUF' + struct.pack('<IQQ', 3, 1, 2)
        header += string(b'tokenizer.ggml.tokens') + struct.pack('<IIQ', 9, 8, 2)
        header += string(b'alpha') + string(b'beta')
        header += string(b'general.alignment') + struct.pack('<II', 4, 64)
        header += string(b'blk.3.ffn_down.weight') + struct.pack('<IQQIQ', 2, 256, 4, 12, 0)
        with TemporaryDirectory() as directory:
            target = Path(directory) / 'fixture.gguf'
            target.write_bytes(header.ljust(192, b'\0') + bytes(576))
            item, = read_tensor_headers(target)
            self.assertEqual((item.name, item.kind, item.shape, item.nbytes, item.offset),
                             ('blk.3.ffn_down.weight', 'Q4_K', (256, 4), 576, 192))
