"""Read the tensor directory of little-endian GGUF v3 without mapping weights."""

from dataclasses import dataclass
from math import prod
from pathlib import Path
from struct import Struct

from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType, GGUFValueType


U32 = Struct('<I')
U64 = Struct('<Q')
FIELD_WIDTH = {GGUFValueType.UINT8: 1, GGUFValueType.INT8: 1, GGUFValueType.BOOL: 1,
               GGUFValueType.UINT16: 2, GGUFValueType.INT16: 2,
               GGUFValueType.UINT32: 4, GGUFValueType.INT32: 4, GGUFValueType.FLOAT32: 4,
               GGUFValueType.UINT64: 8, GGUFValueType.INT64: 8, GGUFValueType.FLOAT64: 8}


@dataclass(frozen=True)
class TensorHeader:
    name: str
    kind: str
    shape: tuple[int, ...]
    nbytes: int
    offset: int


def read_tensor_headers(path: Path) -> list[TensorHeader]:
    file_size = path.stat().st_size
    with path.open('rb') as file:
        def read_exact(size):
            data = file.read(size)
            if len(data) != size:
                raise ValueError(f'Truncated GGUF header: {path}')
            return data

        def uint(layout):
            return layout.unpack(read_exact(layout.size))[0]

        def skip(size):
            if file.tell() + size > file_size:
                raise ValueError(f'Truncated GGUF metadata: {path}')
            file.seek(size, 1)

        def read_name():
            size = uint(U64)
            if size > file_size - file.tell():
                raise ValueError(f'Invalid GGUF name length: {path}')
            return read_exact(size).decode('utf-8')

        if read_exact(4) != b'GGUF' or uint(U32) != 3:
            raise ValueError(f'Expected a little-endian GGUF v3 file: {path}')
        tensor_count, fields = uint(U64), uint(U64)
        alignment = 32
        for _ in range(fields):
            name, kind = read_name(), GGUFValueType(uint(U32))
            if name == 'general.alignment':
                if kind != GGUFValueType.UINT32:
                    raise ValueError('Expected a UINT32 GGUF alignment')
                alignment = uint(U32)
            elif kind == GGUFValueType.STRING:
                skip(uint(U64))
            elif kind == GGUFValueType.ARRAY:
                element, count = GGUFValueType(uint(U32)), uint(U64)
                if element == GGUFValueType.STRING:
                    for _ in range(count):
                        skip(uint(U64))
                elif element in FIELD_WIDTH:
                    skip(FIELD_WIDTH[element] * count)
                else:
                    raise ValueError(f'Unsupported GGUF array element type: {element.name}')
            else:
                skip(FIELD_WIDTH[kind])

        tensors = []
        for _ in range(tensor_count):
            name = read_name()
            dimensions = uint(U32)
            if not 1 <= dimensions <= 4:
                raise ValueError(f'Invalid number of dimensions for {name}')
            shape = tuple(uint(U64) for _ in range(dimensions))
            kind, offset = GGMLQuantizationType(uint(U32)), uint(U64)
            block, block_bytes = GGML_QUANT_SIZES[kind]
            elements = prod(shape)
            if not elements or shape[0] % block:
                raise ValueError(f'Invalid tensor block shape for {name}')
            tensors.append((name, kind.name, shape, elements // block * block_bytes, offset))

        if alignment == 0 or alignment & (alignment - 1):
            raise ValueError(f'Unsupported GGUF alignment: {alignment}')
        first_byte = (file.tell() + alignment - 1) // alignment * alignment
        headers = [TensorHeader(name, kind, shape, nbytes, first_byte + offset)
                   for name, kind, shape, nbytes, offset in tensors]
        if any(tensor.offset + tensor.nbytes > file_size for tensor in headers):
            raise ValueError(f'Truncated GGUF tensor data: {path}')
        return headers
