"""Bounded, append-only FP32 shards; no pickle and no full-tensor concatenation."""
import hashlib
from collections import OrderedDict
import math
from pathlib import Path
import numpy as np
from .common import require, digest, validate_artifact_path

CHUNK_ELEMENTS = 1048576


class ShardWriter:
    def __init__(self, folder, rank, max_bytes, fields=("gradients", "updates")):
        self.folder = Path(folder)
        self.rank = rank
        self.max_bytes = max_bytes
        self.allowed_fields = tuple(fields)
        require(self.allowed_fields and all(f in ("gradients", "updates", "parameters", "master_parameters", "exp_avg", "exp_avg_sq")
                                            for f in self.allowed_fields), "Unsupported vector fields")
        self.bytes = 0
        self.files, self.hashes, self.records = {}, {}, []

    def append(self, field, piece, chunks):
        require(field in self.allowed_fields, "Unsupported vector field")
        if field not in self.files:
            path = self.folder / f"{field}.rank{self.rank}.f32.partial"
            self.files[field] = path.open("xb")
            self.hashes[field] = hashlib.sha256()
        stream = self.files[field]
        offset = stream.tell()
        count = 0
        for chunk in chunks:
            values = np.asarray(chunk, dtype="<f4").reshape(-1)
            require(np.isfinite(values).all(), f"Nonfinite {field} in {piece['name']}")
            data = values.tobytes()
            require(self.bytes + len(data) <= self.max_bytes, "Stage raw-data budget exceeded")
            stream.write(data)
            self.hashes[field].update(data)
            self.bytes += len(data)
            count += values.size
        require(count == piece["end"] - piece["start"], "Written tensor length differs from mapping")
        self.records.append({**piece, "field": field, "file": f"{field}.rank{self.rank}.f32",
                             "offset_bytes": offset, "elements": count})

    def close(self):
        files = {}
        for field, stream in self.files.items():
            stream.flush()
            stream.close()
            name = f"{field}.rank{self.rank}.f32"
            (self.folder / (name + ".partial")).replace(self.folder / name)
            files[name] = {"bytes": (self.folder / name).stat().st_size, "sha256": self.hashes[field].hexdigest()}
        return {"format": "little_endian_float32", "files": files, "pieces": self.records}


def index_pieces(pieces, expected_elements):
    by_name = {}
    for piece in pieces:
        require(type(piece["start"]) is int and type(piece["end"]) is int, "Invalid parameter interval")
        require(0 <= piece["start"] < piece["end"], "Empty or negative parameter interval")
        require(piece["end"] - piece["start"] == piece["elements"], "Parameter interval size mismatch")
        require(all(type(n) is int and n > 0 for n in piece["shape"]), "Invalid parameter shape")
        require(type(piece["offset_bytes"]) is int and piece["offset_bytes"] >= 0 and piece["offset_bytes"] % 4 == 0,
                "Invalid byte offset")
        by_name.setdefault(piece["name"], []).append(piece)
    total = 0
    for name, rows in by_name.items():
        rows.sort(key=lambda p: p["start"])
        cursor = 0
        for row in rows:
            require(row["start"] == cursor, f"Missing/overlapping/duplicated parameter ownership: {name} at {cursor}")
            require(row["shape"] == rows[0]["shape"], f"Shape mismatch: {name}")
            cursor = row["end"]
        require(cursor == math.prod(rows[0]["shape"]), f"Incomplete parameter coverage: {name}")
        total += cursor
    require(total == expected_elements, f"Coverage {total} != expected {expected_elements}")
    return by_name


def verify_storage(folder, storage):
    require(storage["format"] == "little_endian_float32", "Unsupported raw format")
    require(storage["files"], "No raw tensor files")
    for name, expected in storage["files"].items():
        path = validate_artifact_path(folder, name)
        require(path.is_file() and path.stat().st_size == expected["bytes"], f"Raw tensor size mismatch: {path}")
        require(digest(path) == expected["sha256"], f"Raw tensor checksum mismatch: {path}")
        rows = sorted((p for p in storage["pieces"] if p["file"] == name), key=lambda p: p["offset_bytes"])
        offset = 0
        for piece in rows:
            require(piece["offset_bytes"] == offset, f"Gap or duplicate storage interval: {name}")
            offset += piece["elements"] * 4
        require(offset == expected["bytes"], f"Unindexed raw bytes in {name}")
    require(all(p["file"] in storage["files"] for p in storage["pieces"]), "Missing indexed file")


def aligned_chunks(reference_root, reference, candidate_root, candidate):
    # Bounded handles avoid repeated metadata lookups for TP-strided pieces.
    # The caller still verifies every raw file hash and interval before reading.
    handles = OrderedDict()

    def read(root, piece, cursor, count):
        key = (str(root), piece['file'])
        handle = handles.pop(key, None)
        if handle is None:
            handle = validate_artifact_path(root, piece['file']).open('rb')
        handles[key] = handle
        while len(handles) > 32:
            handles.popitem(last=False)[1].close()
        handle.seek(piece['offset_bytes'] + (cursor - piece['start']) * 4)
        data = handle.read(count * 4)
        require(len(data) == count * 4, 'Truncated raw tensor')
        return np.frombuffer(data, dtype='<f4')

    try:
        require(reference.keys() == candidate.keys(), 'Parameter name sets differ')
        for name, left in reference.items():
            right = candidate[name]
            require(left[0]['shape'] == right[0]['shape'], 'Parameter shape changed')
            li = ri = cursor = 0
            while li < len(left) and ri < len(right):
                a, b = left[li], right[ri]
                stop = min(a['end'], b['end'], cursor + CHUNK_ELEMENTS)
                n = stop - cursor
                require(n > 0 and a['start'] <= cursor and b['start'] <= cursor, 'Invalid shard intersection')
                yield name, read(candidate_root, b, cursor, n), read(reference_root, a, cursor, n)
                cursor = stop
                li += cursor == a['end']
                ri += cursor == b['end']
            require(li == len(left) and ri == len(right), 'Incomplete parameter intersection')
    finally:
        for handle in handles.values():
            handle.close()

