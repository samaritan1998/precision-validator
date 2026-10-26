"""Canonical causal SFT batches; masks index target tokens before HF shifting."""
from precision_validator.common import require


def normalize_batch(record, vocab_size):
    require(isinstance(record, dict), "Input must be a JSON object")
    allowed = {"input_ids", "labels", "loss_mask", "attention_mask", "supervised_tokens",
               "sample_positions", "sample_vocab_ids", "provenance"}
    require(not (set(record) - allowed), "Unsupported input fields")
    raw = record.get("input_ids")
    require(isinstance(raw, list) and raw, "Provide nonempty input_ids")
    flat = not isinstance(raw[0], list)
    ids = [raw] if flat else raw
    require(all(isinstance(row, list) and len(row) == len(ids[0]) and len(row) >= 2 for row in ids),
            "Input must be rectangular with sequence length >= 2; pad explicitly")
    batch_size, length = len(ids), len(ids[0])
    require(all(type(i) is int and 0 <= i < vocab_size for row in ids for i in row),
            "Token id is outside the model vocabulary")

    def matrix(key, default, valid):
        value = record.get(key, default)
        if flat and isinstance(value, list) and value and not isinstance(value[0], list):
            value = [value]
        require(isinstance(value, list) and len(value) == batch_size and
                all(isinstance(row, list) and len(row) == length for row in value), f"Invalid {key} shape")
        require(all(type(v) is int and valid(v) for row in value for v in row), f"Invalid {key} values")
        return [row[:] for row in value]

    attention = matrix("attention_mask", [[1] * length for _ in ids], lambda v: v in (0, 1))
    lengths = [sum(row) for row in attention]
    require(all(n >= 2 and row == [1] * n + [0] * (length - n) for row, n in zip(attention, lengths)),
            "Only nonempty right-padded causal sequences are supported; no left padding or mask holes")
    labels = matrix("labels", ids, lambda v: v == -100 or 0 <= v < vocab_size)
    loss_mask = matrix("loss_mask", attention, lambda v: v in (0, 1))
    for row, active, keep in zip(labels, attention, loss_mask):
        for j in range(length):
            require(active[j] or not keep[j], "loss_mask includes padding")
            require("labels" not in record or active[j] or row[j] == -100,
                    "Explicit labels must ignore padded positions with -100")
            if not active[j] or not keep[j] or j == 0:
                row[j] = -100
        require(any(v != -100 for v in row[1:]), "Every sample needs at least one supervised next-token target")
    targets = sum(v != -100 for row in labels for v in row[1:])
    supplied = record.get("supervised_tokens", targets)
    require(type(supplied) is int and supplied == targets, "supervised_tokens differs from shifted labels/masks")

    def indices(key, default, bound):
        values = record.get(key, default)
        require(isinstance(values, list) and values and
                all(type(i) is int and 0 <= i < bound for i in values) and
                len(set(values)) == len(values), f"Invalid {key}")
        return values

    sample_length = min(lengths)
    return {"input_ids": ids, "labels": labels, "attention_mask": attention,
            "supervised_tokens": targets,
            "sample_positions": indices("sample_positions", sorted({0, sample_length // 2, sample_length - 1}), sample_length),
            "sample_vocab_ids": indices("sample_vocab_ids", sorted({0, vocab_size // 2, vocab_size - 1}), vocab_size),
            "provenance": record.get("provenance", {})}
