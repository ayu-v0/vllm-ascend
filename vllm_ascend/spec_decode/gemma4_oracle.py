# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


PLACEHOLDER_TOKEN_ID = -1

ORACLE_TENSOR_KEYS = {
    "oracle_target_argmax",
    "oracle_target_top2_ids",
    "oracle_target_top2_values",
    "oracle_draft_token_ids",
    "oracle_bonus_token_ids",
    "oracle_sampled_token_ids",
}

_GROUP_KEY = re.compile(
    r"^group_(?P<group_id>\d+)_(?P<kind>block_table|slot_mapping)$"
)


def select_full_vocab_processed_logits(processed_logits: Any) -> Any | None:
    if isinstance(processed_logits, tuple):
        logits, indices = processed_logits
        return logits if indices is None else None
    return processed_logits


def _to_python(value: Any, name: str) -> Any:
    device = getattr(value, "device", None)
    if device is not None and getattr(device, "type", None) != "cpu":
        raise AssertionError(f"Gemma4 oracle {name} must be copied to CPU")
    tolist = getattr(value, "tolist", None)
    return tolist() if callable(tolist) else value


def _flatten(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        flattened: list[Any] = []
        for item in value:
            flattened.extend(_flatten(item))
        return flattened
    return [value]


def _flat_ints(value: Any, name: str) -> list[int]:
    return [int(item) for item in _flatten(_to_python(value, name))]


def _vector_ints(value: Any, name: str) -> list[int]:
    data = _to_python(value, name)
    if not isinstance(data, (list, tuple)) or any(
        isinstance(item, (list, tuple)) for item in data
    ):
        raise AssertionError(f"Gemma4 oracle {name} must be one-dimensional")
    return [int(item) for item in data]


def _nested_ints(value: Any, name: str) -> list[list[int]]:
    data = _to_python(value, name)
    if not isinstance(data, (list, tuple)):
        raise AssertionError(f"Gemma4 oracle {name} must be two-dimensional")
    rows: list[list[int]] = []
    for row in data:
        if not isinstance(row, (list, tuple)):
            raise AssertionError(f"Gemma4 oracle {name} must be two-dimensional")
        rows.append([int(item) for item in row])
    return rows


def _nested_floats(value: Any, name: str) -> list[list[float]]:
    data = _to_python(value, name)
    if not isinstance(data, (list, tuple)):
        raise AssertionError(f"Gemma4 oracle {name} must be two-dimensional")
    rows: list[list[float]] = []
    for row in data:
        if not isinstance(row, (list, tuple)):
            raise AssertionError(f"Gemma4 oracle {name} must be two-dimensional")
        rows.append([float(item) for item in row])
    return rows


def _value_shape(value: Any, name: str) -> tuple[int, ...]:
    data = _to_python(value, name)
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(dimension) for dimension in shape)

    def _infer(item: Any) -> tuple[int, ...]:
        if not isinstance(item, (list, tuple)):
            return ()
        if not item:
            return (0,)
        child_shapes = [_infer(child) for child in item]
        if any(child_shape != child_shapes[0] for child_shape in child_shapes[1:]):
            raise AssertionError(f"Gemma4 oracle {name} has a ragged shape")
        return (len(item), *child_shapes[0])

    return _infer(data)


def _expected_shape(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(dimension, int) and dimension >= 0 for dimension in value
    ):
        raise AssertionError(f"Gemma4 oracle {name} shape metadata is invalid")
    return tuple(value)


def reconstruct_greedy_output(
    *,
    target_argmax: Any,
    draft_token_ids: Any,
    bonus_token_ids: Any,
    num_draft_tokens: Sequence[int],
    max_spec_len: int,
) -> list[list[int]]:
    if max_spec_len < 0:
        raise AssertionError("max_spec_len must be non-negative")
    target = _flat_ints(target_argmax, "target_argmax")
    draft = _flat_ints(draft_token_ids, "draft_token_ids")
    bonus = _flat_ints(bonus_token_ids, "bonus_token_ids")
    counts = [int(count) for count in num_draft_tokens]
    if any(count < 0 or count > max_spec_len for count in counts):
        raise AssertionError(
            f"invalid num_draft_tokens={counts} for max_spec_len={max_spec_len}"
        )
    expected_tokens = sum(counts)
    if len(target) != expected_tokens or len(draft) != expected_tokens:
        raise AssertionError(
            "flattened target/draft size does not match num_draft_tokens: "
            f"target={len(target)} draft={len(draft)} expected={expected_tokens}"
        )
    if len(bonus) != len(counts):
        raise AssertionError(
            f"bonus rows={len(bonus)} request rows={len(counts)}"
        )

    output = [
        [PLACEHOLDER_TOKEN_ID] * (max_spec_len + 1) for _ in counts
    ]
    offset = 0
    for row, count in enumerate(counts):
        accepted_all = True
        for position in range(count):
            target_token = target[offset + position]
            draft_token = draft[offset + position]
            if draft_token == target_token:
                output[row][position] = draft_token
                continue
            output[row][position] = target_token
            accepted_all = False
            break
        if accepted_all:
            output[row][count] = bonus[row]
        offset += count
    return output


def validate_greedy_oracle_snapshot(
    *,
    tensors: Mapping[str, Any],
    context: Mapping[str, Any],
    parsed_token_ids: Sequence[Sequence[int]],
    valid_sampled_token_count: Any | None,
) -> None:
    missing = sorted(ORACLE_TENSOR_KEYS - set(tensors))
    if missing:
        raise AssertionError(f"Gemma4 oracle tensors are missing: {missing}")

    target = _flat_ints(tensors["oracle_target_argmax"], "oracle_target_argmax")
    top2_ids = _nested_ints(
        tensors["oracle_target_top2_ids"], "oracle_target_top2_ids"
    )
    top2_values = _nested_floats(
        tensors["oracle_target_top2_values"], "oracle_target_top2_values"
    )
    draft = _flat_ints(
        tensors["oracle_draft_token_ids"], "oracle_draft_token_ids"
    )
    bonus = _flat_ints(
        tensors["oracle_bonus_token_ids"], "oracle_bonus_token_ids"
    )
    actual = _nested_ints(
        tensors["oracle_sampled_token_ids"], "oracle_sampled_token_ids"
    )

    counts = context.get("oracle_num_draft_tokens")
    req_ids = context.get("oracle_req_ids")
    max_spec_len = context.get("oracle_max_spec_len")
    vocab_size = context.get("oracle_vocab_size")
    if not isinstance(counts, list) or not all(
        isinstance(count, int) for count in counts
    ):
        raise AssertionError("oracle_num_draft_tokens must be a list[int]")
    if not isinstance(max_spec_len, int) or not isinstance(vocab_size, int):
        raise AssertionError("oracle max_spec_len/vocab_size metadata is invalid")
    if (
        not isinstance(req_ids, list)
        or len(req_ids) != len(counts)
        or not all(isinstance(req_id, str) for req_id in req_ids)
    ):
        raise AssertionError(
            f"oracle request rows do not match draft counts: "
            f"req_ids={req_ids} counts={counts}"
        )
    if len(top2_ids) != len(target) or len(top2_values) != len(target):
        raise AssertionError(
            "top2 row count does not match target tokens: "
            f"ids={len(top2_ids)} values={len(top2_values)} target={len(target)}"
        )
    if any(len(row) != 2 for row in top2_ids + top2_values):
        raise AssertionError("each top2 row must contain exactly two entries")

    expected = reconstruct_greedy_output(
        target_argmax=target,
        draft_token_ids=draft,
        bonus_token_ids=bonus,
        num_draft_tokens=counts,
        max_spec_len=max_spec_len,
    )
    if actual != expected:
        mismatch: tuple[int, int] | None = None
        for row, (actual_row, expected_row) in enumerate(
            zip(actual, expected, strict=False)
        ):
            for column, (actual_token, expected_token) in enumerate(
                zip(actual_row, expected_row, strict=False)
            ):
                if actual_token != expected_token:
                    mismatch = (row, column)
                    break
            if mismatch is not None:
                break
        req_ids = context.get("oracle_req_ids")
        mismatch_req_id = None
        if (
            mismatch is not None
            and isinstance(req_ids, list)
            and mismatch[0] < len(req_ids)
        ):
            mismatch_req_id = req_ids[mismatch[0]]
        margins = [row[0] - row[1] for row in top2_values]
        raise AssertionError(
            "Gemma4 greedy oracle mismatch: "
            f"trace_id={context.get('trace_id')} req_id={mismatch_req_id} "
            f"mismatch={mismatch} "
            f"num_draft_tokens={counts} expected={expected} actual={actual} "
            f"target_argmax={target} draft_token_ids={draft} "
            f"bonus_token_ids={bonus} "
            f"top2_ids={top2_ids} top2_values={top2_values} margins={margins}"
        )

    expected_parsed = [
        [token for token in row if 0 <= token < vocab_size] for row in expected
    ]
    parsed = [[int(token) for token in row] for row in parsed_token_ids]
    if parsed != expected_parsed:
        raise AssertionError(
            "Gemma4 parsed output does not match the target oracle: "
            f"trace_id={context.get('trace_id')} expected={expected_parsed} "
            f"actual={parsed}"
        )
    if valid_sampled_token_count is not None:
        actual_counts = _vector_ints(
            valid_sampled_token_count, "valid_sampled_token_count"
        )
        expected_counts = [len(row) for row in expected_parsed]
        if len(actual_counts) != len(expected_counts):
            raise AssertionError(
                "Gemma4 valid count rows do not match oracle output: "
                f"expected_rows={len(expected_counts)} "
                f"actual_rows={len(actual_counts)}"
            )
        if actual_counts != expected_counts:
            raise AssertionError(
                "Gemma4 valid count does not match oracle output: "
                f"expected={expected_counts} actual={actual_counts}"
            )


def validate_async_state_snapshot(
    *,
    tensors: Mapping[str, Any],
    context: Mapping[str, Any],
) -> None:
    required = {
        "prev_positions",
        "prev_num_draft_tokens",
        "prev_valid_sampled_token_count",
        "cpu_num_computed_tokens",
        "num_computed_before",
        "num_computed_after",
        "num_accepted_tokens",
    }
    missing = sorted(required - set(tensors))
    if missing:
        raise AssertionError(f"Gemma4 async state tensors are missing: {missing}")
    values = {name: _vector_ints(tensors[name], name) for name in required}

    req_ids = context.get("req_ids")
    previous = context.get("prev_req_id_to_index")
    applied = context.get("state_correction_applied")
    if not isinstance(req_ids, list) or not isinstance(previous, dict):
        raise AssertionError("Gemma4 async request mapping metadata is invalid")
    if not isinstance(applied, bool):
        raise AssertionError("state_correction_applied must be bool")
    if not all(isinstance(req_id, str) for req_id in req_ids) or len(
        set(req_ids)
    ) != len(req_ids):
        raise AssertionError("Gemma4 async request IDs must be unique strings")

    current_rows = len(req_ids)
    for name in (
        "prev_positions",
        "cpu_num_computed_tokens",
        "num_computed_after",
        "num_accepted_tokens",
    ):
        if len(values[name]) != current_rows:
            raise AssertionError(
                f"{name} rows={len(values[name])} requests={current_rows}"
            )
    previous_rows = len(values["num_computed_before"])
    if (
        len(values["prev_num_draft_tokens"]) != previous_rows
        or len(values["prev_valid_sampled_token_count"]) != previous_rows
    ):
        raise AssertionError("previous-state tensor sizes are inconsistent")
    previous_indices = list(previous.values())
    if (
        len(previous) != previous_rows
        or not all(
            isinstance(req_id, str) and isinstance(index, int)
            for req_id, index in previous.items()
        )
        or sorted(previous_indices) != list(range(previous_rows))
    ):
        raise AssertionError(
            "previous request mapping must contain each previous row exactly "
            f"once: mapping={previous} previous_rows={previous_rows}"
        )

    for row, req_id in enumerate(req_ids):
        expected_previous = int(previous.get(req_id, -1))
        actual_previous = values["prev_positions"][row]
        if actual_previous != expected_previous:
            raise AssertionError(
                f"request mapping mismatch row={row} req_id={req_id} "
                f"expected={expected_previous} actual={actual_previous}"
            )
        if applied and actual_previous >= 0:
            if actual_previous >= previous_rows:
                raise AssertionError(
                    f"previous row {actual_previous} is outside {previous_rows}"
                )
            participating = (
                values["prev_num_draft_tokens"][actual_previous] > 0
            )
        else:
            participating = False
        if participating:
            valid_count = values["prev_valid_sampled_token_count"][
                actual_previous
            ]
            expected_computed = (
                values["num_computed_before"][actual_previous] + valid_count
            )
            if values["num_accepted_tokens"][row] != valid_count:
                raise AssertionError(
                    f"accepted count mismatch row={row} expected={valid_count} "
                    f"actual={values['num_accepted_tokens'][row]}"
                )
        else:
            expected_computed = values["cpu_num_computed_tokens"][row]
            if actual_previous < 0 and values["num_accepted_tokens"][row] != 1:
                raise AssertionError(
                    f"new request row={row} must start with one accepted token"
                )
        if values["num_computed_after"][row] != expected_computed:
            raise AssertionError(
                f"num_computed mismatch row={row} expected={expected_computed} "
                f"actual={values['num_computed_after'][row]}"
            )

    expected_group_ids = context.get("oracle_kv_group_ids")
    expected_group_shapes = context.get("oracle_kv_group_shapes")
    if not isinstance(expected_group_ids, list) or not all(
        isinstance(group_id, int) and group_id >= 0
        for group_id in expected_group_ids
    ):
        raise AssertionError("Gemma4 oracle KV group IDs are invalid")
    if not expected_group_ids:
        raise AssertionError("Gemma4 oracle KV group set must not be empty")
    if len(set(expected_group_ids)) != len(expected_group_ids):
        raise AssertionError(
            f"duplicate KV group IDs in oracle context: {expected_group_ids}"
        )
    if not isinstance(expected_group_shapes, dict):
        raise AssertionError("Gemma4 oracle KV group shape metadata is invalid")

    groups: dict[int, set[str]] = {}
    for name in tensors:
        match = _GROUP_KEY.match(name)
        if match is None:
            continue
        group_id = int(match.group("group_id"))
        groups.setdefault(group_id, set()).add(match.group("kind"))
    incomplete = {
        group_id: kinds
        for group_id, kinds in groups.items()
        if kinds != {"block_table", "slot_mapping"}
    }
    if incomplete:
        raise AssertionError(f"KV group snapshot is incomplete: {incomplete}")
    expected_group_set = set(expected_group_ids)
    if set(groups) != expected_group_set:
        raise AssertionError(
            "KV group set does not match oracle context: "
            f"expected={sorted(expected_group_set)} actual={sorted(groups)}"
        )
    if set(expected_group_shapes) != expected_group_set:
        raise AssertionError(
            "KV group shape metadata does not match expected groups: "
            f"groups={expected_group_ids} shapes={expected_group_shapes}"
        )
    for group_id in expected_group_ids:
        shape_metadata = expected_group_shapes[group_id]
        if not isinstance(shape_metadata, dict) or set(shape_metadata) != {
            "block_table",
            "slot_mapping",
        }:
            raise AssertionError(
                f"KV group {group_id} shape metadata is incomplete: "
                f"{shape_metadata}"
            )
        for kind in ("block_table", "slot_mapping"):
            name = f"group_{group_id}_{kind}"
            expected_shape = _expected_shape(
                shape_metadata[kind], f"KV group {group_id} {kind}"
            )
            if kind == "block_table" and (
                len(expected_shape) != 2
                or expected_shape[0] != current_rows
            ):
                raise AssertionError(
                    f"KV group {group_id} block_table request rows do not "
                    f"match requests={current_rows}: shape={expected_shape}"
                )
            if kind == "slot_mapping" and len(expected_shape) != 1:
                raise AssertionError(
                    f"KV group {group_id} slot_mapping must be one-dimensional: "
                    f"shape={expected_shape}"
                )
            actual_shape = _value_shape(tensors[name], name)
            if actual_shape != expected_shape:
                raise AssertionError(
                    f"KV group {group_id} {kind} shape mismatch: "
                    f"expected={expected_shape} actual={actual_shape}"
                )
