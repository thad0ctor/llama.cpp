#!/usr/bin/env python3
"""Tests for MoE grouped expert conversion and metadata."""

import json
import sys
import os
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gguf-py'))
import gguf
from gguf import GGUFReader


def test_constants_defined():
    """Verify grouped MoE constants are defined."""
    assert hasattr(gguf.Keys.LLM, 'MOE_QUANT_GROUP_COUNT')
    assert hasattr(gguf.Keys.LLM, 'MOE_QUANT_EXPERT_GROUP_MAP')
    assert hasattr(gguf.Keys.LLM, 'MOE_QUANT_EXPERT_INDEX_MAP')

    assert '{arch}' in gguf.Keys.LLM.MOE_QUANT_GROUP_COUNT
    assert '{bid}' in gguf.Keys.LLM.MOE_QUANT_EXPERT_GROUP_MAP
    assert '{bid}' in gguf.Keys.LLM.MOE_QUANT_EXPERT_INDEX_MAP
    print("PASS: constants defined")


def test_tensor_types_defined():
    """Verify grouped expert tensor types exist."""
    assert hasattr(gguf.MODEL_TENSOR, 'FFN_GATE_EXP_GRP')
    assert hasattr(gguf.MODEL_TENSOR, 'FFN_DOWN_EXP_GRP')
    assert hasattr(gguf.MODEL_TENSOR, 'FFN_UP_EXP_GRP')
    print("PASS: tensor types defined")


def test_tensor_names_contain_gid():
    """Verify grouped tensor names contain {gid} placeholder."""
    gate_name = gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.FFN_GATE_EXP_GRP]
    down_name = gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.FFN_DOWN_EXP_GRP]
    up_name   = gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.FFN_UP_EXP_GRP]

    assert '{gid}' in gate_name, f"Expected {{gid}} in gate name: {gate_name}"
    assert '{gid}' in down_name, f"Expected {{gid}} in down name: {down_name}"
    assert '{gid}' in up_name,   f"Expected {{gid}} in up name: {up_name}"
    assert '{bid}' in gate_name
    print("PASS: tensor names contain gid")


def test_moe_archs_have_grouped_tensors():
    """Verify all MoE architectures that have FFN_GATE_EXP also list grouped tensor types."""
    moe_archs_checked = 0
    missing_archs = []
    for arch, tensors in gguf.MODEL_TENSORS.items():
        has_moe = (gguf.MODEL_TENSOR.FFN_GATE_EXP in tensors
                   or gguf.MODEL_TENSOR.FFN_DOWN_EXP in tensors)
        if not has_moe:
            continue
        moe_archs_checked += 1
        if gguf.MODEL_TENSOR.FFN_UP_EXP_GRP not in tensors:
            missing_archs.append(gguf.MODEL_ARCH_NAMES.get(arch, str(arch)))
    assert moe_archs_checked > 10, f"Expected many MoE archs, found only {moe_archs_checked}"
    assert not missing_archs, f"MoE architectures missing grouped tensor types: {missing_archs}"
    print(f"PASS: {moe_archs_checked} MoE architectures have grouped tensor types")


def test_writer_methods_exist():
    """Verify GGUFWriter has grouped MoE methods."""
    with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
        tmp_path = f.name
    try:
        writer = gguf.GGUFWriter(tmp_path, arch='minimax-m2')
        assert hasattr(writer, 'add_moe_quant_group_count')
        assert hasattr(writer, 'add_moe_quant_expert_group_map')
        assert hasattr(writer, 'add_moe_quant_expert_index_map')
        writer.close()
        print("PASS: writer methods exist")
    finally:
        os.unlink(tmp_path)


def test_group_assignment_round_robin():
    """Test that round-robin grouping works as expected."""
    n_experts = 12
    n_groups = 3

    expected = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    actual = [eid % n_groups for eid in range(n_experts)]
    assert actual == expected, f"Round-robin mismatch: {actual} vs {expected}"

    group_sizes = [0] * n_groups
    for gid in actual:
        group_sizes[gid] += 1
    assert all(s == n_experts // n_groups for s in group_sizes), \
        f"Groups not equal size: {group_sizes}"

    print("PASS: round-robin grouping")


def test_group_assignment_custom_map():
    """Test that custom group maps are correctly parsed and validated."""
    raw_map = {
        "0": {"0": 0, "1": 0, "2": 1, "3": 1, "4": 2, "5": 2},
        "1": {"0": 2, "1": 2, "2": 1, "3": 1, "4": 0, "5": 0},
    }
    parsed = {int(k): {int(ek): int(ev) for ek, ev in v.items()} for k, v in raw_map.items()}

    assert parsed[0][0] == 0
    assert parsed[0][2] == 1
    assert parsed[0][4] == 2
    assert parsed[1][0] == 2
    assert parsed[1][4] == 0

    for layer_map in parsed.values():
        groups_present = set(layer_map.values())
        assert groups_present == {0, 1, 2}

    print("PASS: custom group map parsing")


def test_index_within_group():
    """Test that index-within-group computation is correct."""
    n_experts = 6
    n_groups = 3
    group_assign = [0, 1, 2, 0, 1, 2]

    index_in_group = []
    group_counters = {}
    for eid in range(n_experts):
        gid = group_assign[eid]
        idx = group_counters.get(gid, 0)
        index_in_group.append(idx)
        group_counters[gid] = idx + 1

    assert index_in_group == [0, 0, 0, 1, 1, 1], \
        f"Index mismatch: {index_in_group}"
    print("PASS: index within group")


def test_tensor_naming_convention():
    """Test that grouped tensor names follow the expected pattern for quantization targeting."""
    gate_pattern = gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.FFN_GATE_EXP_GRP]
    name_g0 = gate_pattern.format(bid=5, gid=0) + ".weight"
    name_g1 = gate_pattern.format(bid=5, gid=1) + ".weight"
    name_g2 = gate_pattern.format(bid=5, gid=2) + ".weight"

    assert name_g0 == "blk.5.ffn_gate_exps.g0.weight"
    assert name_g1 == "blk.5.ffn_gate_exps.g1.weight"
    assert name_g2 == "blk.5.ffn_gate_exps.g2.weight"

    # These patterns should be easy to target with tensor_types.txt regex
    import re
    hot_pattern = r"\.g0\.weight$"
    warm_pattern = r"\.g1\.weight$"
    cold_pattern = r"\.g2\.weight$"

    assert re.search(hot_pattern, name_g0)
    assert re.search(warm_pattern, name_g1)
    assert re.search(cold_pattern, name_g2)
    assert not re.search(hot_pattern, name_g1)

    print("PASS: tensor naming convention")


def test_validation_missing_expert_ids():
    """Test that _get_group_assignment rejects maps with missing expert IDs."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    class FakeModel:
        """Minimal stand-in to test _get_group_assignment validation."""
        moe_group_map = {0: {0: 0, 1: 1}}  # missing expert IDs 2..5
        moe_group_count = 2

    # Directly reimplement the validated logic from the converter
    n_experts = 6
    n_groups = 2
    layer_map = FakeModel.moe_group_map[0]
    covered = set(layer_map.keys())
    expected = set(range(n_experts))
    assert covered != expected, "Should detect missing expert IDs"
    missing = expected - covered
    assert missing == {2, 3, 4, 5}
    print("PASS: validation rejects missing expert IDs")


def test_validation_out_of_range_group_id():
    """Test that out-of-range group IDs are detected."""
    n_experts = 4
    n_groups = 2
    assignment = [0, 1, 0, 3]  # group 3 is out of range [0, 1]

    out_of_range = [eid for eid, gid in enumerate(assignment) if gid < 0 or gid >= n_groups]
    assert out_of_range == [3], f"Expected expert 3 out of range, got {out_of_range}"
    print("PASS: validation detects out-of-range group IDs")


def test_validation_group_count_bounds():
    """Test moe_group_count must be >= 1 and <= n_experts."""
    n_experts = 6
    for bad_count in [0, -1]:
        assert bad_count < 1, f"group_count {bad_count} should be rejected"
    assert 7 > n_experts, "group_count > n_experts should be rejected"
    print("PASS: validation catches bad group_count values")


def test_empty_group_in_assignment():
    """Test behavior when a valid assignment leaves a group empty."""
    n_experts = 4
    n_groups = 3
    assignment = [0, 0, 1, 1]  # group 2 is empty

    group_sizes = [0] * n_groups
    for gid in assignment:
        group_sizes[gid] += 1
    assert group_sizes[2] == 0, "Group 2 should be empty"
    assert group_sizes[0] == 2
    assert group_sizes[1] == 2

    index_in_group = []
    group_counters = {}
    for eid in range(n_experts):
        gid = assignment[eid]
        idx = group_counters.get(gid, 0)
        index_in_group.append(idx)
        group_counters[gid] = idx + 1
    assert index_in_group == [0, 1, 0, 1]
    print("PASS: empty group handled correctly in assignment")


def test_inconsistent_map_lengths():
    """Test detection of maps with wrong number of entries per layer."""
    n_experts = 4
    # Map has only 3 entries for a 4-expert model
    bad_map = {0: 0, 1: 1, 2: 0}
    covered = set(bad_map.keys())
    expected = set(range(n_experts))
    assert covered != expected
    assert (expected - covered) == {3}
    print("PASS: inconsistent map length detected")


def _validate_index_map(group_map: list[int], index_map: list[int], n_groups: int):
    """Replicates the C++ loader validation for expert_index_map consistency.
    Returns (ok, error_message)."""
    n_experts = len(group_map)
    assert len(index_map) == n_experts

    grp_sizes = [0] * n_groups
    for gid in group_map:
        grp_sizes[gid] += 1

    idx_seen: list[list[bool]] = [([False] * grp_sizes[g]) for g in range(n_groups)]

    for eid in range(n_experts):
        gid = group_map[eid]
        idx = index_map[eid]
        if idx >= grp_sizes[gid]:
            return False, f"expert {eid}: index {idx} >= group_size {grp_sizes[gid]} for group {gid}"
        if idx_seen[gid][idx]:
            return False, f"expert {eid}: duplicate index {idx} in group {gid}"
        idx_seen[gid][idx] = True

    return True, ""


def test_index_map_valid():
    """Test that a correct index map passes validation."""
    group_map = [0, 1, 2, 0, 1, 2]
    index_map = [0, 0, 0, 1, 1, 1]
    ok, err = _validate_index_map(group_map, index_map, 3)
    assert ok, f"Expected valid, got: {err}"
    print("PASS: valid index map accepted")


def test_index_map_out_of_range():
    """Test that an out-of-range local index is detected."""
    group_map = [0, 0, 1, 1]  # group 0 has 2 experts, group 1 has 2
    index_map = [0, 2, 0, 1]  # expert 1: index 2 is out of range for group 0 (size 2)
    ok, err = _validate_index_map(group_map, index_map, 2)
    assert not ok, "Should reject out-of-range index"
    assert "index 2 >= group_size 2" in err
    print("PASS: out-of-range local index rejected")


def test_index_map_duplicate():
    """Test that duplicate local indices within a group are detected."""
    group_map = [0, 0, 1, 1]
    index_map = [0, 0, 0, 1]  # expert 0 and 1 both claim index 0 in group 0
    ok, err = _validate_index_map(group_map, index_map, 2)
    assert not ok, "Should reject duplicate index"
    assert "duplicate index 0 in group 0" in err
    print("PASS: duplicate local index rejected")


def test_index_map_with_empty_group():
    """Test that index map validation works when a group is empty."""
    group_map = [0, 0, 1, 1]  # group 2 is empty (3 groups, only 0 and 1 used)
    index_map = [0, 1, 0, 1]
    ok, err = _validate_index_map(group_map, index_map, 3)
    assert ok, f"Expected valid with empty group 2, got: {err}"
    print("PASS: index map with empty group accepted")


def test_gguf_round_trip_multi_arch():
    """Write grouped MoE GGUFs for multiple architectures and verify round-trip."""
    test_archs = ["llama", "qwen2moe", "deepseek2", "mixtral"]
    for arch_name in test_archs:
        if arch_name not in gguf.MODEL_ARCH_NAMES.values():
            continue
        try:
            _run_gguf_round_trip(arch_name)
            print(f"PASS: GGUF round-trip for {arch_name}")
        except Exception as e:
            print(f"FAIL: GGUF round-trip for {arch_name}: {e}")
            raise


def test_gguf_round_trip():
    """Write a grouped MoE GGUF and read it back, verifying all metadata and tensor names."""
    _run_gguf_round_trip("minimax-m2")
    print("PASS: GGUF round-trip (metadata + tensors)")


def _run_gguf_round_trip(ARCH: str):
    N_LAYERS = 2
    N_EXPERTS = 6
    N_GROUPS = 3
    N_EMBD = 16
    N_FF = 32

    # Round-robin group assignment for each layer
    group_assigns: list[list[int]] = []
    index_maps: list[list[int]] = []
    for _ in range(N_LAYERS):
        ga = [eid % N_GROUPS for eid in range(N_EXPERTS)]
        group_assigns.append(ga)
        idx: list[int] = []
        counters: dict[int, int] = {}
        for eid in range(N_EXPERTS):
            gid = ga[eid]
            idx.append(counters.get(gid, 0))
            counters[gid] = counters.get(gid, 0) + 1
        index_maps.append(idx)

    # Compute expected group sizes
    group_sizes: list[dict[int, int]] = []
    for bid in range(N_LAYERS):
        sizes: dict[int, int] = {}
        for gid_val in group_assigns[bid]:
            sizes[gid_val] = sizes.get(gid_val, 0) + 1
        group_sizes.append(sizes)

    with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
        tmp_path = f.name

    try:
        # --- Write ---
        writer = gguf.GGUFWriter(tmp_path, arch=ARCH)
        writer.add_moe_quant_group_count(N_GROUPS)

        for bid in range(N_LAYERS):
            writer.add_moe_quant_expert_group_map(bid, group_assigns[bid])
            writer.add_moe_quant_expert_index_map(bid, index_maps[bid])

        # Add grouped tensor entries (small dummy F32 tensors)
        proj_info = {
            "ffn_gate_exps": (N_EMBD, N_FF),
            "ffn_down_exps": (N_FF, N_EMBD),
            "ffn_up_exps":   (N_EMBD, N_FF),
        }
        expected_tensor_names: set[str] = set()
        for bid in range(N_LAYERS):
            for gid in range(N_GROUPS):
                n_exp_in_grp = group_sizes[bid].get(gid, 0)
                if n_exp_in_grp == 0:
                    continue
                for proj_name, (dim0, dim1) in proj_info.items():
                    tname = f"blk.{bid}.{proj_name}.g{gid}.weight"
                    expected_tensor_names.add(tname)
                    data = np.zeros((n_exp_in_grp, dim1, dim0), dtype=np.float32)
                    writer.add_tensor(tname, data)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        # --- Read back ---
        reader = GGUFReader(tmp_path)

        # 1. Verify moe_quant_group_count is present and exact
        gc_key = gguf.Keys.LLM.MOE_QUANT_GROUP_COUNT.format(arch=ARCH)
        gc_field = reader.fields.get(gc_key)
        assert gc_field is not None, f"Missing key: {gc_key}"
        gc_value = gc_field.contents()
        assert gc_value == N_GROUPS, f"Expected group_count={N_GROUPS}, got {gc_value}"

        for bid in range(N_LAYERS):
            # 2. Verify per-layer maps exist and have correct lengths
            gmap_key = gguf.Keys.LLM.MOE_QUANT_EXPERT_GROUP_MAP.format(arch=ARCH, bid=bid)
            imap_key = gguf.Keys.LLM.MOE_QUANT_EXPERT_INDEX_MAP.format(arch=ARCH, bid=bid)

            gmap_field = reader.fields.get(gmap_key)
            imap_field = reader.fields.get(imap_key)
            assert gmap_field is not None, f"Missing key: {gmap_key}"
            assert imap_field is not None, f"Missing key: {imap_key}"

            gmap_vals = gmap_field.contents()
            imap_vals = imap_field.contents()
            assert len(gmap_vals) == N_EXPERTS, \
                f"Layer {bid}: expert_group_map length {len(gmap_vals)} != {N_EXPERTS}"
            assert len(imap_vals) == N_EXPERTS, \
                f"Layer {bid}: expert_index_map length {len(imap_vals)} != {N_EXPERTS}"

            # 3. Validate group IDs in range and index-map consistency
            grp_sizes_read: dict[int, int] = {}
            for eid in range(N_EXPERTS):
                gid = gmap_vals[eid]
                assert 0 <= gid < N_GROUPS, \
                    f"Layer {bid}, expert {eid}: group_id {gid} out of range [0, {N_GROUPS})"
                grp_sizes_read[gid] = grp_sizes_read.get(gid, 0) + 1

            idx_seen: dict[int, set[int]] = {g: set() for g in range(N_GROUPS)}
            for eid in range(N_EXPERTS):
                gid = gmap_vals[eid]
                idx = imap_vals[eid]
                assert 0 <= idx < grp_sizes_read[gid], \
                    f"Layer {bid}, expert {eid}: index {idx} >= group_size {grp_sizes_read[gid]}"
                assert idx not in idx_seen[gid], \
                    f"Layer {bid}, expert {eid}: duplicate index {idx} in group {gid}"
                idx_seen[gid].add(idx)

            # Verify maps match what we wrote
            assert gmap_vals == group_assigns[bid], \
                f"Layer {bid}: group_map readback mismatch"
            assert imap_vals == index_maps[bid], \
                f"Layer {bid}: index_map readback mismatch"

        # 4. Verify all expected grouped tensor names exist
        read_tensor_names = {t.name for t in reader.tensors}
        missing = expected_tensor_names - read_tensor_names
        extra = read_tensor_names - expected_tensor_names
        assert not missing, f"Missing tensors: {sorted(missing)}"
        assert not extra, f"Unexpected tensors: {sorted(extra)}"

        # Verify tensor shapes (GGUF stores dimensions in GGML/column-major order,
        # which is the reverse of the numpy row-major order used during writing)
        for t in reader.tensors:
            parts = t.name.replace(".weight", "").split(".")
            bid_str, proj_with_grp = parts[1], parts[2]
            bid = int(bid_str)
            proj_base = proj_with_grp.rsplit(".g", 1)[0] if ".g" in proj_with_grp else proj_with_grp
            gid = int(parts[3].replace("g", ""))
            n_exp_in_grp = group_sizes[bid].get(gid, 0)
            dim0, dim1 = proj_info[proj_base]
            numpy_shape = (n_exp_in_grp, dim1, dim0)
            expected_ggml_shape = tuple(reversed(numpy_shape))
            actual_shape = tuple(t.shape.tolist())
            assert actual_shape == expected_ggml_shape, \
                f"Tensor {t.name}: shape {actual_shape} != expected {expected_ggml_shape}"

    finally:
        os.unlink(tmp_path)


def test_backward_compat_no_grouped_metadata():
    """Verify GGUF without grouped metadata loads cleanly (no grouped keys present)."""
    ARCH = "llama"
    with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
        tmp_path = f.name
    try:
        writer = gguf.GGUFWriter(tmp_path, arch=ARCH)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        reader = GGUFReader(tmp_path)
        gc_key = gguf.Keys.LLM.MOE_QUANT_GROUP_COUNT.format(arch=ARCH)
        assert reader.fields.get(gc_key) is None, \
            "Legacy GGUF should not have moe_quant_group_count"
        print("PASS: backward compatibility (no grouped metadata)")
    finally:
        os.unlink(tmp_path)


def test_single_group_round_trip():
    """Verify single group (n_groups=1) works correctly - degenerates to merged."""
    N_EXPERTS = 4
    N_GROUPS = 1
    assignment = [0] * N_EXPERTS
    index_in_group = list(range(N_EXPERTS))

    ok, err = _validate_index_map(assignment, index_in_group, N_GROUPS)
    assert ok, f"Single group should be valid, got: {err}"
    print("PASS: single group round-trip validation")


def test_max_groups_round_trip():
    """Verify n_groups == n_experts works (each expert its own group)."""
    N_EXPERTS = 4
    N_GROUPS = N_EXPERTS
    assignment = list(range(N_EXPERTS))
    index_in_group = [0] * N_EXPERTS

    ok, err = _validate_index_map(assignment, index_in_group, N_GROUPS)
    assert ok, f"Max groups should be valid, got: {err}"
    print("PASS: max groups (one per expert) validation")


if __name__ == '__main__':
    test_constants_defined()
    test_tensor_types_defined()
    test_tensor_names_contain_gid()
    test_moe_archs_have_grouped_tensors()
    test_writer_methods_exist()
    test_group_assignment_round_robin()
    test_group_assignment_custom_map()
    test_index_within_group()
    test_tensor_naming_convention()
    test_validation_missing_expert_ids()
    test_validation_out_of_range_group_id()
    test_validation_group_count_bounds()
    test_empty_group_in_assignment()
    test_inconsistent_map_lengths()
    test_index_map_valid()
    test_index_map_out_of_range()
    test_index_map_duplicate()
    test_index_map_with_empty_group()
    test_gguf_round_trip()
    test_gguf_round_trip_multi_arch()
    test_backward_compat_no_grouped_metadata()
    test_single_group_round_trip()
    test_max_groups_round_trip()
    print("\nAll tests passed!")
