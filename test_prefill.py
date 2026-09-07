import sys
sys.path.insert(0, '/frontend')

import math
import os
import yaml
from frontend.hardware_parser import CloudSystemConfig
from frontend.ops.atlang.prefill_attention import prefill_flash_attention_inference


def _load_cloud_config():
    """Load default cloud system config from test_cloud_system.yaml."""
    with open("configs/architecture/system/test_cloud_system.yaml", "r") as f:
        system_config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    return CloudSystemConfig.from_yaml(system_config_yaml)


def test_kv_divisible():
    """Test case: kv_head_num divisible by core_array_size (original behavior)."""
    print("Test 1: kv_head_num (8) divisible by core_array_size (4)")
    config = _load_cloud_config()

    attention_shape = (512, 8, 1, 64, 64)  # 8 KV heads, divisible by 4
    intermediate_result_dir = "kick_the_tires/test_prefill/kv_divisible"
    os.makedirs(intermediate_result_dir, exist_ok=True)
    kernel = prefill_flash_attention_inference(
        config,
        attention_shape,
        dtype='float16',
        min_tQ=128,
        min_tS=128,
        intermediate_result_dir=intermediate_result_dir,
    )
    assert kernel is not None
    print("  ✓ Kernel created successfully")


def test_kv_less_than_array():
    """Test case: kv_head_num < core_array_size (main new case)."""
    print("\nTest 2: kv_head_num (2) < core_array_size (4)")
    config = _load_cloud_config()

    attention_shape = (512, 2, 1, 64, 64)  # 2 KV heads on 4 columns
    intermediate_result_dir = "kick_the_tires/test_prefill/kv_less_than_array"
    os.makedirs(intermediate_result_dir, exist_ok=True)
    kernel = prefill_flash_attention_inference(
        config,
        attention_shape,
        dtype='float16',
        min_tQ=128,
        min_tS=128,
        intermediate_result_dir=intermediate_result_dir,
    )
    assert kernel is not None
    print("  ✓ Kernel created successfully (2 heads on 4 columns)")


def test_kv_single():
    """Test case: single KV head."""
    print("\nTest 3: kv_head_num (1) single head")
    config = _load_cloud_config()

    attention_shape = (512, 1, 8, 64, 64)  # 1 KV head, 8 Q heads
    intermediate_result_dir = "kick_the_tires/test_prefill/kv_single"
    os.makedirs(intermediate_result_dir, exist_ok=True)
    kernel = prefill_flash_attention_inference(
        config,
        attention_shape,
        dtype='float16',
        min_tQ=128,
        min_tS=128,
        intermediate_result_dir=intermediate_result_dir,
    )
    assert kernel is not None
    print("  ✓ Kernel created successfully (1 head on 4 columns)")


def test_ceil_div_logic():
    """Verify ceil_div logic."""
    print("\nTest 4: Verify ceil_div calculations")
    test_cases = [
        (8, 4, 2),   # 8 heads / 4 cols = 2
        (2, 4, 1),   # ceil(2/4) = 1
        (1, 4, 1),   # ceil(1/4) = 1
        (16, 4, 4),  # 16 / 4 = 4
    ]
    for kv_heads, core_size, expected_gkv in test_cases:
        gkv = math.ceil(kv_heads / core_size) if kv_heads >= core_size else 1
        print(f"  kv_heads={kv_heads}, core_size={core_size} → Gkv={gkv} (expected {expected_gkv})")
        assert gkv == expected_gkv, f"Mismatch: got {gkv}, expected {expected_gkv}"
    print("  ✓ All ceil_div calculations correct")


def test_gqa_case():
    """Test case with GQA (kv_group_num > 1)."""
    print("\nTest 5: GQA case (kv_group_num=4)")
    config = _load_cloud_config()

    attention_shape = (512, 8, 4, 64, 64)  # 8 KV heads, 4 Q heads per KV head (GQA)
    intermediate_result_dir = "kick_the_tires/test_prefill/kv_gqa"
    os.makedirs(intermediate_result_dir, exist_ok=True)
    kernel = prefill_flash_attention_inference(
        config,
        attention_shape,
        dtype='float16',
        min_tQ=128,
        min_tS=128,
        intermediate_result_dir=intermediate_result_dir,
    )
    assert kernel is not None
    print("  ✓ Kernel created successfully (GQA with group size 4)")


if __name__ == "__main__":
    try:
        test_ceil_div_logic()
        test_kv_divisible()
        test_kv_less_than_array()
        test_kv_single()
        test_gqa_case()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
