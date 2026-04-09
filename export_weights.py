"""
Export all weights and test data from the ONNX model to binary files
for the C implementation to load.
"""
import torch
import numpy as np
import onnx
import onnx.numpy_helper
import onnxruntime
import os

from model_pytorch import KataGoModel

OUTPUT_DIR = r'c:\hack_ta\weights'
TEST_DIR = r'c:\hack_ta\test_data'


def save_tensor(tensor, filepath):
    """Save a numpy array or torch tensor as raw float32 binary."""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    arr = np.ascontiguousarray(tensor, dtype=np.float32)
    arr.tofile(filepath)


def export_weights(onnx_path):
    """Export all model weights to binary files."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = KataGoModel.from_onnx(onnx_path)
    model.eval()

    # === Conv spatial: [22, 384, 3, 3] (PyTorch format, C code uses same) ===
    save_tensor(model.conv_spatial.weight, os.path.join(OUTPUT_DIR, 'conv_spatial_weight.bin'))

    # === Global linear: weight [384, 19] ===
    save_tensor(model.linear_global.weight, os.path.join(OUTPUT_DIR, 'linear_global_weight.bin'))

    # === Transformer blocks ===
    for i in range(18):
        block_dir = os.path.join(OUTPUT_DIR, f'block{i}')
        os.makedirs(block_dir, exist_ok=True)
        block = model.blocks[i]

        # Norm1 weight: [384]
        save_tensor(block.attn.norm1.weight, os.path.join(block_dir, 'norm1_weight.bin'))

        # Q/K/V/Out projections: PyTorch stores as [out, in]
        save_tensor(block.attn.q_proj.weight, os.path.join(block_dir, 'q_weight.bin'))
        save_tensor(block.attn.k_proj.weight, os.path.join(block_dir, 'k_weight.bin'))
        save_tensor(block.attn.v_proj.weight, os.path.join(block_dir, 'v_weight.bin'))
        save_tensor(block.attn.out_proj.weight, os.path.join(block_dir, 'out_weight.bin'))

        # RoPE tables: [361, 32]
        save_tensor(block.attn.rope.cos_table, os.path.join(block_dir, 'cos_table.bin'))
        save_tensor(block.attn.rope.sin_table, os.path.join(block_dir, 'sin_table.bin'))

        # Norm2 weight: [384]
        save_tensor(block.ffn.norm2.weight, os.path.join(block_dir, 'norm2_weight.bin'))

        # FFN: linear1, gate, linear2
        save_tensor(block.ffn.ffn_linear1.weight, os.path.join(block_dir, 'ffn1_weight.bin'))
        save_tensor(block.ffn.ffn_gate.weight, os.path.join(block_dir, 'gate_weight.bin'))
        save_tensor(block.ffn.ffn_linear2.weight, os.path.join(block_dir, 'ffn2_weight.bin'))

    # === Trunk final norm ===
    save_tensor(model.norm_trunkfinal.sub.flatten(), os.path.join(OUTPUT_DIR, 'trunk_sub.bin'))
    save_tensor(model.norm_trunkfinal.div.flatten(), os.path.join(OUTPUT_DIR, 'trunk_div.bin'))
    save_tensor(model.norm_trunkfinal.mul.flatten(), os.path.join(OUTPUT_DIR, 'trunk_mul.bin'))
    save_tensor(model.norm_trunkfinal.beta.flatten(), os.path.join(OUTPUT_DIR, 'trunk_beta.bin'))

    # === Policy head ===
    save_tensor(model.policy_conv1p.weight, os.path.join(OUTPUT_DIR, 'policy_conv1p_weight.bin'))
    save_tensor(model.policy_conv1g.weight, os.path.join(OUTPUT_DIR, 'policy_conv1g_weight.bin'))
    save_tensor(model.policy_biasg, os.path.join(OUTPUT_DIR, 'policy_biasg.bin'))
    save_tensor(model.policy_linear_pass.weight, os.path.join(OUTPUT_DIR, 'policy_linear_pass_weight.bin'))
    save_tensor(model.policy_linear_pass.bias, os.path.join(OUTPUT_DIR, 'policy_linear_pass_bias.bin'))
    save_tensor(model.policy_linear_pass2.weight, os.path.join(OUTPUT_DIR, 'policy_linear_pass2_weight.bin'))
    save_tensor(model.policy_linear_g.weight, os.path.join(OUTPUT_DIR, 'policy_linear_g_weight.bin'))
    save_tensor(model.policy_bias2, os.path.join(OUTPUT_DIR, 'policy_bias2.bin'))
    save_tensor(model.policy_conv2p.weight, os.path.join(OUTPUT_DIR, 'policy_conv2p_weight.bin'))

    # === Value head ===
    save_tensor(model.value_conv1.weight, os.path.join(OUTPUT_DIR, 'value_conv1_weight.bin'))
    save_tensor(model.value_bias1, os.path.join(OUTPUT_DIR, 'value_bias1.bin'))
    save_tensor(model.value_linear2.weight, os.path.join(OUTPUT_DIR, 'value_linear2_weight.bin'))
    save_tensor(model.value_linear2.bias, os.path.join(OUTPUT_DIR, 'value_linear2_bias.bin'))
    save_tensor(model.value_linear_valuehead.weight, os.path.join(OUTPUT_DIR, 'value_linear_valuehead_weight.bin'))
    save_tensor(model.value_linear_valuehead.bias, os.path.join(OUTPUT_DIR, 'value_linear_valuehead_bias.bin'))
    save_tensor(model.value_linear_miscvaluehead.weight, os.path.join(OUTPUT_DIR, 'value_linear_miscvaluehead_weight.bin'))
    save_tensor(model.value_linear_miscvaluehead.bias, os.path.join(OUTPUT_DIR, 'value_linear_miscvaluehead_bias.bin'))
    save_tensor(model.value_linear_moremiscvaluehead.weight, os.path.join(OUTPUT_DIR, 'value_linear_moremiscvaluehead_weight.bin'))
    save_tensor(model.value_linear_moremiscvaluehead.bias, os.path.join(OUTPUT_DIR, 'value_linear_moremiscvaluehead_bias.bin'))
    save_tensor(model.value_conv_ownership.weight, os.path.join(OUTPUT_DIR, 'value_conv_ownership_weight.bin'))

    print(f"All weights exported to {OUTPUT_DIR}")
    return model


def export_test_data(onnx_path, model):
    """Export test inputs and expected outputs for C verification."""
    os.makedirs(TEST_DIR, exist_ok=True)

    np.random.seed(42)
    batch_size = 1
    input_spatial = np.random.randn(batch_size, 22, 19, 19).astype(np.float32)
    input_global = np.random.randn(batch_size, 19).astype(np.float32)

    # Save inputs
    save_tensor(input_spatial, os.path.join(TEST_DIR, 'input_spatial.bin'))
    save_tensor(input_global, os.path.join(TEST_DIR, 'input_global.bin'))

    # Run ONNX for reference outputs
    ort_session = onnxruntime.InferenceSession(onnx_path)
    ort_inputs = {'input_spatial': input_spatial, 'input_global': input_global}
    ort_outputs = ort_session.run(None, ort_inputs)

    output_names = ['policy', 'value', 'miscvalue', 'moremiscvalue', 'ownership']
    for name, arr in zip(output_names, ort_outputs):
        save_tensor(arr, os.path.join(TEST_DIR, f'output_{name}.bin'))
        print(f"  {name}: shape={arr.shape}, first5={arr.flatten()[:5]}")

    # Also save PyTorch outputs for reference
    with torch.no_grad():
        pt_spatial = torch.from_numpy(input_spatial)
        pt_global = torch.from_numpy(input_global)
        pt_outputs = model(pt_spatial, pt_global)

    for name, pt_out in zip(output_names, pt_outputs):
        save_tensor(pt_out, os.path.join(TEST_DIR, f'pt_output_{name}.bin'))

    print(f"Test data exported to {TEST_DIR}")

    # Write a metadata file with shapes
    with open(os.path.join(TEST_DIR, 'metadata.txt'), 'w') as f:
        f.write(f"batch_size={batch_size}\n")
        f.write(f"input_spatial_shape={list(input_spatial.shape)}\n")
        f.write(f"input_global_shape={list(input_global.shape)}\n")
        for name, arr in zip(output_names, ort_outputs):
            f.write(f"output_{name}_shape={list(arr.shape)}\n")


if __name__ == '__main__':
    onnx_path = r'c:\hack_ta\b18c384h12tfrs_1_fd2-s3268823040-d1410226031.onnx'
    model = export_weights(onnx_path)
    export_test_data(onnx_path, model)
