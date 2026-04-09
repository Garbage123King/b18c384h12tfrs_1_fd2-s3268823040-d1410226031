"""
Pure PyTorch implementation of the KataGo-style Go AI model extracted from ONNX.

Model: b18c384h12tfrs_1 (18 blocks, 384 channels, 12 heads)
Architecture: Transformer with Rotary Position Encoding (RoPE)

Inputs:
  - input_spatial: [B, 22, 19, 19]
  - input_global:  [B, 19]

Outputs:
  - out_policy:        [B, 6, 362]
  - out_value:         [B, 3]
  - out_miscvalue:     [B, 10]
  - out_moremiscvalue: [B, 8]
  - out_ownership:     [B, 1, 19, 19]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnx
import onnx.numpy_helper
import onnxruntime
import math


class RMSNorm(nn.Module):
    """RMSNorm as used in the ONNX model (LayerNorm without centering)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # x: [B, N, C] where N=361, C=384
        norm = x * x
        mean = norm.mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mean + self.eps)
        x_normed = x / rms
        return self.weight * x_normed


class Swish(nn.Module):
    """SiLU / Swish activation: x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class BatchNormBias(nn.Module):
    """BatchNorm-like bias layer used in KataGo: (x - mean) / sqrt(var + eps) * gamma + beta
    But here the mean/var/gamma are stored as fixed parameters (from training stats)."""
    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.sub = nn.Parameter(torch.zeros(1, num_features, 1, 1))  # running_mean
        self.div = nn.Parameter(torch.ones(1, num_features, 1, 1))   # running_var
        self.mul = nn.Parameter(torch.ones(1, num_features, 1, 1))   # gamma (weight)
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.eps = eps

    def forward(self, x):
        x = (x - self.sub) / self.div
        x = x * self.mul + self.beta
        return x


class RotaryPositionEncoding(nn.Module):
    """Rotary Position Encoding (RoPE) for 2D spatial positions on a 19x19 Go board.
    
    Uses interleaved pairing: dims (0,1), (2,3), ..., (30,31) are rotated together.
    The cos/sin tables (361, 32) are applied element-wise to the full 32-dim head.
    Formula: x_rot = x * cos + interleave(-x_odd, x_even) * sin
    """
    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.half_dim = head_dim // 2  # 16
        # Precomputed cos/sin tables: [361, 32]
        self.cos_table = nn.Parameter(torch.zeros(361, 32), requires_grad=False)
        self.sin_table = nn.Parameter(torch.zeros(361, 32), requires_grad=False)

    def forward(self, x):
        """
        x: [B, N, num_heads, head_dim] where N=361, head_dim=32
        """
        # cos/sin: [361, 32] -> [1, 361, 1, 32]
        cos = self.cos_table.unsqueeze(0).unsqueeze(2)
        sin = self.sin_table.unsqueeze(0).unsqueeze(2)

        # Reshape x to interleave pairs: [B, N, H, D//2, 2]
        B, N, H, D = x.shape
        x_pairs = x.reshape(B, N, H, D // 2, 2)

        # Split into even and odd indices
        x_even = x_pairs[..., 0]  # [B, N, H, 16] - indices 0,2,4,...
        x_odd = x_pairs[..., 1]   # [B, N, H, 16] - indices 1,3,5,...

        # Build the rotated pair: [-x_odd, x_even] interleaved back
        # This is equivalent to the rotation matrix [[cos, -sin], [sin, cos]]
        # applied to each (x_even, x_odd) pair
        neg_x_odd = -x_odd
        x_rot_pairs = torch.stack([neg_x_odd, x_even], dim=-1)  # [B, N, H, 16, 2]
        x_rotated = x_rot_pairs.reshape(B, N, H, D)  # [B, N, H, 32]

        # Apply rotation: x * cos + rotated * sin
        output = x * cos + x_rotated * sin

        return output


class AttentionBlock(nn.Module):
    """Multi-head self-attention with RoPE."""
    def __init__(self, dim, num_heads, q_weight, k_weight, v_weight, out_weight,
                 norm1_weight, cos_table, sin_table):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 32
        self.dim = dim

        self.norm1 = RMSNorm(dim)
        self.norm1.weight = nn.Parameter(torch.from_numpy(norm1_weight))

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.q_proj.weight = nn.Parameter(torch.from_numpy(q_weight.T))

        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj.weight = nn.Parameter(torch.from_numpy(k_weight.T))

        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj.weight = nn.Parameter(torch.from_numpy(v_weight.T))

        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj.weight = nn.Parameter(torch.from_numpy(out_weight.T))

        self.rope = RotaryPositionEncoding(num_heads, self.head_dim)
        self.rope.cos_table = nn.Parameter(torch.from_numpy(cos_table.copy()), requires_grad=False)
        self.rope.sin_table = nn.Parameter(torch.from_numpy(sin_table.copy()), requires_grad=False)

    def forward(self, x):
        """
        x: [B, C, H, W] -> flatten to [B, N, C] for attention, then back
        """
        B, C, H, W = x.shape
        N = H * W  # 361

        # Reshape to [B, N, C]
        x_flat = x.reshape(B, C, N).transpose(1, 2)  # [B, 361, 384]

        # Norm + residual for attention
        residual = x_flat.float()
        x_normed = self.norm1(x_flat.float())

        # Q, K, V projections
        q = self.q_proj(x_normed)  # [B, N, 384]
        k = self.k_proj(x_normed)
        v = self.v_proj(x_normed)

        # Reshape to [B, N, num_heads, head_dim]
        q = q.reshape(B, N, self.num_heads, self.head_dim)
        k = k.reshape(B, N, self.num_heads, self.head_dim)
        v = v.reshape(B, N, self.num_heads, self.head_dim)

        # Apply RoPE to q and k
        q = self.rope(q)
        k = self.rope(k)

        # Rearrange for attention: [B, num_heads, N, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v)  # [B, num_heads, N, head_dim]

        # Rearrange back: [B, N, num_heads * head_dim]
        out = out.transpose(1, 2).reshape(B, N, self.dim)

        # Output projection
        out = self.out_proj(out)

        # Residual
        x_flat = (residual + out).to(x.dtype)

        return x_flat  # [B, N, C]


class FFNBlock(nn.Module):
    """Feed-forward network with SwiGLU activation."""
    def __init__(self, dim, ffn_dim, linear1_weight, gate_weight, linear2_weight, norm2_weight):
        super().__init__()
        self.norm2 = RMSNorm(dim)
        self.norm2.weight = nn.Parameter(torch.from_numpy(norm2_weight))

        self.ffn_linear1 = nn.Linear(dim, ffn_dim, bias=False)
        self.ffn_linear1.weight = nn.Parameter(torch.from_numpy(linear1_weight.T))

        self.ffn_gate = nn.Linear(dim, ffn_dim, bias=False)
        self.ffn_gate.weight = nn.Parameter(torch.from_numpy(gate_weight.T))

        self.ffn_linear2 = nn.Linear(ffn_dim, dim, bias=False)
        self.ffn_linear2.weight = nn.Parameter(torch.from_numpy(linear2_weight.T))

    def forward(self, x_flat):
        """
        x_flat: [B, N, C]
        """
        residual = x_flat.float()
        x_normed = self.norm2(x_flat.float())

        # SwiGLU: silu(linear1(x)) * gate(x)
        x1 = self.ffn_linear1(x_normed)
        x_gate = self.ffn_gate(x_normed)
        x_ffn = F.silu(x1) * x_gate

        x_ffn = self.ffn_linear2(x_ffn)

        x_flat = (residual + x_ffn).to(x_flat.dtype)
        return x_flat


class TransformerBlock(nn.Module):
    """Full transformer block: Attention + FFN."""
    def __init__(self, dim, num_heads, ffn_dim, q_weight, k_weight, v_weight, out_weight,
                 norm1_weight, norm2_weight, ffn1_weight, gate_weight, ffn2_weight,
                 cos_table, sin_table):
        super().__init__()
        self.attn = AttentionBlock(dim, num_heads, q_weight, k_weight, v_weight, out_weight,
                                    norm1_weight, cos_table, sin_table)
        self.ffn = FFNBlock(dim, ffn_dim, ffn1_weight, gate_weight, ffn2_weight, norm2_weight)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = self.attn(x)      # [B, N, C]
        x_flat = self.ffn(x_flat)   # [B, N, C]
        # Reshape back to [B, C, H, W]
        x = x_flat.transpose(1, 2).reshape(B, C, H, W)
        return x


class KataGoModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Model hyperparameters
        self.dim = 384
        self.num_heads = 12
        self.head_dim = 32
        self.ffn_dim = 1024  # 384 * 8 / 3 ≈ 1024
        self.num_blocks = 18
        self.board_size = 19

    @classmethod
    def from_onnx(cls, onnx_path):
        """Load weights from ONNX model."""
        model = cls()
        onnx_model = onnx.load(onnx_path)

        # Extract all initializers into a dict (with .copy() for writable tensors)
        weights = {}
        for init in onnx_model.graph.initializer:
            weights[init.name] = onnx.numpy_helper.to_array(init).copy()

        # Also extract constant nodes that hold positional encodings
        for node in onnx_model.graph.node:
            if node.op_type == 'Constant':
                for a in node.attribute:
                    if a.name == 'value':
                        try:
                            arr = onnx.numpy_helper.to_array(a.t)
                            if arr.shape == (361, 32):
                                # Store cos/sin tables by node name
                                weights[node.name] = arr.copy()
                        except:
                            pass

        # === Initial Conv: input_spatial -> [B, 384, 19, 19] ===
        model.conv_spatial = nn.Conv2d(22, 384, kernel_size=3, stride=1, padding=1, bias=False)
        model.conv_spatial.weight = nn.Parameter(torch.from_numpy(weights['model.conv_spatial.weight']))

        # === Global embedding: input_global [B, 19] -> [B, 384] -> [B, 384, 1, 1] ===
        model.linear_global = nn.Linear(19, 384, bias=False)
        model.linear_global.weight = nn.Parameter(torch.from_numpy(weights['onnx::MatMul_4766'].T))

        # === Transformer Blocks ===
        model.blocks = nn.ModuleList()
        for i in range(model.num_blocks):
            prefix = f'model.blocks.{i}'

            # Find the MatMul weights for this block
            # Each block uses 7 MatMul weights:
            # q_proj, k_proj, v_proj, out_proj (attention)
            # ffn_linear1, ffn_gate, ffn_linear2 (FFN)
            # We need to find them by tracing the ONNX graph
            block_weights = cls._extract_block_weights(onnx_model, weights, i)

            # Get cos/sin tables (shared across blocks from block 0)
            cos_table = cls._get_cos_table(onnx_model, i)
            sin_table = cls._get_sin_table(onnx_model, i)

            block = TransformerBlock(
                dim=model.dim,
                num_heads=model.num_heads,
                ffn_dim=model.ffn_dim,
                q_weight=block_weights['q'],
                k_weight=block_weights['k'],
                v_weight=block_weights['v'],
                out_weight=block_weights['out'],
                norm1_weight=block_weights['norm1'],
                norm2_weight=block_weights['norm2'],
                ffn1_weight=block_weights['ffn1'],
                gate_weight=block_weights['gate'],
                ffn2_weight=block_weights['ffn2'],
                cos_table=cos_table,
                sin_table=sin_table,
            )
            model.blocks.append(block)

        # === Trunk final norm + activation ===
        model.norm_trunkfinal = BatchNormBias(384)
        model.norm_trunkfinal.sub = nn.Parameter(torch.from_numpy(weights['onnx::Sub_5163']))
        model.norm_trunkfinal.div = nn.Parameter(torch.from_numpy(weights['onnx::Div_5164']))
        model.norm_trunkfinal.mul = nn.Parameter(torch.from_numpy(weights['onnx::Mul_5165']))
        model.norm_trunkfinal.beta = nn.Parameter(torch.from_numpy(weights['model.norm_trunkfinal.beta']))
        model.act_trunkfinal = Swish()

        # === Policy Head ===
        model.policy_conv1p = nn.Conv2d(384, 48, kernel_size=1, bias=False)
        model.policy_conv1p.weight = nn.Parameter(torch.from_numpy(weights['model.policy_head.conv1p.weight']))

        model.policy_conv1g = nn.Conv2d(384, 48, kernel_size=1, bias=False)
        model.policy_conv1g.weight = nn.Parameter(torch.from_numpy(weights['model.policy_head.conv1g.weight']))

        model.policy_biasg = nn.Parameter(torch.from_numpy(weights['model.policy_head.biasg.beta']))
        model.policy_actg = Swish()

        model.policy_linear_pass = nn.Linear(144, 48)
        model.policy_linear_pass.weight = nn.Parameter(torch.from_numpy(weights['model.policy_head.linear_pass.weight']))
        model.policy_linear_pass.bias = nn.Parameter(torch.from_numpy(weights['model.policy_head.linear_pass.bias']))

        model.policy_act_pass = Swish()

        model.policy_linear_pass2 = nn.Linear(48, 6, bias=False)
        model.policy_linear_pass2.weight = nn.Parameter(torch.from_numpy(weights['onnx::MatMul_5169'].T))

        model.policy_linear_g = nn.Linear(144, 48, bias=False)
        model.policy_linear_g.weight = nn.Parameter(torch.from_numpy(weights['onnx::MatMul_5170'].T))

        model.policy_bias2 = nn.Parameter(torch.from_numpy(weights['model.policy_head.bias2.beta']))
        model.policy_act2 = Swish()

        model.policy_conv2p = nn.Conv2d(48, 6, kernel_size=1, bias=False)
        model.policy_conv2p.weight = nn.Parameter(torch.from_numpy(weights['model.policy_head.conv2p.weight']))

        # === Value Head ===
        model.value_conv1 = nn.Conv2d(384, 96, kernel_size=1, bias=False)
        model.value_conv1.weight = nn.Parameter(torch.from_numpy(weights['model.value_head.conv1.weight']))

        model.value_bias1 = nn.Parameter(torch.from_numpy(weights['model.value_head.bias1.beta']))
        model.value_act1 = Swish()

        model.value_linear2 = nn.Linear(288, 128)
        model.value_linear2.weight = nn.Parameter(torch.from_numpy(weights['model.value_head.linear2.weight']))
        model.value_linear2.bias = nn.Parameter(torch.from_numpy(weights['model.value_head.linear2.bias']))

        model.value_act2 = Swish()

        model.value_linear_valuehead = nn.Linear(128, 3)
        model.value_linear_valuehead.weight = nn.Parameter(torch.from_numpy(weights['model.value_head.linear_valuehead.weight']))
        model.value_linear_valuehead.bias = nn.Parameter(torch.from_numpy(weights['model.value_head.linear_valuehead.bias']))

        model.value_linear_miscvaluehead = nn.Linear(128, 10)
        model.value_linear_miscvaluehead.weight = nn.Parameter(torch.from_numpy(weights['model.value_head.linear_miscvaluehead.weight']))
        model.value_linear_miscvaluehead.bias = nn.Parameter(torch.from_numpy(weights['model.value_head.linear_miscvaluehead.bias']))

        model.value_linear_moremiscvaluehead = nn.Linear(128, 8)
        model.value_linear_moremiscvaluehead.weight = nn.Parameter(torch.from_numpy(weights['model.value_head.linear_moremiscvaluehead.weight']))
        model.value_linear_moremiscvaluehead.bias = nn.Parameter(torch.from_numpy(weights['model.value_head.linear_moremiscvaluehead.bias']))

        model.value_conv_ownership = nn.Conv2d(96, 1, kernel_size=1, bias=False)
        model.value_conv_ownership.weight = nn.Parameter(torch.from_numpy(weights['model.value_head.conv_ownership.weight']))

        return model

    @staticmethod
    def _extract_block_weights(onnx_model, weights, block_idx):
        """Extract the attention and FFN weights for a given block by tracing ONNX nodes."""
        prefix = f'/model/blocks.{block_idx}/'

        # Build set of initializer names for quick lookup
        init_names = set()
        for init in onnx_model.graph.initializer:
            init_names.add(init.name)

        # Find MatMul nodes for this block (only those with initializer weights)
        # Use the full node name path to determine which layer it belongs to
        result = {}
        result['norm1'] = weights[f'model.blocks.{block_idx}.norm1.weight'].copy()
        result['norm2'] = weights[f'model.blocks.{block_idx}.norm2.weight'].copy()

        for node in onnx_model.graph.node:
            if node.op_type == 'MatMul' and node.name.startswith(prefix):
                weight_name = node.input[1]
                # Only include weights that are initializers (not intermediate tensors)
                if weight_name in init_names:
                    # Determine layer type from full node name path
                    # e.g., /model/blocks.0/q_proj/MatMul -> q_proj
                    node_path = node.name  # /model/blocks.0/q_proj/MatMul
                    # Remove the prefix and the trailing /MatMul
                    layer_name = node_path[len(prefix):]  # q_proj/MatMul
                    layer_name = layer_name.rsplit('/MatMul', 1)[0]  # q_proj

                    if layer_name == 'q_proj':
                        result['q'] = weights[weight_name]
                    elif layer_name == 'k_proj':
                        result['k'] = weights[weight_name]
                    elif layer_name == 'v_proj':
                        result['v'] = weights[weight_name]
                    elif layer_name == 'out_proj':
                        result['out'] = weights[weight_name]
                    elif layer_name == 'ffn_linear1':
                        result['ffn1'] = weights[weight_name]
                    elif layer_name == 'ffn_linear_gate':
                        result['gate'] = weights[weight_name]
                    elif layer_name == 'ffn_linear2':
                        result['ffn2'] = weights[weight_name]

        return result

    @staticmethod
    def _get_cos_table(onnx_model, block_idx):
        """Get the cos positional encoding table for a block."""
        # The cos table is in Constant_13 of block 0, shared via reshape for other blocks
        # Find the Constant node with shape (361, 32) that starts with positive values
        for node in onnx_model.graph.node:
            if node.op_type == 'Constant' and f'blocks.{block_idx}' in node.name:
                for a in node.attribute:
                    if a.name == 'value':
                        try:
                            arr = onnx.numpy_helper.to_array(a.t)
                            if arr.shape == (361, 32):
                                # First element should be ~1.0 for cos
                                if arr[0, 0] > 0.9:
                                    return arr
                        except:
                            pass
        # Fallback: use block 0's table
        for node in onnx_model.graph.node:
            if node.name == '/model/blocks.0/Constant_13':
                for a in node.attribute:
                    if a.name == 'value':
                        return onnx.numpy_helper.to_array(a.t)
        raise ValueError(f"Could not find cos table for block {block_idx}")

    @staticmethod
    def _get_sin_table(onnx_model, block_idx):
        """Get the sin positional encoding table for a block."""
        for node in onnx_model.graph.node:
            if node.op_type == 'Constant' and f'blocks.{block_idx}' in node.name:
                for a in node.attribute:
                    if a.name == 'value':
                        try:
                            arr = onnx.numpy_helper.to_array(a.t)
                            if arr.shape == (361, 32):
                                # First element should be ~0.0 for sin
                                if abs(arr[0, 0]) < 0.1:
                                    return arr
                        except:
                            pass
        for node in onnx_model.graph.node:
            if node.name == '/model/blocks.0/Constant_18':
                for a in node.attribute:
                    if a.name == 'value':
                        return onnx.numpy_helper.to_array(a.t)
        raise ValueError(f"Could not find sin table for block {block_idx}")

    def forward(self, input_spatial, input_global):
        """
        Args:
            input_spatial: [B, 22, 19, 19]
            input_global:  [B, 19]
        Returns:
            out_policy:        [B, 6, 362]
            out_value:         [B, 3]
            out_miscvalue:     [B, 10]
            out_moremiscvalue: [B, 8]
            out_ownership:     [B, 1, 19, 19]
        """
        B = input_spatial.shape[0]

        # === Initial embedding ===
        x = self.conv_spatial(input_spatial)  # [B, 384, 19, 19]

        # Global embedding
        g_emb = self.linear_global(input_global)  # [B, 384]
        g_emb = g_emb.unsqueeze(-1).unsqueeze(-1)  # [B, 384, 1, 1]
        x = x + g_emb  # broadcast add

        # === Transformer blocks ===
        for block in self.blocks:
            x = block(x)

        # === Trunk final norm + activation ===
        x = self.norm_trunkfinal(x)
        x = self.act_trunkfinal(x)

        # === Policy Head ===
        policy_p = self.policy_conv1p(x)       # [B, 48, 19, 19]
        policy_g = self.policy_conv1g(x)       # [B, 48, 19, 19]
        policy_g = policy_g + self.policy_biasg
        policy_g = self.policy_actg(policy_g)

        # GPool for policy
        policy_gpool_feat = self._policy_gpool(policy_g)  # [B, 144]

        # Linear pass path
        policy_pass = self.policy_linear_pass(policy_gpool_feat)  # [B, 48]
        policy_pass = self.policy_act_pass(policy_pass)
        policy_pass_logits = self.policy_linear_pass2(policy_pass)  # [B, 6]

        # Linear g path (global feature bias)
        policy_g_bias = self.policy_linear_g(policy_gpool_feat)  # [B, 48]

        # Combine conv1p with global bias
        policy_g_bias = policy_g_bias.unsqueeze(-1).unsqueeze(-1)  # [B, 48, 1, 1]
        policy_combined = policy_p + policy_g_bias  # [B, 48, 19, 19]
        policy_combined = policy_combined + self.policy_bias2
        policy_combined = self.policy_act2(policy_combined)

        policy_spatial = self.policy_conv2p(policy_combined)  # [B, 6, 19, 19]

        # Reshape spatial policy and concat with pass
        policy_spatial = policy_spatial.reshape(B, 6, 361)  # [B, 6, 361]
        policy_pass_logits = policy_pass_logits.unsqueeze(-1)  # [B, 6, 1]
        out_policy = torch.cat([policy_spatial, policy_pass_logits], dim=2)  # [B, 6, 362]

        # === Value Head ===
        value_x = self.value_conv1(x)          # [B, 96, 19, 19]
        value_x = value_x + self.value_bias1
        value_act = self.value_act1(value_x)

        # Ownership from value head (before gpool, using the activated features)
        out_ownership = self.value_conv_ownership(value_act)  # [B, 1, 19, 19]

        # GPool for value
        value_gpool_feat = self._value_gpool(value_act)  # [B, 288]

        value_h = self.value_linear2(value_gpool_feat)  # [B, 128]
        value_h = self.value_act2(value_h)

        out_value = self.value_linear_valuehead(value_h)             # [B, 3]
        out_miscvalue = self.value_linear_miscvaluehead(value_h)     # [B, 10]
        out_moremiscvalue = self.value_linear_moremiscvaluehead(value_h)  # [B, 8]

        return out_policy, out_value, out_miscvalue, out_moremiscvalue, out_ownership

    def _policy_gpool(self, x):
        """Policy head global pooling: mean, mean*scale, max."""
        B, C, H, W = x.shape
        x_float = x.float()

        # scale_val = sqrt(spatial_size) - 14
        scale_val = math.sqrt(H * W) - 14  # 19 - 14 = 5

        # Mean pooling
        mean_pool = x_float.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]

        # Scaled mean: mean * scale_val / 10
        scaled_mean = mean_pool * (scale_val / 10.0)

        # Max pooling: reshape to [B, C, H*W], max, reshape back
        max_pool = x_float.reshape(B, C, -1).max(dim=-1, keepdim=True)[0]  # [B, C, 1]
        max_pool = max_pool.unsqueeze(-1)  # [B, C, 1, 1]

        # Concat and squeeze
        pooled = torch.cat([mean_pool, scaled_mean, max_pool], dim=1)  # [B, 3*C, 1, 1]
        pooled = pooled.squeeze(-1).squeeze(-1)  # [B, 3*C]
        return pooled

    def _value_gpool(self, x):
        """Value head global pooling: mean, mean*scale, mean*scale^2_adjusted."""
        B, C, H, W = x.shape
        x_float = x.float()

        # scale_val = sqrt(spatial_size) - 14
        scale_val = math.sqrt(H * W) - 14  # 19 - 14 = 5

        # Mean pooling
        mean_pool = x_float.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]

        # Scaled mean: mean * scale_val / 10
        scaled_mean = mean_pool * (scale_val / 10.0)

        # Quadratic feature: mean * (scale_val^2 / 100 - 0.1)
        quad_feat = mean_pool * (scale_val * scale_val / 100.0 - 0.1)

        # Concat and squeeze
        pooled = torch.cat([mean_pool, scaled_mean, quad_feat], dim=1)  # [B, 3*C, 1, 1]
        pooled = pooled.squeeze(-1).squeeze(-1)  # [B, 3*C]
        return pooled


def verify_model(onnx_path, pytorch_model, atol=1e-4, rtol=1e-3):
    """Verify that the PyTorch model matches the ONNX model output."""
    print("=" * 60)
    print("Verifying PyTorch model against ONNX model")
    print("=" * 60)

    # Create ONNX runtime session
    ort_session = onnxruntime.InferenceSession(onnx_path)

    # Generate random inputs
    np.random.seed(42)
    batch_size = 2
    input_spatial = np.random.randn(batch_size, 22, 19, 19).astype(np.float32)
    input_global = np.random.randn(batch_size, 19).astype(np.float32)

    # Run ONNX model
    ort_inputs = {
        'input_spatial': input_spatial,
        'input_global': input_global,
    }
    ort_outputs = ort_session.run(None, ort_inputs)

    output_names = ['out_policy', 'out_value', 'out_miscvalue', 'out_moremiscvalue', 'out_ownership']

    # Run PyTorch model
    pytorch_model.eval()
    with torch.no_grad():
        pt_input_spatial = torch.from_numpy(input_spatial)
        pt_input_global = torch.from_numpy(input_global)
        pt_outputs = pytorch_model(pt_input_spatial, pt_input_global)

    # Compare outputs
    all_pass = True
    for i, (name, ort_out) in enumerate(zip(output_names, ort_outputs)):
        pt_out = pt_outputs[i].numpy()
        max_diff = np.max(np.abs(ort_out - pt_out))
        mean_diff = np.mean(np.abs(ort_out - pt_out))

        # Check with both atol and rtol
        match = np.allclose(ort_out, pt_out, atol=atol, rtol=rtol)

        status = "PASS" if match else "FAIL"
        if not match:
            all_pass = False

        print(f"\n{name}:")
        print(f"  ONNX shape:   {ort_out.shape}")
        print(f"  PyTorch shape: {pt_out.shape}")
        print(f"  Max abs diff:  {max_diff:.6e}")
        print(f"  Mean abs diff: {mean_diff:.6e}")
        print(f"  Status: {status}")

        if not match:
            # Show first few values for debugging
            print(f"  ONNX first 5:   {ort_out.flatten()[:5]}")
            print(f"  PyTorch first 5: {pt_out.flatten()[:5]}")

    print("\n" + "=" * 60)
    if all_pass:
        print("ALL OUTPUTS MATCH! PyTorch model is consistent with ONNX model.")
    else:
        print("SOME OUTPUTS DO NOT MATCH. Please check the implementation.")
    print("=" * 60)

    return all_pass


if __name__ == '__main__':
    onnx_path = r'c:\hack_ta\b18c384h12tfrs_1_fd2-s3268823040-d1410226031.onnx'

    print("Loading ONNX model and building PyTorch model...")
    model = KataGoModel.from_onnx(onnx_path)
    model.eval()

    print(f"Model loaded successfully with {sum(p.numel() for p in model.parameters())} parameters")

    verify_model(onnx_path, model)
