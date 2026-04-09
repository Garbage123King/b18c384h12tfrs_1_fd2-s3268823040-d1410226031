/*
 * KataGo b18c384h12tfrs_1 - Pure C implementation
 * 18 blocks, 384 channels, 12 heads, head_dim=32
 *
 * Model architecture:
 *   Input: spatial [22][19][19] + global [19]
 *   Output: policy [6][362], value [3], miscvalue [10], moremiscvalue [8], ownership [1][19][19]
 */

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========== Constants ========== */
#define BOARD_SIZE  19
#define N_POS       (BOARD_SIZE * BOARD_SIZE)  /* 361 */
#define IN_CHANNELS 22
#define DIM         384
#define NUM_HEADS   12
#define HEAD_DIM    32
#define FFN_DIM     1024
#define NUM_BLOCKS  18

/* ========== Utility ========== */

float* alloc_float(int count) {
    float* p = (float*)malloc(sizeof(float) * count);
    if (p == NULL) { fprintf(stderr, "malloc failed for %d floats\n", count); exit(1); }
    return p;
}

void load_bin_into(const char* path, float* dst, int expected_count) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    int read = (int)fread(dst, sizeof(float), expected_count, f);
    if (read != expected_count) {
        fprintf(stderr, "Short read from %s: expected %d, got %d\n", path, expected_count, read);
        exit(1);
    }
    fclose(f);
}

float* load_bin(const char* path, int expected_count) {
    float* data = alloc_float(expected_count);
    load_bin_into(path, data, expected_count);
    return data;
}

float sigmoid_f(float x) {
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

float swish_f(float x) {
    return x * sigmoid_f(x);
}

/* ========== Conv 3x3 ========== */
/* input[ic][19][19], kernel[oc][ic][3][3], output[oc][19][19] */

void conv3x3(const float* input, int ic, float* output, int oc, const float* kernel) {
    float* padded = alloc_float(ic * 21 * 21);
    for (int c = 0; c < ic; c++) {
        for (int j = 0; j < 21; j++) {
            for (int k = 0; k < 21; k++) {
                if (j == 0 || k == 0 || j == 20 || k == 20)
                    padded[c * 21 * 21 + j * 21 + k] = 0.0f;
                else
                    padded[c * 21 * 21 + j * 21 + k] = input[c * 19 * 19 + (j - 1) * 19 + (k - 1)];
            }
        }
    }

    for (int O = 0; O < oc; O++) {
        for (int y = 0; y < 19; y++) {
            for (int x = 0; x < 19; x++) {
                float s = 0.0f;
                for (int c = 0; c < ic; c++) {
                    for (int dy = 0; dy < 3; dy++) {
                        for (int dx = 0; dx < 3; dx++) {
                            float iv = padded[c * 21 * 21 + (y + dy) * 21 + (x + dx)];
                            float kv = kernel[((O * ic + c) * 3 + dy) * 3 + dx];
                            s += iv * kv;
                        }
                    }
                }
                output[O * 19 * 19 + y * 19 + x] = s;
            }
        }
    }
    free(padded);
}

/* ========== Conv 1x1 ========== */
/* input[ic][h][w], kernel[oc][ic], output[oc][h][w] */

void conv1x1_spatial(const float* input, int ic, float* output, int oc, int h, int w, const float* kernel) {
    for (int O = 0; O < oc; O++) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                float s = 0.0f;
                for (int c = 0; c < ic; c++) {
                    s += input[c * h * w + y * w + x] * kernel[O * ic + c];
                }
                output[O * h * w + y * w + x] = s;
            }
        }
    }
}

/* ========== Linear (matmul + optional bias) ========== */

void linear(const float* input, int in_dim, float* output, int out_dim, const float* weight, const float* bias) {
    for (int j = 0; j < out_dim; j++) {
        float s = (bias != NULL) ? bias[j] : 0.0f;
        for (int i = 0; i < in_dim; i++) {
            s += input[i] * weight[j * in_dim + i];
        }
        output[j] = s;
    }
}

/* ========== RMSNorm ========== */
/* input[N][C], weight[C], output[N][C] */

void rmsnorm(const float* input, float* output, int N, int C, const float* weight) {
    for (int n = 0; n < N; n++) {
        float sum_sq = 0.0f;
        for (int c = 0; c < C; c++) {
            float v = input[n * C + c];
            sum_sq += v * v;
        }
        float mean_sq = sum_sq / (float)C;
        float rms = sqrtf(mean_sq + 1e-6f);
        for (int c = 0; c < C; c++) {
            output[n * C + c] = input[n * C + c] / rms * weight[c];
        }
    }
}

/* ========== BatchNorm-like bias layer ========== */
/* input[C][19][19], output[C][19][19]: (x - sub) / div * mul + beta */

void batchnorm_bias(const float* input, float* output, int C,
                    const float* sub, const float* div, const float* mul, const float* beta) {
    for (int c = 0; c < C; c++) {
        for (int y = 0; y < 19; y++) {
            for (int x = 0; x < 19; x++) {
                float v = input[c * 19 * 19 + y * 19 + x];
                v = (v - sub[c]) / div[c];
                v = v * mul[c] + beta[c];
                output[c * 19 * 19 + y * 19 + x] = v;
            }
        }
    }
}

/* ========== Swish activation in-place ========== */

void swish_inplace(float* data, int count) {
    for (int i = 0; i < count; i++) {
        data[i] = swish_f(data[i]);
    }
}

/* ========== Add broadcast (per-channel) ========== */
/* input[C][19][19], adder[C], output[C][19][19] */

void add_broadcast_spatial(const float* input, float* output, int C, const float* adder) {
    for (int c = 0; c < C; c++) {
        for (int y = 0; y < 19; y++) {
            for (int x = 0; x < 19; x++) {
                output[c * 19 * 19 + y * 19 + x] = input[c * 19 * 19 + y * 19 + x] + adder[c];
            }
        }
    }
}

/* ========== RoPE (Rotary Position Encoding) ========== */
/* qk: [N][num_heads][head_dim] = [361][12][32], interleaved pairing */

void apply_rope(float* qk, int N, const float* cos_table, const float* sin_table) {
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < NUM_HEADS; h++) {
            for (int d = 0; d < HEAD_DIM; d += 2) {
                int idx = (n * NUM_HEADS + h) * HEAD_DIM + d;
                float x_even = qk[idx];
                float x_odd  = qk[idx + 1];
                float cos_e = cos_table[n * HEAD_DIM + d];
                float sin_e = sin_table[n * HEAD_DIM + d];
                float cos_o = cos_table[n * HEAD_DIM + d + 1];
                float sin_o = sin_table[n * HEAD_DIM + d + 1];
                qk[idx]     = x_even * cos_e + (-x_odd) * sin_e;
                qk[idx + 1] = x_odd  * cos_o + x_even   * sin_o;
            }
        }
    }
}

/* ========== Softmax ========== */

void softmax(float* x, int len) {
    float max_val = x[0];
    for (int i = 1; i < len; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < len; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < len; i++) {
        x[i] /= sum;
    }
}

/* ========== Full Transformer Block ========== */

void transformer_block(float* x_spatial,
                       const float* q_w, const float* k_w, const float* v_w, const float* out_w,
                       const float* norm1_w, const float* norm2_w,
                       const float* ffn1_w, const float* gate_w, const float* ffn2_w,
                       const float* cos_table, const float* sin_table) {

    float* x_flat = alloc_float(N_POS * DIM);
    float* residual = alloc_float(N_POS * DIM);
    float* x_normed = alloc_float(N_POS * DIM);
    float* q = alloc_float(N_POS * DIM);
    float* k = alloc_float(N_POS * DIM);
    float* v = alloc_float(N_POS * DIM);

    /* Reshape [C][19][19] -> [N][C] */
    for (int c = 0; c < DIM; c++) {
        for (int y = 0; y < 19; y++) {
            for (int x = 0; x < 19; x++) {
                int n = y * 19 + x;
                x_flat[n * DIM + c] = x_spatial[c * 19 * 19 + y * 19 + x];
            }
        }
    }

    /* ---- Attention ---- */
    memcpy(residual, x_flat, sizeof(float) * N_POS * DIM);
    rmsnorm(x_flat, x_normed, N_POS, DIM, norm1_w);

    for (int n = 0; n < N_POS; n++) {
        linear(x_normed + n * DIM, DIM, q + n * DIM, DIM, q_w, NULL);
        linear(x_normed + n * DIM, DIM, k + n * DIM, DIM, k_w, NULL);
        linear(x_normed + n * DIM, DIM, v + n * DIM, DIM, v_w, NULL);
    }

    apply_rope(q, N_POS, cos_table, sin_table);
    apply_rope(k, N_POS, cos_table, sin_table);

    float* attn = alloc_float(NUM_HEADS * N_POS * N_POS);
    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    for (int h = 0; h < NUM_HEADS; h++) {
        for (int qi = 0; qi < N_POS; qi++) {
            for (int ki = 0; ki < N_POS; ki++) {
                float dot = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++) {
                    dot += q[(qi * NUM_HEADS + h) * HEAD_DIM + d] * k[(ki * NUM_HEADS + h) * HEAD_DIM + d];
                }
                attn[(h * N_POS + qi) * N_POS + ki] = dot * scale;
            }
        }
        for (int qi = 0; qi < N_POS; qi++) {
            softmax(attn + (h * N_POS + qi) * N_POS, N_POS);
        }
    }

    float* attn_out = alloc_float(N_POS * DIM);
    for (int h = 0; h < NUM_HEADS; h++) {
        for (int qi = 0; qi < N_POS; qi++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                float s = 0.0f;
                for (int ki = 0; ki < N_POS; ki++) {
                    s += attn[(h * N_POS + qi) * N_POS + ki] * v[(ki * NUM_HEADS + h) * HEAD_DIM + d];
                }
                attn_out[(qi * NUM_HEADS + h) * HEAD_DIM + d] = s;
            }
        }
    }

    float* proj_out = alloc_float(N_POS * DIM);
    for (int n = 0; n < N_POS; n++) {
        linear(attn_out + n * DIM, DIM, proj_out + n * DIM, DIM, out_w, NULL);
    }
    for (int i = 0; i < N_POS * DIM; i++) {
        x_flat[i] = residual[i] + proj_out[i];
    }

    free(q); free(k); free(v); free(attn); free(attn_out); free(proj_out);

    /* ---- FFN ---- */
    memcpy(residual, x_flat, sizeof(float) * N_POS * DIM);
    rmsnorm(x_flat, x_normed, N_POS, DIM, norm2_w);

    float* ffn1_out = alloc_float(N_POS * FFN_DIM);
    float* gate_out = alloc_float(N_POS * FFN_DIM);
    float* ffn_inter = alloc_float(N_POS * FFN_DIM);
    float* ffn2_out = alloc_float(N_POS * DIM);

    for (int n = 0; n < N_POS; n++) {
        linear(x_normed + n * DIM, DIM, ffn1_out + n * FFN_DIM, FFN_DIM, ffn1_w, NULL);
        linear(x_normed + n * DIM, DIM, gate_out  + n * FFN_DIM, FFN_DIM, gate_w, NULL);
        for (int j = 0; j < FFN_DIM; j++) {
            ffn_inter[n * FFN_DIM + j] = swish_f(ffn1_out[n * FFN_DIM + j]) * gate_out[n * FFN_DIM + j];
        }
        linear(ffn_inter + n * FFN_DIM, FFN_DIM, ffn2_out + n * DIM, DIM, ffn2_w, NULL);
    }

    for (int i = 0; i < N_POS * DIM; i++) {
        x_flat[i] = residual[i] + ffn2_out[i];
    }

    free(ffn1_out); free(gate_out); free(ffn_inter); free(ffn2_out);
    free(residual); free(x_normed);

    /* Reshape [N][C] -> [C][19][19] */
    for (int c = 0; c < DIM; c++) {
        for (int y = 0; y < 19; y++) {
            for (int x = 0; x < 19; x++) {
                int n = y * 19 + x;
                x_spatial[c * 19 * 19 + y * 19 + x] = x_flat[n * DIM + c];
            }
        }
    }
    free(x_flat);
}

/* ========== GPool ========== */

void policy_gpool(const float* input, int C, float* output) {
    float div = (float)(BOARD_SIZE * BOARD_SIZE);
    float sqrtN = (float)BOARD_SIZE;
    for (int c = 0; c < C; c++) {
        float s = 0.0f;
        float max_val = -1e30f;
        for (int y = 0; y < 19; y++) {
            for (int x = 0; x < 19; x++) {
                float v = input[c * 19 * 19 + y * 19 + x];
                s += v;
                float mask = (y < BOARD_SIZE && x < BOARD_SIZE) ? 1.0f : 0.0f;
                float temp = v + (mask - 1.0f);
                if (temp > max_val) max_val = temp;
            }
        }
        float mean = s / div;
        output[c]         = mean;
        output[C + c]     = mean * (sqrtN - 14.0f) * 0.1f;
        output[2 * C + c] = max_val;
    }
}

void value_gpool(const float* input, int C, float* output) {
    float div = (float)(BOARD_SIZE * BOARD_SIZE);
    float sqrtN = (float)BOARD_SIZE;
    for (int c = 0; c < C; c++) {
        float s = 0.0f;
        for (int y = 0; y < 19; y++) {
            for (int x = 0; x < 19; x++) {
                s += input[c * 19 * 19 + y * 19 + x];
            }
        }
        float mean = s / div;
        float sc = sqrtN - 14.0f;
        output[c]         = mean;
        output[C + c]     = mean * sc * 0.1f;
        output[2 * C + c] = mean * (sc * sc / 100.0f - 0.1f);
    }
}

/* ========== Error comparison ========== */

float compare_tensors(const float* a, const float* b, int count, const char* name) {
    float max_diff = 0.0f;
    int max_idx = 0;
    int err_count = 0;
    for (int i = 0; i < count; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_idx = i;
        }
        if (diff > 0.005f) {
            if (err_count < 5)
                fprintf(stderr, "ERROR: %s[%d] got %f, expected %f, diff %f\n",
                        name, i, a[i], b[i], diff);
            err_count++;
        }
    }
    if (err_count > 0)
        fprintf(stderr, "  %s: %d errors (diff > 0.005)\n", name, err_count);
    printf("  %s: max_diff=%.6e at idx %d\n", name, max_diff, max_idx);
    return max_diff;
}


/* ========== Weight storage ========== */

typedef struct {
    float conv_spatial_weight[22 * 384 * 3 * 3];
    float linear_global_weight[384 * 19];
    struct {
        float norm1_weight[DIM];
        float q_weight[DIM * DIM];
        float k_weight[DIM * DIM];
        float v_weight[DIM * DIM];
        float out_weight[DIM * DIM];
        float cos_table[N_POS * HEAD_DIM];
        float sin_table[N_POS * HEAD_DIM];
        float norm2_weight[DIM];
        float ffn1_weight[DIM * FFN_DIM];
        float gate_weight[DIM * FFN_DIM];
        float ffn2_weight[FFN_DIM * DIM];
    } blocks[NUM_BLOCKS];
    float trunk_sub[DIM];
    float trunk_div[DIM];
    float trunk_mul[DIM];
    float trunk_beta[DIM];
    float policy_conv1p_weight[384 * 48];
    float policy_conv1g_weight[384 * 48];
    float policy_biasg[48];
    float policy_linear_pass_weight[144 * 48];
    float policy_linear_pass_bias[48];
    float policy_linear_pass2_weight[48 * 6];
    float policy_linear_g_weight[144 * 48];
    float policy_bias2[48];
    float policy_conv2p_weight[48 * 6];
    float value_conv1_weight[384 * 96];
    float value_bias1[96];
    float value_linear2_weight[288 * 128];
    float value_linear2_bias[128];
    float value_linear_valuehead_weight[128 * 3];
    float value_linear_valuehead_bias[3];
    float value_linear_miscvaluehead_weight[128 * 10];
    float value_linear_miscvaluehead_bias[10];
    float value_linear_moremiscvaluehead_weight[128 * 8];
    float value_linear_moremiscvaluehead_bias[8];
    float value_conv_ownership_weight[96];
} ModelWeights;


void load_weights(ModelWeights* W, const char* weights_dir) {
    char path[512];

    #define LOAD(field, count) do { \
        sprintf(path, "%s/" #field ".bin", weights_dir); \
        load_bin_into(path, W->field, count); \
    } while(0)

    LOAD(conv_spatial_weight, 22 * 384 * 3 * 3);
    LOAD(linear_global_weight, 384 * 19);

    for (int i = 0; i < NUM_BLOCKS; i++) {
        #define LOAD_B(field, count) do { \
            sprintf(path, "%s/block%d/" #field ".bin", weights_dir, i); \
            load_bin_into(path, W->blocks[i].field, count); \
        } while(0)

        LOAD_B(norm1_weight, DIM);
        LOAD_B(q_weight, DIM * DIM);
        LOAD_B(k_weight, DIM * DIM);
        LOAD_B(v_weight, DIM * DIM);
        LOAD_B(out_weight, DIM * DIM);
        LOAD_B(cos_table, N_POS * HEAD_DIM);
        LOAD_B(sin_table, N_POS * HEAD_DIM);
        LOAD_B(norm2_weight, DIM);
        LOAD_B(ffn1_weight, DIM * FFN_DIM);
        LOAD_B(gate_weight, DIM * FFN_DIM);
        LOAD_B(ffn2_weight, FFN_DIM * DIM);

        #undef LOAD_B
    }

    LOAD(trunk_sub, DIM);
    LOAD(trunk_div, DIM);
    LOAD(trunk_mul, DIM);
    LOAD(trunk_beta, DIM);

    LOAD(policy_conv1p_weight, 384 * 48);
    LOAD(policy_conv1g_weight, 384 * 48);
    LOAD(policy_biasg, 48);
    LOAD(policy_linear_pass_weight, 144 * 48);
    LOAD(policy_linear_pass_bias, 48);
    LOAD(policy_linear_pass2_weight, 48 * 6);
    LOAD(policy_linear_g_weight, 144 * 48);
    LOAD(policy_bias2, 48);
    LOAD(policy_conv2p_weight, 48 * 6);

    LOAD(value_conv1_weight, 384 * 96);
    LOAD(value_bias1, 96);
    LOAD(value_linear2_weight, 288 * 128);
    LOAD(value_linear2_bias, 128);
    LOAD(value_linear_valuehead_weight, 128 * 3);
    LOAD(value_linear_valuehead_bias, 3);
    LOAD(value_linear_miscvaluehead_weight, 128 * 10);
    LOAD(value_linear_miscvaluehead_bias, 10);
    LOAD(value_linear_moremiscvaluehead_weight, 128 * 8);
    LOAD(value_linear_moremiscvaluehead_bias, 8);
    LOAD(value_conv_ownership_weight, 96);

    #undef LOAD

    printf("All weights loaded from %s\n", weights_dir);
}


/* ========== Full Forward Pass ========== */

void forward(const ModelWeights* W,
             const float* input_spatial,
             const float* input_global,
             float* out_policy,
             float* out_value,
             float* out_miscvalue,
             float* out_moremiscvalue,
             float* out_ownership)
{
    /* === Initial embedding === */
    float* x = alloc_float(DIM * 19 * 19);
    conv3x3(input_spatial, IN_CHANNELS, x, DIM, W->conv_spatial_weight);

    float g_emb[DIM];
    linear(input_global, 19, g_emb, DIM, W->linear_global_weight, NULL);
    add_broadcast_spatial(x, x, DIM, g_emb);

    /* === Transformer blocks === */
    for (int i = 0; i < NUM_BLOCKS; i++) {
        printf("  Block %d...\n", i);
        transformer_block(x,
                          W->blocks[i].q_weight, W->blocks[i].k_weight,
                          W->blocks[i].v_weight, W->blocks[i].out_weight,
                          W->blocks[i].norm1_weight, W->blocks[i].norm2_weight,
                          W->blocks[i].ffn1_weight, W->blocks[i].gate_weight,
                          W->blocks[i].ffn2_weight,
                          W->blocks[i].cos_table, W->blocks[i].sin_table);
    }

    /* === Trunk final norm + activation === */
    float* trunk = alloc_float(DIM * 19 * 19);
    batchnorm_bias(x, trunk, DIM, W->trunk_sub, W->trunk_div, W->trunk_mul, W->trunk_beta);
    swish_inplace(trunk, DIM * 19 * 19);
    free(x);
    x = trunk;

    /* === Policy Head === */
    float* policy_p = alloc_float(48 * 19 * 19);
    conv1x1_spatial(x, 384, policy_p, 48, 19, 19, W->policy_conv1p_weight);

    float* policy_g = alloc_float(48 * 19 * 19);
    conv1x1_spatial(x, 384, policy_g, 48, 19, 19, W->policy_conv1g_weight);
    add_broadcast_spatial(policy_g, policy_g, 48, W->policy_biasg);
    swish_inplace(policy_g, 48 * 19 * 19);

    float policy_gpool_feat[144];
    policy_gpool(policy_g, 48, policy_gpool_feat);

    float policy_pass[48];
    linear(policy_gpool_feat, 144, policy_pass, 48, W->policy_linear_pass_weight, W->policy_linear_pass_bias);
    swish_inplace(policy_pass, 48);

    float policy_pass_logits[6];
    linear(policy_pass, 48, policy_pass_logits, 6, W->policy_linear_pass2_weight, NULL);

    float policy_g_bias[48];
    linear(policy_gpool_feat, 144, policy_g_bias, 48, W->policy_linear_g_weight, NULL);

    add_broadcast_spatial(policy_p, policy_p, 48, policy_g_bias);
    add_broadcast_spatial(policy_p, policy_p, 48, W->policy_bias2);
    swish_inplace(policy_p, 48 * 19 * 19);

    float* policy_spatial = alloc_float(6 * 19 * 19);
    conv1x1_spatial(policy_p, 48, policy_spatial, 6, 19, 19, W->policy_conv2p_weight);

    for (int c = 0; c < 6; c++) {
        for (int y = 0; y < 19; y++) {
            for (int xi = 0; xi < 19; xi++) {
                out_policy[c * 362 + y * 19 + xi] = policy_spatial[c * 19 * 19 + y * 19 + xi];
            }
        }
        out_policy[c * 362 + 361] = policy_pass_logits[c];
    }

    /* === Value Head === */
    float* value_x = alloc_float(96 * 19 * 19);
    conv1x1_spatial(x, 384, value_x, 96, 19, 19, W->value_conv1_weight);
    add_broadcast_spatial(value_x, value_x, 96, W->value_bias1);
    swish_inplace(value_x, 96 * 19 * 19);

    conv1x1_spatial(value_x, 96, out_ownership, 1, 19, 19, W->value_conv_ownership_weight);

    float value_gpool_feat[288];
    value_gpool(value_x, 96, value_gpool_feat);

    float value_h[128];
    linear(value_gpool_feat, 288, value_h, 128, W->value_linear2_weight, W->value_linear2_bias);
    swish_inplace(value_h, 128);

    linear(value_h, 128, out_value, 3, W->value_linear_valuehead_weight, W->value_linear_valuehead_bias);
    linear(value_h, 128, out_miscvalue, 10, W->value_linear_miscvaluehead_weight, W->value_linear_miscvaluehead_bias);
    linear(value_h, 128, out_moremiscvalue, 8, W->value_linear_moremiscvaluehead_weight, W->value_linear_moremiscvaluehead_bias);

    free(trunk); free(policy_p); free(policy_g); free(policy_spatial); free(value_x);
}


/* ========== Main ========== */

int main(int argc, char** argv) {
    const char* weights_dir = "weights";
    const char* test_dir = "test_data";
    char path[512];

    if (argc >= 2) weights_dir = argv[1];
    if (argc >= 3) test_dir = argv[2];

    printf("=== KataGo b18c384h12tfrs_1 C Implementation ===\n\n");

    printf("Loading weights...\n");
    ModelWeights* W = (ModelWeights*)malloc(sizeof(ModelWeights));
    if (!W) { fprintf(stderr, "Failed to allocate model weights\n"); return 1; }
    load_weights(W, weights_dir);

    printf("Loading test data...\n");
    sprintf(path, "%s/input_spatial.bin", test_dir);
    float* input_spatial = load_bin(path, 22 * 19 * 19);

    sprintf(path, "%s/input_global.bin", test_dir);
    float* input_global = load_bin(path, 19);

    float out_policy[6 * 362];
    float out_value[3];
    float out_miscvalue[10];
    float out_moremiscvalue[8];
    float out_ownership[1 * 19 * 19];

    printf("Running forward pass...\n");
    forward(W, input_spatial, input_global,
            out_policy, out_value, out_miscvalue, out_moremiscvalue, out_ownership);

    printf("\nForward pass complete.\n");

    printf("\nComparing with reference outputs:\n");

    float* ref;

    sprintf(path, "%s/output_policy.bin", test_dir);
    ref = load_bin(path, 6 * 362);
    compare_tensors(out_policy, ref, 6 * 362, "policy");
    free(ref);

    sprintf(path, "%s/output_value.bin", test_dir);
    ref = load_bin(path, 3);
    compare_tensors(out_value, ref, 3, "value");
    free(ref);

    sprintf(path, "%s/output_miscvalue.bin", test_dir);
    ref = load_bin(path, 10);
    compare_tensors(out_miscvalue, ref, 10, "miscvalue");
    free(ref);

    sprintf(path, "%s/output_moremiscvalue.bin", test_dir);
    ref = load_bin(path, 8);
    compare_tensors(out_moremiscvalue, ref, 8, "moremiscvalue");
    free(ref);

    sprintf(path, "%s/output_ownership.bin", test_dir);
    ref = load_bin(path, 19 * 19);
    compare_tensors(out_ownership, ref, 19 * 19, "ownership");
    free(ref);

    free(input_spatial);
    free(input_global);
    free(W);

    printf("\nDone.\n");
    return 0;
}
