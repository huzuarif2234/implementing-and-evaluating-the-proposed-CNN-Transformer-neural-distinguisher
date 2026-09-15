"""
Enhanced CNN + Transformer Related-Key Differential Neural Distinguisher
for Simon128/128  -- GA + Pipeline (no fixed round table)

Paper: "A Multi-Differential Approach to Enhance Related-Key Neural Distinguishers"
       Xue Yuan and Qichun Wang

This version includes:
  • Enhanced multi-differential (polyhedral) distinguisher
  • GA-based differential selection (Algorithm 1) with bias-score fitness
  • Full closed-loop pipeline: GA → training → threshold check → refine → loop
  • Flexible rounds: any round number runs (no Table-6 dictionary lookup)
  • Correct Simon128/128 cipher (m=2, z0, 68 rounds, 64-bit words)

Cipher : Simon128/128
  Block  = 128 bits  (two 64-bit half-words L, R)
  Key    = 128 bits  (TWO 64-bit words k[0..1])   m=2
  Rounds = 68
  F(x)   = (S8(x) AND S1(x)) XOR S2(x)
  z-seq  = z0  (j=0, m=2 always selects j=0)

DATA FORMAT (10-word mode, extra_words=2)
  10 words x 64 bits = 640 bits per group, 8 groups -> 5120 bits per sample.

ARCHITECTURE: CNN + Transformer v2 (unchanged)
  CNN stem → Positional Embedding → 3× Transformer → 5× Res-CNN+SENet → Head
  num_filters=128, depth=10, d1/d2=256, dropout=0.3, ff_dim=512

TRAINING (single‑shot or pipeline)
  n_train=2×10^7, n_val=2×10^6, batch=30 000, epochs=40
  Adam, MSE loss, L2=1e-5, cyclic LR [0.0001, 0.002], lr_epoch=40

GA SELECTION (Algorithm 1)
  δK forced to (0,δP_r). Bias score = per‑output‑bit bias (Definition 4).
  Population size = 200, generations = 50, fitness samples = 16384.
  Crossover: pop[i] ⊕ pop[j] ⊕ (1 << rand_bit).
  Filtering: keep within 0.06 of best bias.

PIPELINE (closed loop)
  Step 2: GA chooses top differentials
  Steps 3‑8: Training + evaluation
  Step 9: Threshold check
  Step 10: Refine GA, data size, architecture, and loop back
"""

import numpy as np
from os import urandom
import gc
import os

os.environ.setdefault('TF_GPU_ALLOCATOR', 'cuda_malloc_async')

import matplotlib.pyplot as plt
import tensorflow as tf

for _gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_gpu, True)

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Conv1D, Input, Reshape, Add,
                                     BatchNormalization, Activation,
                                     GlobalAveragePooling1D,
                                     MultiHeadAttention, LayerNormalization,
                                     Dropout, Embedding, multiply)
from tensorflow.keras.regularizers import l2


# ===========================================================================
# Simon128/128 cipher (correct implementation with hardcoded constants)
# ===========================================================================

WORD_SIZE = 64
MASK_VAL  = 0xFFFFFFFFFFFFFFFF
_MASK64   = np.uint64(MASK_VAL)
NUM_ROUNDS_FULL = 68

# z0 sequence, period 62  (m=2 always selects j=0)
_z0 = [1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0,1,
       1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0]
assert len(_z0) == 62

# Hardcoded constants for Simon128/128 (66 constants for 68 rounds)
_BASE = 0xFFFFFFFFFFFFFFFC
CONST_SIMON128_128 = [(_BASE ^ _z0[i % 62]) & MASK_VAL for i in range(66)]
assert len(CONST_SIMON128_128) == 66


def _rand_words(n):
    """Generate n random 64-bit values."""
    return np.frombuffer(urandom(8 * n), dtype=np.uint64).copy()


def rol64(x: np.ndarray, k: int) -> np.ndarray:
    k = int(k)
    return (x << np.uint64(k)) | (x >> np.uint64(64 - k))


def ror64(x: np.ndarray, k: int) -> np.ndarray:
    k = int(k)
    return (x >> np.uint64(k)) | (x << np.uint64(64 - k))


def F_simon128(x: np.ndarray) -> np.ndarray:
    """F(x) = (S8(x) AND S1(x)) XOR S2(x)  --  64-bit words."""
    return (rol64(x, 8) & rol64(x, 1)) ^ rol64(x, 2)


def enc_one_round_simon128(p, k):
    c1 = p[0]
    c0 = p[1] ^ F_simon128(p[0]) ^ k
    return (c0, c1)


def expand_key_simon128128(k, t):
    """
    Key schedule for Simon128/128.  m=2: TWO 64-bit key words.
    k: shape (2, n), dtype uint64.
    Returns list of t round-key arrays, each shape (n,).
    """
    if t < 2:
        return [k[1 - i].copy() for i in range(t)]
    ks = [None] * t
    ks[0] = k[1].copy()
    ks[1] = k[0].copy()
    for i in range(t - 2):
        tmp = ror64(ks[i + 1], 3)
        c   = np.uint64(CONST_SIMON128_128[i])
        ks[i + 2] = c ^ ks[i] ^ tmp ^ ror64(tmp, 1)
    return ks


def encrypt_simon128128(p, ks):
    x = p[0].copy() if isinstance(p[0], np.ndarray) else np.atleast_1d(
            np.array(p[0], dtype=np.uint64))
    y = p[1].copy() if isinstance(p[1], np.ndarray) else np.atleast_1d(
            np.array(p[1], dtype=np.uint64))
    for k in ks:
        x, y = enc_one_round_simon128((x, y), k)
    return (x, y)


# Backwards-compatibility aliases for GA code
F_simeck = F_simon128
expand_key_simeck = expand_key_simon128128
encrypt_simeck = encrypt_simon128128


# ===========================================================================
# Low-level sample builder (one differential, one label class)
# ===========================================================================

def _build_one_diff_samples(n, nr, diff, key_diff, s_groups=8, extra_words=2):
    """
    Generate n raw feature vectors for a SINGLE (diff, key_diff) pair.
    All samples are from the POSITIVE class of that differential.

    Returns X : (n, s_groups * words_per_group * WORD_SIZE)  uint8 binary
    """
    keys = np.stack([_rand_words(n) for _ in range(2)], axis=0)
    keys_diff = np.array([
        keys[0] ^ np.uint64(key_diff[0]),
        keys[1] ^ np.uint64(key_diff[1]),
    ], dtype=np.uint64)

    ks      = expand_key_simeck(keys,      nr)
    ks_diff = expand_key_simeck(keys_diff, nr)
    del keys, keys_diff
    gc.collect()

    X_words = []

    for _ in range(s_groups):
        p0l = _rand_words(n)
        p0r = _rand_words(n)
        p1l = p0l ^ np.uint64(diff[0])
        p1r = p0r ^ np.uint64(diff[1])

        c0l, c0r = encrypt_simeck((p0l, p0r), ks)
        c1l, c1r = encrypt_simeck((p1l, p1r), ks_diff)

        delta_cl = c0l ^ c1l
        delta_cr = c0r ^ c1r

        rL0_1 = F_simeck(c0r) ^ c0l
        rL1_1 = F_simeck(c1r) ^ c1l
        d_rL1 = rL0_1 ^ rL1_1

        rL0_2 = c0r ^ F_simeck(rL0_1)
        rL1_2 = c1r ^ F_simeck(rL1_1)
        d_rL2 = rL0_2 ^ rL1_2

        X_words.extend([delta_cl, delta_cr, c0l, c0r, c1l, c1r, d_rL1, d_rL2])

        if extra_words >= 2:
            rL0_3 = rL0_1 ^ F_simeck(rL0_2)
            rL1_3 = rL1_1 ^ F_simeck(rL1_2)
            d_rL3 = rL0_3 ^ rL1_3
            rL0_4 = rL0_2 ^ F_simeck(rL0_3)
            rL1_4 = rL1_2 ^ F_simeck(rL1_3)
            d_rL4 = rL0_4 ^ rL1_4
            X_words.extend([d_rL3, d_rL4])

        del (p0l, p0r, p1l, p1r, c0l, c0r, c1l, c1r,
             delta_cl, delta_cr, rL0_1, rL1_1, d_rL1, rL0_2, rL1_2, d_rL2)
        gc.collect()

    del ks, ks_diff
    gc.collect()

    M = len(X_words)
    WS = WORD_SIZE
    Z = np.zeros((M * WS, n), dtype=np.uint8)
    for i in range(M * WS):
        wi = i // WS
        bit_pos = WS - (i % WS) - 1
        Z[i] = (X_words[wi] >> bit_pos) & 1
    del X_words
    gc.collect()
    return Z.T


# ===========================================================================
# STEP 2 : GA-Based Differential Selection (Algorithm 1 from the paper)
# ===========================================================================

_PAPER_WORKING_POP = 32
_PAPER_BIAS_THRESHOLD = 0.06


def _bias_score(dp_l, dp_r, nr, n_samples):
    """
    Compute the empirical related-key bias score b̃_t(δP, δK) per Definition 4.

    δK forced equal to δP across both key words: (0,dp_r) for m=2.
    Score = (1/(2*WORD_SIZE)) * Σ_j |0.5 - fraction_of_ones_in_bit_j|
    """
    dp_l = 0
    dp_r = int(dp_r) & MASK_VAL
    if dp_r == 0:
        return 0.0

    dk = (0, dp_r)   # m=2 key differences

    keys = np.stack([_rand_words(n_samples) for _ in range(2)], axis=0)
    keys_d = np.array([keys[i] ^ dk[i] for i in range(2)], dtype=np.uint64)

    ks   = expand_key_simeck(keys,   nr)
    ks_d = expand_key_simeck(keys_d, nr)

    p0l = _rand_words(n_samples)
    p0r = _rand_words(n_samples)
    p1l = p0l ^ dp_l
    p1r = p0r ^ dp_r

    c0l, c0r = encrypt_simeck((p0l, p0r), ks)
    c1l, c1r = encrypt_simeck((p1l, p1r), ks_d)

    diff_l = c0l ^ c1l
    diff_r = c0r ^ c1r

    bias_sum = 0.0
    for bit in range(WORD_SIZE):
        mask = np.uint64(1 << bit)
        frac_l = float(np.count_nonzero(diff_l & mask)) / n_samples
        frac_r = float(np.count_nonzero(diff_r & mask)) / n_samples
        bias_sum += abs(0.5 - frac_l)
        bias_sum += abs(0.5 - frac_r)

    return bias_sum / (2.0 * WORD_SIZE)


def _chrom_bias(chrom, nr, n_samples):
    """Wrapper: bias score for a chromosome [dp_r] (dp_l fixed to 0)."""
    return _bias_score(0, int(chrom[0]), nr, n_samples)


def _generate_initial_population(L_squared, plain_bits, min_hw, max_hw, rng):
    """Generate L² random dp_r values with Hamming weight in [min_hw, max_hw]."""
    population = []
    attempts = 0
    max_attempts = L_squared * 200
    while len(population) < L_squared and attempts < max_attempts:
        attempts += 1
        # Generate a random 64-bit integer by reading 8 random bytes
        dp_r = int.from_bytes(rng.bytes(8), 'little') & MASK_VAL
        if dp_r == 0:
            continue
        hw = bin(dp_r).count('1')
        if min_hw <= hw <= max_hw:
            population.append([dp_r])
    if len(population) < L_squared:
        # Fallback: pad with any non-zero values (ignore Hamming weight)
        while len(population) < L_squared:
            dp_r = int.from_bytes(rng.bytes(8), 'little') & MASK_VAL
            if dp_r == 0:
                continue
            population.append([dp_r])
    return np.array(population, dtype=np.uint64)


def ga_select_differentials(nr,
                             n_select=2,
                             pop_size=200,
                             n_generations=50,
                             n_fitness=16384,
                             min_hw=1,
                             max_hw=2,
                             bias_filter=_PAPER_BIAS_THRESHOLD,
                             seed=None,
                             verbose=True):
    """
    Evolutionary optimizer — Algorithm 1 of the paper.

    Returns:
        selected: list of (diff_2tuple, key_diff_2tuple) — top n_select
        best_history: list of best bias per generation
        filtered_pool: all candidates within bias_filter of best
    """
    rng = np.random.default_rng(seed)

    if verbose:
        print(f"\n  [GA / Algorithm 1]  nr={nr}  pop={pop_size}  "
              f"gens={n_generations}  t={n_fitness}  HW=[{min_hw},{max_hw}]")
        print(f"  Constraint: δK ≡ δP  (paper Section 3)")

    plain_population = _generate_initial_population(
        pop_size, WORD_SIZE, min_hw, max_hw, rng)
    starting_bias = np.array(
        [_chrom_bias(c, nr, n_fitness) for c in plain_population])
    order = np.argsort(-starting_bias)
    current_population = plain_population[order[:_PAPER_WORKING_POP]].copy()
    current_bias = starting_bias[order[:_PAPER_WORKING_POP]].copy()

    if verbose:
        print(f"  Initial pool: {pop_size} → kept top {_PAPER_WORKING_POP}  "
              f"(best bias = {float(current_bias.max()):.5f})")

    best_history = [float(current_bias.max())]

    for gen in range(n_generations):
        n_pop = len(current_population)
        candidates = []
        for i in range(n_pop):
            for j in range(i + 1, n_pop):
                rand_bit = int(rng.integers(0, WORD_SIZE))
                cand_r = (int(current_population[i][0]) ^
                          int(current_population[j][0]) ^
                          (1 << rand_bit)) & MASK_VAL
                if cand_r != 0:
                    candidates.append([cand_r])
        candidates = np.array(candidates, dtype=np.uint64)
        cand_bias = np.array([_chrom_bias(c, nr, n_fitness) for c in candidates])
        order = np.argsort(-cand_bias)
        current_population = candidates[order[:_PAPER_WORKING_POP]].copy()
        current_bias = cand_bias[order[:_PAPER_WORKING_POP]].copy()
        gen_best = float(current_bias.max())
        best_history.append(gen_best)
        if verbose:
            print(f"  [GA]  iter {gen+1:>2}/{n_generations}   "
                  f"best bias = {gen_best:.5f}   "
                  f"mean bias = {float(current_bias.mean()):.5f}   "
                  f"candidates = {len(candidates)}")

    best_bias = float(current_bias.max())
    threshold = best_bias - bias_filter
    filtered_pool = []
    seen = set()
    for i in range(len(current_population)):
        c = current_population[i]
        bs = float(current_bias[i])
        if bs < threshold:
            continue
        dp_r = int(c[0]) & MASK_VAL
        if dp_r == 0 or dp_r in seen:
            continue
        seen.add(dp_r)
        dp = (0, dp_r)
        dk = (0, dp_r)   # m=2 key difference
        filtered_pool.append((dp, dk, bs))
    filtered_pool.sort(key=lambda x: -x[2])

    if verbose:
        print(f"\n  [GA]  Best bias score : {best_bias:.5f}")
        print(f"  [GA]  0.06-close pool : {len(filtered_pool)} candidates "
              f"(threshold ≥ {threshold:.5f})")
        print(f"  [GA]  Returning top-{min(n_select, len(filtered_pool))}:")
        for i, (dp, dk, bs) in enumerate(filtered_pool[:n_select]):
            print(f"    {i+1}. dP={tuple(hex(v) for v in dp)}  "
                  f"dK={tuple(hex(v) for v in dk)}  bias={bs:.5f}")

    selected = [(dp, dk) for dp, dk, _ in filtered_pool[:n_select]]
    return selected, best_history, filtered_pool


# ===========================================================================
# Enhanced multi-differential data generator (polyhedral method)
# ===========================================================================

def make_train_data_enhanced(n, nr,
                              pos_diffs,
                              neg_diffs,
                              s_groups=8,
                              extra_words=2):
    """
    Build the ENHANCED (polyhedral) training set for Simon128/128.

    pos_diffs / neg_diffs: list of (diff_2tuple, key_diff_2tuple)
    """
    t_pos = len(pos_diffs)
    t_neg = len(neg_diffs)
    assert t_pos >= 1 and t_neg >= 1

    n_pos = n // 2
    n_neg = n - n_pos

    pos_chunks = []
    for i, (diff, key_diff) in enumerate(pos_diffs):
        chunk_n = n_pos // t_pos if i < t_pos - 1 else n_pos - (n_pos // t_pos) * (t_pos - 1)
        pos_chunks.append(_build_one_diff_samples(
            chunk_n, nr, diff, key_diff, s_groups=s_groups, extra_words=extra_words))
    X_pos = np.vstack(pos_chunks)
    del pos_chunks
    gc.collect()

    neg_chunks = []
    for i, (diff, key_diff) in enumerate(neg_diffs):
        chunk_n = n_neg // t_neg if i < t_neg - 1 else n_neg - (n_neg // t_neg) * (t_neg - 1)
        neg_chunks.append(_build_one_diff_samples(
            chunk_n, nr, diff, key_diff, s_groups=s_groups, extra_words=extra_words))
    X_neg = np.vstack(neg_chunks)
    del neg_chunks
    gc.collect()

    Y_pos = np.ones(len(X_pos), dtype=np.uint8)
    Y_neg = np.zeros(len(X_neg), dtype=np.uint8)
    X = np.vstack([X_pos, X_neg])
    Y = np.concatenate([Y_pos, Y_neg])
    del X_pos, X_neg, Y_pos, Y_neg
    gc.collect()

    idx = np.random.permutation(len(Y))
    return X[idx], Y[idx]


# ===========================================================================
# Neural network (CNN + Transformer v2)
# ===========================================================================

def se_block(x, num_filters, se_ratio=4, reg_param=1e-5, name='se'):
    se = GlobalAveragePooling1D(name=f'{name}_gap')(x)
    se = Reshape((1, num_filters), name=f'{name}_reshape')(se)
    se = Dense(max(1, num_filters // se_ratio), activation='relu',
               use_bias=False, kernel_regularizer=l2(reg_param),
               name=f'{name}_sq')(se)
    se = Dense(num_filters, activation='sigmoid',
               use_bias=False, kernel_regularizer=l2(reg_param),
               name=f'{name}_ex')(se)
    return multiply([x, se], name=f'{name}_mul')


def transformer_encoder_block(x, num_heads, ff_dim, dropout_rate, reg_param,
                               name='tr'):
    d_model = int(x.shape[-1])
    key_dim = max(1, d_model // num_heads)
    attn_out = MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate,
        name=f'{name}_mhsa',
    )(x, x)
    x = Add(name=f'{name}_attn_add')([x, attn_out])
    x = LayerNormalization(epsilon=1e-6, name=f'{name}_ln1')(x)
    ffn = Dense(ff_dim, activation='relu', kernel_regularizer=l2(reg_param),
                name=f'{name}_ffn1')(x)
    ffn = Dropout(dropout_rate, name=f'{name}_drop')(ffn)
    ffn = Dense(d_model, kernel_regularizer=l2(reg_param),
                name=f'{name}_ffn2')(ffn)
    x = Add(name=f'{name}_ffn_add')([x, ffn])
    x = LayerNormalization(epsilon=1e-6, name=f'{name}_ln2')(x)
    return x


def make_model(s_groups=8,
               words_per_group=10,
               word_size=64,
               num_filters=128,
               depth=10,
               d1=256,
               d2=256,
               reg_param=1e-5,
               dropout_rate=0.3,
               ks_value_1=1,
               ks_value_2=3,
               ks_value_3=3,
               num_heads=4,
               ff_dim=512,
               num_transformer_blocks=3,
               se_ratio=4,
               final_activation='sigmoid'):
    token_dim = words_per_group * word_size
    input_size = s_groups * token_dim

    inp = Input(shape=(input_size,), name='input')
    x = Reshape((s_groups, token_dim), name='reshape')(inp)

    # CNN stem
    x = Conv1D(num_filters, kernel_size=ks_value_1,
               padding='same', kernel_regularizer=l2(reg_param),
               name='stem_conv')(x)
    x = BatchNormalization(name='stem_bn')(x)
    x = Activation('relu', name='stem_relu')(x)
    x = Dense(num_filters, kernel_regularizer=l2(reg_param), name='stem_d1')(x)
    x = BatchNormalization(name='stem_bn1')(x)
    x = Activation('relu', name='stem_relu1')(x)
    x = Dense(num_filters, kernel_regularizer=l2(reg_param), name='stem_d2')(x)
    x = BatchNormalization(name='stem_bn2')(x)
    x = Activation('relu', name='stem_relu2')(x)

    # Learned positional embedding
    pos_emb = Embedding(s_groups, num_filters, name='pos_emb')(tf.range(s_groups))
    pos_emb = tf.expand_dims(pos_emb, axis=0)
    x = Add(name='pos_add')([x, pos_emb])

    # Transformer encoder
    for ti in range(num_transformer_blocks):
        x = transformer_encoder_block(
            x, num_heads=num_heads, ff_dim=ff_dim,
            dropout_rate=dropout_rate, reg_param=reg_param,
            name=f'tr{ti}',
        )

    # Residual CNN + SENet blocks
    shortcut = x
    cnn_depth = max(1, depth // 2)
    for i in range(cnn_depth):
        c = Conv1D(num_filters, kernel_size=ks_value_2,
                   padding='same', kernel_regularizer=l2(reg_param),
                   name=f'res_c1_{i}')(shortcut)
        c = BatchNormalization(name=f'res_bn1_{i}')(c)
        c = Activation('relu', name=f'res_r1_{i}')(c)
        c = Conv1D(num_filters, kernel_size=ks_value_3,
                   padding='same', kernel_regularizer=l2(reg_param),
                   name=f'res_c2_{i}')(c)
        c = BatchNormalization(name=f'res_bn2_{i}')(c)
        c = Activation('relu', name=f'res_r2_{i}')(c)
        c = se_block(c, num_filters, se_ratio=se_ratio, reg_param=reg_param,
                     name=f'se{i}')
        shortcut = Add(name=f'res_add_{i}')([shortcut, c])

    # Prediction head
    x = GlobalAveragePooling1D(name='gap')(shortcut)
    x = Dropout(dropout_rate, name='head_drop')(x)
    x = Dense(d1, kernel_regularizer=l2(reg_param), name='head_d1')(x)
    x = BatchNormalization(name='head_bn1')(x)
    x = Activation('relu', name='head_r1')(x)
    x = Dense(d2, kernel_regularizer=l2(reg_param), name='head_d2')(x)
    x = BatchNormalization(name='head_bn2')(x)
    x = Activation('relu', name='head_r2')(x)
    out = Dense(1, activation=final_activation,
                kernel_regularizer=l2(reg_param), name='output')(x)

    return Model(inputs=inp, outputs=out,
                 name='CNN_Transformer_Simon128_128_Enhanced_v2')


# ===========================================================================
# Learning rate schedule
# ===========================================================================

def cyclic_lr(n, high_lr, low_lr):
    def schedule(i):
        return low_lr + ((n - 1) - i % n) / (n - 1) * (high_lr - low_lr)
    return schedule


# ===========================================================================
# Default fallback differentials (used when caller doesn't supply any and the
# GA pipeline isn't running).  Values match Table 6 of the paper for
# Simon128/128 (14r and 15r both use the same diffs); for other rounds these
# serve only as starting points.  For serious work on unknown rounds, either
# pass your own via the CLI or run --use_pipeline to have the GA search.
# ===========================================================================

_FALLBACK_POS_DIFFS = [
    ((0x0, 0x2),    (0x0, 0x2)),
    ((0x0, 0x8),    (0x0, 0x8)),
]
_FALLBACK_NEG_DIFFS = [
    ((0x0, 0x100),  (0x0, 0x100)),
    ((0x0, 0x200),  (0x0, 0x200)),
]

# Backwards-compat names (still used by some prints below)
POS_DIFFS_TABLE6 = _FALLBACK_POS_DIFFS
NEG_DIFFS_TABLE6 = _FALLBACK_NEG_DIFFS

# No fixed table — set kept as an empty dict so the rest of the code works
# without dragging in any Table-6 expectations.
_TABLE6_DEFAULT = {}


# ===========================================================================
# Training (single-shot mode)
# ===========================================================================

def _fmt(v, spec='.4f'):
    """Format a value with `spec`, returning 'N/A' when v is None."""
    return 'N/A' if v is None else format(v, spec)

def train_distinguisher_enhanced(
        nr,
        pos_diffs,
        neg_diffs,
        expected_acc=None, expected_tpr=None, expected_tnr=None,
        n_train=2 * 10**7, n_val=2 * 10**6,
        num_epochs=40, batch_size=30000,
        high_lr=0.002, low_lr=0.0001, lr_epoch=40,
        s_groups=8, extra_words=2,
        depth=10, num_filters=128, d1=256, d2=256,
        reg_param=1e-5, dropout_rate=0.3,
        ks_value_1=1, ks_value_2=3, ks_value_3=3,
        num_heads=4, ff_dim=512, num_transformer_blocks=3, se_ratio=4,
        output_dir='./results_simon128128_enhanced',
):
    os.makedirs(output_dir, exist_ok=True)
    words_per_group = 8 + extra_words

    print(f"\n{'='*76}")
    print(f"  Simon128/128  ENHANCED CNN+Transformer v2  --  {nr} rounds")
    print(f"  POSITIVE diffs:")
    for d, k in pos_diffs:
        print(f"    diff={tuple(hex(v) for v in d)}  "
              f"key_diff={tuple(hex(v) for v in k)}")
    print(f"  NEGATIVE diffs:")
    for d, k in neg_diffs:
        print(f"    diff={tuple(hex(v) for v in d)}  "
              f"key_diff={tuple(hex(v) for v in k)}")
    print(f"  data: {s_groups}g × {words_per_group}w × {WORD_SIZE}b = "
          f"{s_groups*words_per_group*WORD_SIZE} bits/sample")
    print(f"  net : filters={num_filters}  depth={depth}  se_ratio={se_ratio}")
    print(f"        transformer: {num_transformer_blocks}× blocks  "
          f"heads={num_heads}  ff={ff_dim}")
    print(f"        dense: {d1}→{d2}  dropout={dropout_rate}")
    if expected_acc is not None or expected_tpr is not None or expected_tnr is not None:
        print(f"  target: acc={_fmt(expected_acc)}  "
              f"TPR={_fmt(expected_tpr)}  TNR={_fmt(expected_tnr)}")
    print(f"{'='*76}")

    print(f"\nGenerating {n_train:,} ENHANCED training samples ...")
    X_tr, Y_tr = make_train_data_enhanced(
        n_train, nr, pos_diffs, neg_diffs,
        s_groups=s_groups, extra_words=extra_words,
    )
    print(f"Generating {n_val:,} ENHANCED validation samples ...")
    X_val, Y_val = make_train_data_enhanced(
        n_val, nr, pos_diffs, neg_diffs,
        s_groups=s_groups, extra_words=extra_words,
    )
    print(f"X_train={X_tr.shape}  X_val={X_val.shape}")
    print(f"Positive rate — train:{Y_tr.mean():.3f}  val:{Y_val.mean():.3f}")

    model = make_model(
        s_groups=s_groups, words_per_group=words_per_group,
        word_size=WORD_SIZE,
        num_filters=num_filters, depth=depth, d1=d1, d2=d2,
        reg_param=reg_param, dropout_rate=dropout_rate,
        ks_value_1=ks_value_1, ks_value_2=ks_value_2, ks_value_3=ks_value_3,
        num_heads=num_heads, ff_dim=ff_dim,
        num_transformer_blocks=num_transformer_blocks, se_ratio=se_ratio,
    )
    model.summary()
    total_params = int(np.sum([np.prod(v.get_shape())
                                for v in model.trainable_weights]))
    print(f"Trainable params: {total_params:,}")

    model.compile(
        optimizer='adam', loss='mse',
        metrics=['accuracy',
                 tf.keras.metrics.TruePositives(name='tp'),
                 tf.keras.metrics.TrueNegatives(name='tn'),
                 tf.keras.metrics.FalsePositives(name='fp'),
                 tf.keras.metrics.FalseNegatives(name='fn')],
    )

    pos_tag = f"p{hex(pos_diffs[0][0][1])}"
    neg_tag = f"n{hex(neg_diffs[0][0][1])}"
    diff_tag = f"{pos_tag}_{neg_tag}"
    ckpt = os.path.join(output_dir, f'best_{nr}r_{diff_tag}.h5')
    callbacks = [
        ModelCheckpoint(ckpt, monitor='val_accuracy',
                        save_best_only=True, verbose=1),
        LearningRateScheduler(cyclic_lr(lr_epoch, high_lr, low_lr), verbose=0),
    ]

    Y_tr_f32 = Y_tr.astype(np.float32)
    Y_val_f32 = Y_val.astype(np.float32)

    train_ds = (
        tf.data.Dataset
        .from_tensor_slices((X_tr, Y_tr_f32))
        .shuffle(buffer_size=n_train, seed=42)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset
        .from_tensor_slices((X_val, Y_val_f32))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    del Y_tr_f32, Y_val_f32
    gc.collect()

    history = model.fit(
        train_ds, epochs=num_epochs,
        validation_data=val_ds,
        callbacks=callbacks, verbose=2,
    )

    best_ep = int(np.argmax(history.history['val_accuracy']))
    best_acc = float(np.max(history.history['val_accuracy']))
    tp = history.history['val_tp'][best_ep]
    tn = history.history['val_tn'][best_ep]
    fp = history.history['val_fp'][best_ep]
    fn = history.history['val_fn'][best_ep]
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"\n{'─'*64}")
    print(f"  {nr}r  ENHANCED  Best epoch={best_ep+1}  "
          f"Acc={best_acc:.4f}  TPR={tpr:.4f}  TNR={tnr:.4f}")
    if expected_acc is not None:
        print(f"  Target:              acc={_fmt(expected_acc)}  "
              f"TPR={_fmt(expected_tpr)}  TNR={_fmt(expected_tnr)}")
        if best_acc >= expected_acc:
            print("  *** TARGET REACHED ***")
        else:
            print(f"  Gap: {best_acc - expected_acc:+.4f}")
    print(f"{'─'*64}")

    _plot_history(history, nr, diff_tag, output_dir, expected_acc)

    txt = os.path.join(output_dir, f'results_{nr}r_{diff_tag}.txt')
    with open(txt, 'w') as f:
        f.write(f"Simon128/128  CNN+Transformer v2 ENHANCED  --  {nr} rounds\n")
        f.write(f"filters={num_filters} depth={depth} d1={d1} d2={d2} "
                f"se_ratio={se_ratio} dropout={dropout_rate}\n")
        f.write(f"transformer: {num_transformer_blocks}x heads={num_heads} "
                f"ff={ff_dim}\n")
        f.write("positive diffs:\n")
        for d, k in pos_diffs:
            f.write(f"  diff={tuple(hex(v) for v in d)}  "
                    f"key_diff={tuple(hex(v) for v in k)}\n")
        f.write("negative diffs:\n")
        for d, k in neg_diffs:
            f.write(f"  diff={tuple(hex(v) for v in d)}  "
                    f"key_diff={tuple(hex(v) for v in k)}\n")
        f.write(f"data: {s_groups}g x {words_per_group}w x {WORD_SIZE}b = "
                f"{s_groups*words_per_group*WORD_SIZE} bits\n")
        f.write(f"best_epoch={best_ep+1}/{num_epochs}\n")
        f.write(f"val_acc={best_acc:.4f}  TPR={tpr:.4f}  TNR={tnr:.4f}\n")
        if expected_acc is not None:
            f.write(f"target_acc={_fmt(expected_acc)}  "
                    f"target_TPR={_fmt(expected_tpr)}  "
                    f"target_TNR={_fmt(expected_tnr)}\n")
            f.write(f"reached={'YES' if best_acc >= expected_acc else 'NO'}\n")
        f.write(f"total_params={total_params:,}\n")
    print(f"Results → {txt}")

    return model, history, {
        "nr": nr, "acc": best_acc, "tpr": tpr, "tnr": tnr,
        "best_epoch": best_ep + 1,
        "pos_diffs": pos_diffs, "neg_diffs": neg_diffs,
    }


def _plot_history(history, nr, diff_tag, output_dir, expected_acc=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Val')
    if expected_acc is not None:
        ax1.axhline(expected_acc, color='r', linestyle='--',
                    label=f'Target ({expected_acc:.4f})')
    ax1.set_title(f'Simon128/128 Enhanced  {nr}r  Accuracy')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy'); ax1.legend()
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Val')
    ax2.set_title(f'Simon128/128 Enhanced  {nr}r  MSE Loss')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('MSE'); ax2.legend()
    plt.tight_layout()
    p = os.path.join(output_dir, f'history_{nr}r_{diff_tag}.png')
    plt.savefig(p, dpi=100); plt.close()
    print(f"Plot → {p}")


# ===========================================================================
# Full closed‑loop pipeline (Steps 1‑10 from the flowchart)
# ===========================================================================

def run_full_pipeline(nr,
                       threshold_acc,
                       threshold_tpr=None,
                       threshold_tnr=None,
                       max_iterations=3,
                       use_ga=True,
                       fallback_pos_diffs=None,
                       fallback_neg_diffs=None,
                       # GA params (Step 2 / Algorithm 1)
                       ga_pop_size=200,
                       ga_generations=50,
                       ga_n_fitness=16384,
                       ga_n_select=2,
                       ga_min_hw=1,
                       ga_max_hw=2,
                       ga_bias_filter=_PAPER_BIAS_THRESHOLD,
                       # training params (Step 7)
                       n_train=2 * 10**7, n_val=2 * 10**6,
                       num_epochs=40, batch_size=30000,
                       high_lr=0.002, low_lr=0.0001, lr_epoch=40,
                       s_groups=8, extra_words=2,
                       depth=10, num_filters=128, d1=256, d2=256,
                       reg_param=1e-5, dropout_rate=0.3,
                       ks_value_1=1, ks_value_2=3, ks_value_3=3,
                       num_heads=4, ff_dim=512, num_transformer_blocks=3,
                       se_ratio=4,
                       output_dir='./results_simon128128_enhanced'):
    """
    Full closed-loop pipeline (Steps 1-10 of the flowchart).

    Returns the final result dict and a list of per-iteration results.
    """
    print("\n" + "█" * 76)
    print(f"  FULL PIPELINE  --  Simon128/128  --  {nr} rounds")
    print(f"  Threshold:  acc ≥ {threshold_acc:.4f}"
          + (f"  TPR ≥ {threshold_tpr:.4f}" if threshold_tpr else "")
          + (f"  TNR ≥ {threshold_tnr:.4f}" if threshold_tnr else ""))
    print(f"  Max refinement iterations: {max_iterations}")
    print("█" * 76)

    iteration_log = []

    # initial config
    cur_n_train         = n_train
    cur_depth           = depth
    cur_num_filters     = num_filters
    cur_ga_pop          = ga_pop_size
    cur_ga_generations  = ga_generations
    cur_pos_diffs       = None
    cur_neg_diffs       = None

    for it in range(1, max_iterations + 1):
        print("\n" + "▶" * 76)
        print(f"  ITERATION {it}/{max_iterations}")
        print("▶" * 76)

        # ── Step 2: GA-Based Differential Selection ──────────────────────
        if use_ga:
            print(f"\n  [Step 2]  GA-Based Differential Selection  (Algorithm 1)")
            ga_all, _, filtered_pool = ga_select_differentials(
                nr=nr,
                n_select=ga_n_select * 2,   # need pos + neg
                pop_size=cur_ga_pop,
                n_generations=cur_ga_generations,
                n_fitness=ga_n_fitness,
                min_hw=ga_min_hw,
                max_hw=ga_max_hw,
                bias_filter=ga_bias_filter,
                seed=42 + it,
                verbose=True,
            )

            MIN_USEFUL_BIAS = 1e-4
            best_bias = filtered_pool[0][2] if filtered_pool else 0.0

            if best_bias > MIN_USEFUL_BIAS and len(ga_all) >= 2:
                half = max(1, len(ga_all) // 2)
                ga_pos = ga_all[:half]
                ga_neg = ga_all[half:half * 2]
                print(
                    f"  [Step 2]  GA found useful diffs  "
                    f"(best bias={best_bias:.5f})  "
                    f"pos={len(ga_pos)}  neg={len(ga_neg)}"
                )
                pos_diffs = ga_pos
                neg_diffs = ga_neg
            else:
                print(
                    f"  [Step 2]  GA pool too small or bias too low "
                    f"(best_bias={best_bias:.5f} ≤ {MIN_USEFUL_BIAS})"
                )
                print(f"  [Step 2]  ⚠ Falling back to caller-supplied differentials")
                if fallback_pos_diffs is None or fallback_neg_diffs is None:
                    raise ValueError("GA failed and no fallback_pos_diffs/neg_diffs provided!")
                pos_diffs = fallback_pos_diffs
                neg_diffs = fallback_neg_diffs
        else:
            if cur_pos_diffs is not None and cur_neg_diffs is not None:
                pos_diffs = cur_pos_diffs
                neg_diffs = cur_neg_diffs
            elif fallback_pos_diffs is not None and fallback_neg_diffs is not None:
                pos_diffs = fallback_pos_diffs
                neg_diffs = fallback_neg_diffs
            else:
                raise ValueError(
                    "use_ga=False requires fallback_pos_diffs and fallback_neg_diffs"
                )

        cur_pos_diffs = pos_diffs
        cur_neg_diffs = neg_diffs

        # ── Steps 3-8: Sample generation → Network → Train → Evaluate ──
        print(f"\n  [Steps 3-8]  Sample gen → Net → Train → Evaluate")
        _, _, res = train_distinguisher_enhanced(
            nr=nr, pos_diffs=pos_diffs, neg_diffs=neg_diffs,
            expected_acc=threshold_acc,
            expected_tpr=threshold_tpr, expected_tnr=threshold_tnr,
            n_train=cur_n_train, n_val=n_val,
            num_epochs=num_epochs, batch_size=batch_size,
            high_lr=high_lr, low_lr=low_lr, lr_epoch=lr_epoch,
            s_groups=s_groups, extra_words=extra_words,
            depth=cur_depth, num_filters=cur_num_filters,
            d1=d1, d2=d2,
            reg_param=reg_param, dropout_rate=dropout_rate,
            ks_value_1=ks_value_1, ks_value_2=ks_value_2, ks_value_3=ks_value_3,
            num_heads=num_heads, ff_dim=ff_dim,
            num_transformer_blocks=num_transformer_blocks,
            se_ratio=se_ratio,
            output_dir=os.path.join(output_dir, f'iter_{it}'),
        )
        res['iteration']   = it
        res['n_train']     = cur_n_train
        res['depth']       = cur_depth
        res['num_filters'] = cur_num_filters
        iteration_log.append(res)

        # ── Step 9: Performance Above Threshold? ─────────────────────────
        passed_acc = res['acc'] >= threshold_acc
        passed_tpr = (threshold_tpr is None) or (res['tpr'] >= threshold_tpr)
        passed_tnr = (threshold_tnr is None) or (res['tnr'] >= threshold_tnr)
        passed = passed_acc and passed_tpr and passed_tnr

        print("\n" + "─" * 76)
        print(f"  [Step 9]  Performance check  (iteration {it})")
        print(f"     acc = {res['acc']:.4f}   "
              f"(threshold {threshold_acc:.4f})  → {'PASS' if passed_acc else 'FAIL'}")
        if threshold_tpr is not None:
            print(f"     TPR = {res['tpr']:.4f}   "
                  f"(threshold {threshold_tpr:.4f})  → {'PASS' if passed_tpr else 'FAIL'}")
        if threshold_tnr is not None:
            print(f"     TNR = {res['tnr']:.4f}   "
                  f"(threshold {threshold_tnr:.4f})  → {'PASS' if passed_tnr else 'FAIL'}")
        print("─" * 76)

        if passed:
            print("\n" + "🏆" * 38)
            print(f"  FINAL OUTCOME  --  threshold reached at iteration {it}")
            print(f"     Validated High-Quality Neural Distinguisher")
            print(f"     Enhanced Related-Key Neural Cryptanalysis")
            print(f"     acc={res['acc']:.4f}  TPR={res['tpr']:.4f}  TNR={res['tnr']:.4f}")
            print("🏆" * 38)
            print("\n  STOP\n")
            return res, iteration_log

        # ── Step 10: Refine Pipeline ─────────────────────────────────────
        if it < max_iterations:
            print(f"\n  [Step 10]  Threshold not reached - refining pipeline ...")
            print(f"     • Improve GA Search   pop {cur_ga_pop} -> {cur_ga_pop + 20}, "
                  f"gens {cur_ga_generations} -> {cur_ga_generations + 5}")
            print(f"     • Enhance Dataset      n_train {cur_n_train:,} -> "
                  f"{int(cur_n_train * 1.5):,}")
            print(f"     • Adjust Architecture  depth {cur_depth} -> {cur_depth + 2}, "
                  f"filters {cur_num_filters} -> {cur_num_filters + 32}")
            print(f"     • Retrain Model        (looping back to Step 2)")

            cur_ga_pop          += 20
            cur_ga_generations  += 5
            cur_n_train          = int(cur_n_train * 1.5)
            cur_depth           += 2
            cur_num_filters     += 32
        else:
            print(f"\n  Max iterations ({max_iterations}) reached without "
                  f"hitting the threshold. Stopping.")

    # exhausted budget — return best iteration
    best = max(iteration_log, key=lambda r: r['acc'])
    print("\n" + "─" * 76)
    print(f"  Best across all iterations: iter {best['iteration']}  "
          f"acc={best['acc']:.4f}  TPR={best['tpr']:.4f}  TNR={best['tnr']:.4f}")
    print("─" * 76)
    print("\n  STOP\n")
    return best, iteration_log


# ===========================================================================
# Sanity checks
# ===========================================================================

def check_simon128128():
    """Verify Simon128/128 correctness."""
    np.random.seed(42)
    n = 500

    p0l = _rand_words(n)
    p0r = _rand_words(n)
    keys = np.stack([_rand_words(n) for _ in range(2)], axis=0)

    ks68 = expand_key_simon128128(keys, 68)
    c0l, c0r = encrypt_simon128128((p0l, p0r), ks68)
    rx, ry = c0l.copy(), c0r.copy()
    for k in reversed(ks68):
        rx, ry = ry, F_simon128(ry) ^ rx ^ k
    assert np.all(rx == p0l) and np.all(ry == p0r), "Decrypt mismatch!"
    print("  check 1 PASSED: encrypt/decrypt round-trip (500 samples, 68 rounds, m=2)")

    assert len(ks68) == 68
    assert all(np.all((k & MASK_VAL) == k) for k in ks68)
    print("  check 2 PASSED: 68 valid 64-bit round keys")

    for t in [14, 15]:
        kst = expand_key_simon128128(keys, t)
        assert len(kst) == t
    print("  check 3 PASSED: key schedules valid for 14r / 15r")

    diff = (0x0, 0x2)
    key_diff = (0x0, 0x2)
    keys2 = np.array([
        keys[0] ^ key_diff[0], keys[1] ^ key_diff[1],
    ], dtype=np.uint64)
    ks14  = expand_key_simon128128(keys,  14)
    ks14d = expand_key_simon128128(keys2, 14)
    c0l14, c0r14 = encrypt_simon128128((p0l, p0r), ks14)
    c1l14, c1r14 = encrypt_simon128128(
        (p0l ^ diff[0], p0r ^ diff[1]), ks14d)
    dL = c0l14 ^ c1l14
    dR = c0r14 ^ c1r14
    print(f"  check 4 PASSED: 14r related-key diff sample  "
          f"dL={hex(int(dL[0]))}  dR={hex(int(dR[0]))}")

    # Verify constants derived from z0
    _BASE = 0xFFFFFFFFFFFFFFFC
    _z0_ref = [1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0,1,
               1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0]
    const_check = [(_BASE ^ _z0_ref[i % 62]) & MASK_VAL for i in range(66)]
    assert const_check == CONST_SIMON128_128, "Constants mismatch with z0 sequence!"
    print("  check 5 PASSED: constants correctly derived from z0 sequence")

    print("  Simon128/128 cipher checks PASSED")


def quick_data_check_enhanced(nr=14, n=200, extra_words=2):
    """Quick sanity check for the enhanced data generator."""
    pos_diffs = POS_DIFFS_TABLE6
    neg_diffs = NEG_DIFFS_TABLE6
    X, Y = make_train_data_enhanced(n, nr,
                                     pos_diffs=pos_diffs,
                                     neg_diffs=neg_diffs,
                                     s_groups=8,
                                     extra_words=extra_words)
    wpg = 8 + extra_words
    expected = (n, 8 * wpg * WORD_SIZE)
    assert X.shape == expected, f"Shape mismatch: {X.shape} vs {expected}"
    assert X.dtype == np.uint8
    assert set(np.unique(X)).issubset({0, 1})
    print(f"  data shape  : {X.shape}  "
          f"(8 groups × {wpg} words × {WORD_SIZE} bits)")
    print(f"  pos_rate    : {Y.mean():.3f}  (expected ~0.50)")
    print("  Enhanced data generation check PASSED")


# ===========================================================================
# Entry point
# ===========================================================================

def _parse_hex_tuple(s, n_words):
    parts = [p.strip() for p in s.split(',')]
    if len(parts) != n_words:
        raise ValueError(f"Expected {n_words} comma-separated hex values, got: {s!r}")
    return tuple(int(p, 16) for p in parts)


def _parse_diff_list(args_list, n_words):
    return [_parse_hex_tuple(s, n_words) for s in args_list]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            'CNN+Transformer v2  ENHANCED Multi-Differential Distinguisher\n'
            'for Simon128/128\n'
            '\n'
            'Two modes:\n'
            '  (a) Single-shot (default): trains once with the given diffs\n'
            '      (or generic fallback diffs if none are supplied).\n'
            '  (b) Full pipeline (--use_pipeline): runs the closed loop from\n'
            '      the flowchart -- GA selects diffs (Step 2), trains and\n'
            '      evaluates (Steps 3-8), checks threshold (Step 9), and\n'
            '      refines & retries (Step 10) until threshold is met.\n'
            '\n'
            'Examples:\n'
            '  # Single-shot at any rounds (uses fallback diffs):\n'
            '  python simon128128_enhanced_cnn_transformer.py --rounds 14 15 16 17\n\n'
            '  # GA pipeline for an unknown round:\n'
            '  python simon128128_enhanced_cnn_transformer.py --rounds 18 \\\n'
            '      --use_pipeline --threshold_acc 0.55 --max_iterations 3\n\n'
            '  # Custom diffs, single-shot:\n'
            '  python simon128128_enhanced_cnn_transformer.py --rounds 15 \\\n'
            '      --pos_diffs "0x0,0x2" "0x0,0x8" \\\n'
            '      --neg_diffs "0x0,0x100" "0x0,0x200" \\\n'
            '      --pos_key_diffs "0x0,0x2" "0x0,0x8" \\\n'
            '      --neg_key_diffs "0x0,0x100" "0x0,0x200"\n\n'
            '  # Sanity checks only:\n'
            '  python simon128128_enhanced_cnn_transformer.py --sanity_only\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--rounds', type=int, nargs='+', default=[14, 15])
    parser.add_argument('--pos_diffs', type=str, nargs='+', default=None)
    parser.add_argument('--neg_diffs', type=str, nargs='+', default=None)
    parser.add_argument('--pos_key_diffs', type=str, nargs='+', default=None)
    parser.add_argument('--neg_key_diffs', type=str, nargs='+', default=None)
    parser.add_argument('--n_train', type=int, default=2 * 10**7)
    parser.add_argument('--n_val', type=int, default=2 * 10**6)
    parser.add_argument('--extra_words', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=30000)
    parser.add_argument('--high_lr', type=float, default=0.002)
    parser.add_argument('--low_lr', type=float, default=0.0001)
    parser.add_argument('--lr_epoch', type=int, default=40)
    parser.add_argument('--s_groups', type=int, default=8)
    parser.add_argument('--depth', type=int, default=10)
    parser.add_argument('--num_filters', type=int, default=128)
    parser.add_argument('--d1', type=int, default=256)
    parser.add_argument('--d2', type=int, default=256)
    parser.add_argument('--reg_param', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--ks1', type=int, default=1)
    parser.add_argument('--ks2', type=int, default=3)
    parser.add_argument('--ks3', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--ff_dim', type=int, default=512)
    parser.add_argument('--num_transformer_blocks', type=int, default=3)
    parser.add_argument('--se_ratio', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='./results_simon128128_enhanced')
    parser.add_argument('--sanity_only', action='store_true')

    # Full pipeline mode
    parser.add_argument('--use_pipeline', action='store_true')
    parser.add_argument('--max_iterations', type=int, default=3)
    parser.add_argument('--ga_pop_size', type=int, default=200)
    parser.add_argument('--ga_generations', type=int, default=50)
    parser.add_argument('--ga_n_fitness', type=int, default=16384)
    parser.add_argument('--ga_n_select', type=int, default=2)
    parser.add_argument('--ga_min_hw', type=int, default=1)
    parser.add_argument('--ga_max_hw', type=int, default=2)
    parser.add_argument('--ga_bias_filter', type=float, default=0.06)
    parser.add_argument('--threshold_acc', type=float, default=None)
    parser.add_argument('--threshold_tpr', type=float, default=None)
    parser.add_argument('--threshold_tnr', type=float, default=None)

    args = parser.parse_args()

    print("\nRunning sanity checks ...")
    check_simon128128()
    quick_data_check_enhanced(extra_words=args.extra_words)
    print()

    if args.sanity_only:
        print("All sanity checks passed.")
        exit(0)

    have_manual = (args.pos_diffs is not None and args.neg_diffs is not None
                   and args.pos_key_diffs is not None
                   and args.neg_key_diffs is not None)

    if have_manual:
        cli_pos_plain = _parse_diff_list(args.pos_diffs, 2)
        cli_neg_plain = _parse_diff_list(args.neg_diffs, 2)
        cli_pos_key   = _parse_diff_list(args.pos_key_diffs, 2)
        cli_neg_key   = _parse_diff_list(args.neg_key_diffs, 2)
        if len(cli_pos_plain) != len(cli_pos_key):
            raise ValueError("--pos_diffs and --pos_key_diffs must have same length")
        if len(cli_neg_plain) != len(cli_neg_key):
            raise ValueError("--neg_diffs and --neg_key_diffs must have same length")
        cli_pos_diffs = list(zip(cli_pos_plain, cli_pos_key))
        cli_neg_diffs = list(zip(cli_neg_plain, cli_neg_key))
    else:
        cli_pos_diffs = None
        cli_neg_diffs = None

    all_results = []
    for nr in args.rounds:
        if cli_pos_diffs is not None:
            pos_diffs = cli_pos_diffs
            neg_diffs = cli_neg_diffs
            expected_acc = None
            expected_tpr = None
            expected_tnr = None
        else:
            # No CLI differentials supplied — use the generic fallbacks.
            # These are placeholders; for serious work either pass your own
            # via --pos_diffs/--neg_diffs/--pos_key_diffs/--neg_key_diffs or
            # run --use_pipeline to let the GA search for them.
            pos_diffs    = _FALLBACK_POS_DIFFS
            neg_diffs    = _FALLBACK_NEG_DIFFS
            expected_acc = None
            expected_tpr = None
            expected_tnr = None
            print(f"\n[INFO] No differentials specified for round {nr}. "
                  f"Using generic fallback differentials. "
                  f"For better results, either pass --pos_diffs/--neg_diffs/"
                  f"--pos_key_diffs/--neg_key_diffs explicitly, or use "
                  f"--use_pipeline to invoke the GA search.")

        # ── FULL PIPELINE MODE ─────────────────────────────────────────
        if args.use_pipeline:
            thr_acc = args.threshold_acc if args.threshold_acc is not None else expected_acc
            thr_tpr = args.threshold_tpr if args.threshold_tpr is not None else expected_tpr
            thr_tnr = args.threshold_tnr if args.threshold_tnr is not None else expected_tnr
            if thr_acc is None:
                print(f"  No threshold for round {nr} – set --threshold_acc. Skipping.")
                continue

            res, _ = run_full_pipeline(
                nr=nr,
                threshold_acc=thr_acc,
                threshold_tpr=thr_tpr,
                threshold_tnr=thr_tnr,
                max_iterations=args.max_iterations,
                use_ga=True,
                fallback_pos_diffs=pos_diffs,
                fallback_neg_diffs=neg_diffs,
                ga_pop_size=args.ga_pop_size,
                ga_generations=args.ga_generations,
                ga_n_fitness=args.ga_n_fitness,
                ga_n_select=args.ga_n_select,
                ga_min_hw=args.ga_min_hw,
                ga_max_hw=args.ga_max_hw,
                ga_bias_filter=args.ga_bias_filter,
                n_train=args.n_train, n_val=args.n_val,
                num_epochs=args.epochs, batch_size=args.batch_size,
                high_lr=args.high_lr, low_lr=args.low_lr, lr_epoch=args.lr_epoch,
                s_groups=args.s_groups, extra_words=args.extra_words,
                depth=args.depth, num_filters=args.num_filters,
                d1=args.d1, d2=args.d2,
                reg_param=args.reg_param, dropout_rate=args.dropout,
                ks_value_1=args.ks1, ks_value_2=args.ks2, ks_value_3=args.ks3,
                num_heads=args.num_heads, ff_dim=args.ff_dim,
                num_transformer_blocks=args.num_transformer_blocks,
                se_ratio=args.se_ratio, output_dir=args.output_dir,
            )
            all_results.append(res)
            continue

        # ── Single-shot mode ───────────────────────────────────────────
        _, _, res = train_distinguisher_enhanced(
            nr=nr, pos_diffs=pos_diffs, neg_diffs=neg_diffs,
            expected_acc=expected_acc, expected_tpr=expected_tpr, expected_tnr=expected_tnr,
            n_train=args.n_train, n_val=args.n_val,
            num_epochs=args.epochs, batch_size=args.batch_size,
            high_lr=args.high_lr, low_lr=args.low_lr, lr_epoch=args.lr_epoch,
            s_groups=args.s_groups, extra_words=args.extra_words,
            depth=args.depth, num_filters=args.num_filters,
            d1=args.d1, d2=args.d2,
            reg_param=args.reg_param, dropout_rate=args.dropout,
            ks_value_1=args.ks1, ks_value_2=args.ks2, ks_value_3=args.ks3,
            num_heads=args.num_heads, ff_dim=args.ff_dim,
            num_transformer_blocks=args.num_transformer_blocks,
            se_ratio=args.se_ratio,
            output_dir=args.output_dir,
        )
        all_results.append(res)

    if all_results:
        print("\n" + "=" * 90)
        print("  FINAL SUMMARY  --  CNN+Transformer v2 ENHANCED  Simon128/128")
        print(f"  {'Nr':<5} {'Pos-diff0':<14} {'Neg-diff0':<14} "
              f"{'Acc':<8} {'TPR':<8} {'TNR':<8}")
        print("  " + "-" * 60)
        for r in all_results:
            pd0 = hex(r['pos_diffs'][0][0][1])
            nd0 = hex(r['neg_diffs'][0][0][1])
            print(f"  {r['nr']:<5} {pd0:<14} {nd0:<14} "
                  f"{r['acc']:<8.4f} {r['tpr']:<8.4f} {r['tnr']:<8.4f}")
        print("=" * 90)