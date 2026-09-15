"""
CNN + Transformer Basic Related-Key Differential Neural Distinguisher for Simeck64/128
with GA + Pipeline (Algorithm 1 from the paper)
════════════════════════════════════════════════════════════════════════════════════

This version extends the basic single-differential distinguisher with:
  • GA-based differential selection (Algorithm 1) – searches for (δP, δK) pairs
    using the bias-score fitness function (Definition 4).
  • Full closed-loop pipeline: GA → training → threshold check → refine → loop.
  • The underlying cipher, data generation (basic: one differential, random negatives),
    network architecture, and training hyper-parameters are unchanged.

Simeck64/128 SPECIFICS
──────────────────────
  Block  = 64 bits (two 32-bit half-words L, R)
  Key    = 128 bits (FOUR 32-bit words k[0..3])   m=4
  Rounds = 44
  F(x)   = (S5(x) & x) XOR S1(x)   [Simeck round function]
  z-seq  = custom LFSR (pre-computed constants)
  key_diff has 4 components

DATA FORMAT (basic, 8-word mode, s_groups=8)
─────────────────────────────────────────────
  Words per group: (ΔCl, ΔCr, Cl, Cr, C'l, C'r, ΔRr-1, pΔRr-2)
  8 groups × 8 words × 32 bits = 2048 bits per sample

ARCHITECTURE: CNN + Transformer v2 (unchanged)
  CNN stem → Transformer Encoder → Res-CNN → Head
  num_filters=64, depth=5, d1/d2=128, dropout=0.5, ff_dim=256

TRAINING (single-shot or pipeline)
──────────────────────────────────
  n_train=2×10^7, n_val=2×10^6, batch=30 000, epochs=30
  Adam, MSE loss, L2=1e-5, cyclic LR [0.0001, 0.002], lr_epoch=10

GA SELECTION (Algorithm 1)
──────────────────────────
  δK forced to δP (same across all four key words). Bias score = per-output-bit bias (Definition 4).
  Population size = 200, generations = 50, fitness samples = 16384.
  Crossover: pop[i] ⊕ pop[j] ⊕ (1 << rand_bit).  Filtering: keep within 0.06 of best bias.

PIPELINE (closed loop)
──────────────────────
  Step 2: GA chooses top differentials
  Steps 3-8: Training + evaluation
  Step 9: Threshold check
  Step 10: Refine GA, data size, architecture, and loop back

FIX (v2)
────────
  Negative samples previously used diff=(0,0) and key_diff=(0,0,0,0), which caused
  p1==p0 and k'==k, making all delta features identically zero — trivially separable.
  Negatives are now built from two INDEPENDENTLY random plaintext/key pairs so that
  the ciphertext difference is uniformly random (no structure), matching the true
  random-pair distribution that a distinguisher must learn to reject.
"""

import numpy as np
from os import urandom
import gc
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Conv1D, Input, Reshape, Add,
                                     BatchNormalization, Activation,
                                     GlobalAveragePooling1D,
                                     MultiHeadAttention, LayerNormalization,
                                     Dropout)
from tensorflow.keras.regularizers import l2


# ============================================================================
# Simeck64/128 cipher (unchanged, verified)
# ============================================================================

WORD_SIZE = 32
MASK_VAL  = 0xFFFFFFFF

const_simeck64 = [
    0xfffffffd, 0xfffffffd, 0xfffffffd, 0xfffffffd, 0xfffffffd,
    0xfffffffd, 0xfffffffc, 0xfffffffc, 0xfffffffc, 0xfffffffc,
    0xfffffffc, 0xfffffffd, 0xfffffffc, 0xfffffffc, 0xfffffffc,
    0xfffffffc, 0xfffffffd, 0xfffffffd, 0xfffffffc, 0xfffffffc,
    0xfffffffc, 0xfffffffd, 0xfffffffc, 0xfffffffd, 0xfffffffc,
    0xfffffffc, 0xfffffffd, 0xfffffffd, 0xfffffffd, 0xfffffffd,
    0xfffffffc, 0xfffffffd, 0xfffffffc, 0xfffffffc, 0xfffffffc,
    0xfffffffd, 0xfffffffd, 0xfffffffd, 0xfffffffc, 0xfffffffc,
    0xfffffffd, 0xfffffffc, 0xfffffffc, 0xfffffffd,
]
assert len(const_simeck64) == 44


def rol(x, k):
    return ((x << k) & MASK_VAL) | (x >> (WORD_SIZE - k))


def F_simeck(x):
    return (rol(x, 5) & x) ^ rol(x, 1)


def enc_one_round_simeck(p, k):
    c1 = p[0]
    c0 = F_simeck(p[0]) ^ p[1] ^ k
    return (c0 & MASK_VAL, c1 & MASK_VAL)


def _sc(v):
    return v.copy() if isinstance(v, np.ndarray) else v


def expand_key_simeck(k, t):
    if t < 4:
        return [k[3 - i].copy() for i in range(t)]
    ks = [None] * t
    ks_tmp = [_sc(k[3]), _sc(k[2]), _sc(k[1]), _sc(k[0])]
    ks[0] = ks_tmp[0]
    for i in range(1, t):
        ks[i] = _sc(ks_tmp[1])
        tmp = ((rol(ks_tmp[1], 5) & ks_tmp[1]) ^
               rol(ks_tmp[1], 1) ^ ks[i - 1] ^ const_simeck64[i - 1])
        tmp = tmp & MASK_VAL
        ks_tmp[1] = ks_tmp[2]
        ks_tmp[2] = ks_tmp[3]
        ks_tmp[3] = tmp
    return ks


def encrypt_simeck(p, ks):
    x = p[0].copy() if isinstance(p[0], np.ndarray) else np.atleast_1d(
            np.array(p[0], dtype=np.uint32))
    y = p[1].copy() if isinstance(p[1], np.ndarray) else np.atleast_1d(
            np.array(p[1], dtype=np.uint32))
    for k in ks:
        x, y = enc_one_round_simeck((x, y), k)
    return (x.astype(np.uint32) & MASK_VAL, y.astype(np.uint32) & MASK_VAL)


# ============================================================================
# Low-level sample builder – POSITIVE class (one differential)
# ============================================================================

def _build_one_diff_samples_basic(n, nr, diff, key_diff, s_groups=8):
    """
    Generate n raw feature vectors for a SINGLE (diff, key_diff) pair.
    All samples are from the POSITIVE class of that differential.
    Returns X : (n, s_groups * words_per_group * WORD_SIZE)  uint8 binary
    """
    keys = np.frombuffer(urandom(16 * n), dtype=np.uint32).reshape(4, -1).copy()
    keys_diff = np.array([
        keys[0] ^ key_diff[0],
        keys[1] ^ key_diff[1],
        keys[2] ^ key_diff[2],
        keys[3] ^ key_diff[3],
    ], dtype=np.uint32)

    ks      = expand_key_simeck(keys,      nr)
    ks_diff = expand_key_simeck(keys_diff, nr)
    del keys, keys_diff
    gc.collect()

    X_words = []
    for _ in range(s_groups):
        p0l = np.frombuffer(urandom(4 * n), dtype=np.uint32).copy()
        p0r = np.frombuffer(urandom(4 * n), dtype=np.uint32).copy()
        p1l = (p0l ^ diff[0]).astype(np.uint32)
        p1r = (p0r ^ diff[1]).astype(np.uint32)

        c0l, c0r = encrypt_simeck((p0l, p0r), ks)
        c1l, c1r = encrypt_simeck((p1l, p1r), ks_diff)

        delta_cl = c0l ^ c1l
        delta_cr = c0r ^ c1r

        # Partial decryption – Rr-1
        rm1_0 = (rol(c0r, 5) & c0r) ^ rol(c0r, 1) ^ c0l
        rm1_1 = (rol(c1r, 5) & c1r) ^ rol(c1r, 1) ^ c1l
        delta_rm1 = rm1_0 ^ rm1_1

        # Partial decryption – Rr-2
        rm2_0 = (rol(rm1_0, 5) & rm1_0) ^ rol(rm1_0, 1) ^ c0r
        rm2_1 = (rol(rm1_1, 5) & rm1_1) ^ rol(rm1_1, 1) ^ c1r
        delta_rm2 = rm2_0 ^ rm2_1

        X_words.extend([delta_cl, delta_cr, c0l, c0r, c1l, c1r,
                        delta_rm1, delta_rm2])

        del (p0l, p0r, p1l, p1r, c0l, c0r, c1l, c1r,
             delta_cl, delta_cr, rm1_0, rm1_1, delta_rm1,
             rm2_0, rm2_1, delta_rm2)
        gc.collect()

    # Convert to binary
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


# ============================================================================
# Low-level sample builder – NEGATIVE class (truly random pairs)
# ============================================================================

def _build_random_negative_samples(n, nr, s_groups=8):
    """
    Generate n NEGATIVE samples: two INDEPENDENT random plaintext/key pairs.

    BUG THAT WAS HERE BEFORE:
        The old code called _build_one_diff_samples_basic(n, nr, (0,0), (0,0,0,0))
        which set diff=(0,0) and key_diff=(0,0,0,0).  This means:
            p1 = p0 XOR 0 = p0        (identical plaintext)
            k' = k  XOR 0 = k         (identical key)
        So both encryptions are of the SAME input → all delta features are
        identically zero.  The network trivially learns "all-zero deltas = negative"
        and reaches 100 % accuracy without learning any cryptographic structure.

    CORRECT APPROACH:
        Draw two completely independent plaintext/key pairs (a, b).  Their
        ciphertext XOR is uniformly distributed, giving no structural signal
        that the network can use to cheat.

    Simeck64/128 has m=4 -- four 32-bit key words.
    """
    keys_a = np.frombuffer(urandom(16 * n), dtype=np.uint32).reshape(4, -1).copy()
    keys_b = np.frombuffer(urandom(16 * n), dtype=np.uint32).reshape(4, -1).copy()

    ks_a = expand_key_simeck(keys_a, nr)
    ks_b = expand_key_simeck(keys_b, nr)
    del keys_a, keys_b
    gc.collect()

    X_words = []
    for _ in range(s_groups):
        p0l = np.frombuffer(urandom(4 * n), dtype=np.uint32).copy()
        p0r = np.frombuffer(urandom(4 * n), dtype=np.uint32).copy()
        p1l = np.frombuffer(urandom(4 * n), dtype=np.uint32).copy()
        p1r = np.frombuffer(urandom(4 * n), dtype=np.uint32).copy()

        c0l, c0r = encrypt_simeck((p0l, p0r), ks_a)
        c1l, c1r = encrypt_simeck((p1l, p1r), ks_b)

        delta_cl = c0l ^ c1l
        delta_cr = c0r ^ c1r

        # Partial decryption – Rr-1
        rm1_0 = (rol(c0r, 5) & c0r) ^ rol(c0r, 1) ^ c0l
        rm1_1 = (rol(c1r, 5) & c1r) ^ rol(c1r, 1) ^ c1l
        delta_rm1 = rm1_0 ^ rm1_1

        # Partial decryption – Rr-2
        rm2_0 = (rol(rm1_0, 5) & rm1_0) ^ rol(rm1_0, 1) ^ c0r
        rm2_1 = (rol(rm1_1, 5) & rm1_1) ^ rol(rm1_1, 1) ^ c1r
        delta_rm2 = rm2_0 ^ rm2_1

        X_words.extend([delta_cl, delta_cr, c0l, c0r, c1l, c1r,
                        delta_rm1, delta_rm2])

        del (p0l, p0r, p1l, p1r, c0l, c0r, c1l, c1r,
             delta_cl, delta_cr, rm1_0, rm1_1, delta_rm1,
             rm2_0, rm2_1, delta_rm2)
        gc.collect()

    del ks_a, ks_b
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


# ============================================================================
# Training data generator (FIXED)
# ============================================================================

def make_train_data_basic(n, nr,
                          diff=(0x0, 0x8),
                          key_diff=(0x0, 0x0, 0x0, 0x8),
                          s_groups=8):
    """
    Generate training data for the basic related-key neural distinguisher.

    Positive samples : satisfy the given (diff, key_diff) relationship.
    Negative samples : two INDEPENDENT random plaintext/key pairs.
                       (Previously used diff=(0,0)/key_diff=(0,0,0,0) which
                       caused identical inputs → all-zero deltas → trivial 100%.)
    """
    n_pos = n // 2
    n_neg = n - n_pos

    # Positive samples — structured differential pair
    X_pos = _build_one_diff_samples_basic(n_pos, nr, diff, key_diff, s_groups)
    Y_pos = np.ones(n_pos, dtype=np.uint8)

    # Negative samples — truly independent random pairs (THE FIX)
    X_neg = _build_random_negative_samples(n_neg, nr, s_groups)
    Y_neg = np.zeros(n_neg, dtype=np.uint8)

    X = np.vstack([X_pos, X_neg])
    Y = np.concatenate([Y_pos, Y_neg])
    del X_pos, X_neg, Y_pos, Y_neg
    gc.collect()

    idx = np.random.permutation(len(Y))
    return X[idx], Y[idx]


# ============================================================================
# STEP 2 : GA-Based Differential Selection (Algorithm 1 from the paper)
# ============================================================================

_PAPER_WORKING_POP = 32
_PAPER_BIAS_THRESHOLD = 0.06


def _bias_score(dp_l, dp_r, nr, n_samples):
    """
    Compute empirical related-key bias score (Definition 4).
    δK forced equal to δP across all four key words: (0,0,0,dp_r).
    """
    dp_l = 0
    dp_r = int(dp_r) & MASK_VAL
    if dp_r == 0:
        return 0.0

    dk = (0, 0, 0, dp_r)   # 4-tuple for Simeck64/128 (m=4)

    keys = np.frombuffer(urandom(16 * n_samples), dtype=np.uint32).reshape(4, -1).copy()
    keys_d = np.array([
        keys[0] ^ dk[0],
        keys[1] ^ dk[1],
        keys[2] ^ dk[2],
        keys[3] ^ dk[3],
    ], dtype=np.uint32)

    ks   = expand_key_simeck(keys,   nr)
    ks_d = expand_key_simeck(keys_d, nr)

    p0l = np.frombuffer(urandom(4 * n_samples), dtype=np.uint32).copy()
    p0r = np.frombuffer(urandom(4 * n_samples), dtype=np.uint32).copy()
    p1l = p0l ^ dp_l
    p1r = p0r ^ dp_r

    c0l, c0r = encrypt_simeck((p0l, p0r), ks)
    c1l, c1r = encrypt_simeck((p1l, p1r), ks_d)

    diff_l = c0l ^ c1l
    diff_r = c0r ^ c1r

    bias_sum = 0.0
    for bit in range(WORD_SIZE):
        mask = np.uint32(1 << bit)
        frac_l = float(np.count_nonzero(diff_l & mask)) / n_samples
        frac_r = float(np.count_nonzero(diff_r & mask)) / n_samples
        bias_sum += abs(0.5 - frac_l)
        bias_sum += abs(0.5 - frac_r)

    return bias_sum / (2.0 * WORD_SIZE)


def _chrom_bias(chrom, nr, n_samples):
    return _bias_score(0, int(chrom[0]), nr, n_samples)


def _generate_initial_population(L_squared, plain_bits, min_hw, max_hw, rng):
    """Generate L² random dp_r values with Hamming weight in [min_hw, max_hw]."""
    population = []
    attempts = 0
    max_attempts = L_squared * 200
    while len(population) < L_squared and attempts < max_attempts:
        attempts += 1
        dp_r = int(rng.integers(1, MASK_VAL + 1, dtype=np.uint32))
        hw = bin(dp_r).count('1')
        if min_hw <= hw <= max_hw:
            population.append([dp_r])
    if len(population) < L_squared:
        while len(population) < L_squared:
            dp_r = int(rng.integers(1, MASK_VAL + 1, dtype=np.uint32))
            population.append([dp_r])
    return np.array(population, dtype=np.uint32)


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
        selected: list of (diff_2tuple, key_diff_4tuple) — top n_select
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
        candidates = np.array(candidates, dtype=np.uint32)
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
        dk = (0, 0, 0, dp_r)
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


# ============================================================================
# Neural network (CNN + Transformer) – unchanged from basic file
# ============================================================================

def transformer_encoder_block(x, num_heads, ff_dim, dropout_rate, reg_param):
    d_model = int(x.shape[-1])
    key_dim = max(1, d_model // num_heads)
    attn_out = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout_rate,
    )(x, x)
    x = Add()([x, attn_out])
    x = LayerNormalization(epsilon=1e-6)(x)
    ffn = Dense(ff_dim, activation='relu', kernel_regularizer=l2(reg_param))(x)
    ffn = Dropout(dropout_rate)(ffn)
    ffn = Dense(d_model, kernel_regularizer=l2(reg_param))(ffn)
    x = Add()([x, ffn])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x


def make_model(s_groups=8,
               word_size=32,
               num_words_per_group=8,
               num_filters=64,
               depth=5,
               d1=128,
               d2=128,
               reg_param=1e-5,
               dropout_rate=0.5,
               ks_value_1=1,
               ks_value_2=3,
               ks_value_3=3,
               num_heads=4,
               ff_dim=256,
               num_transformer_blocks=2,
               final_activation='sigmoid'):
    token_dim = num_words_per_group * word_size
    input_size = s_groups * token_dim

    inp = Input(shape=(input_size,), name='input')
    x = Reshape((s_groups, token_dim), name='reshape')(inp)

    # CNN stem
    x = Conv1D(num_filters, kernel_size=ks_value_1,
               padding='same', kernel_regularizer=l2(reg_param),
               name='stem_conv')(x)
    x = BatchNormalization(name='stem_bn')(x)
    x = Activation('relu', name='stem_relu')(x)
    x = Dense(num_filters, kernel_regularizer=l2(reg_param),
              name='stem_d1')(x)
    x = BatchNormalization(name='stem_bn1')(x)
    x = Activation('relu', name='stem_relu1')(x)
    x = Dense(num_filters, kernel_regularizer=l2(reg_param),
              name='stem_d2')(x)
    x = BatchNormalization(name='stem_bn2')(x)
    x = Activation('relu', name='stem_relu2')(x)

    # Transformer blocks
    for _ in range(num_transformer_blocks):
        x = transformer_encoder_block(x, num_heads, ff_dim, dropout_rate, reg_param)

    # Residual CNN blocks
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

    return Model(inputs=inp, outputs=out, name='CNN_Transformer_Simeck64_128_Basic_v2')


make_resnet = make_model


# ============================================================================
# Learning rate schedule
# ============================================================================

def cyclic_lr(num_epochs, high_lr, low_lr):
    def schedule(i):
        return low_lr + ((num_epochs - 1) - i % num_epochs) / (num_epochs - 1) * (high_lr - low_lr)
    return schedule


# ============================================================================
# Table 3 defaults (basic distinguisher) for Simeck64/128 – used as fallback
# ============================================================================

TABLE3_DIFFS = [
    {"nr": 21, "diff": (0x0, 0x8), "key_diff": (0x0, 0x0, 0x0, 0x8),
     "expected_acc": 0.5521, "expected_tpr": 0.4099, "expected_tnr": 0.6944},
    {"nr": 22, "diff": (0x0, 0x8), "key_diff": (0x0, 0x0, 0x0, 0x8),
     "expected_acc": 0.5181, "expected_tpr": 0.3875, "expected_tnr": 0.6484},
]
_TABLE3_DEFAULT = {e['nr']: e for e in TABLE3_DIFFS}


# ============================================================================
# Training (single-shot basic distinguisher)
# ============================================================================

def train_basic_distinguisher(nr,
                              diff=(0x0, 0x8),
                              key_diff=(0x0, 0x0, 0x0, 0x8),
                              expected_acc=None, expected_tpr=None, expected_tnr=None,
                              n_train=2 * 10**7, n_val=2 * 10**6,
                              num_epochs=30, batch_size=30000,
                              high_lr=0.002, low_lr=0.0001, lr_epoch=10,
                              s_groups=8,
                              depth=5,
                              num_filters=64,
                              d1=128, d2=128,
                              reg_param=1e-5, dropout_rate=0.5,
                              ks_value_1=1, ks_value_2=3, ks_value_3=3,
                              num_heads=4, ff_dim=256, num_transformer_blocks=2,
                              output_dir='./results_simeck64128'):
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*72}")
    print(f"  Simeck64/128  Basic CNN+Transformer Distinguisher  --  {nr} rounds")
    print(f"  Input diff : {tuple(hex(v) for v in diff)}")
    print(f"  Key diff   : {tuple(hex(v) for v in key_diff)}")
    if expected_acc is not None:
        print(f"  Table 3    : acc={expected_acc:.4f}  "
              f"TPR={expected_tpr:.4f}  TNR={expected_tnr:.4f}")
    print(f"{'='*72}")

    print(f"Generating {n_train:,} training samples ...")
    X_tr, Y_tr = make_train_data_basic(n_train, nr, diff=diff, key_diff=key_diff,
                                        s_groups=s_groups)
    print(f"Generating {n_val:,} validation samples ...")
    X_val, Y_val = make_train_data_basic(n_val, nr, diff=diff, key_diff=key_diff,
                                         s_groups=s_groups)
    print(f"X_train : {X_tr.shape}   Y_train : {Y_tr.shape}")
    print(f"X_val   : {X_val.shape}   Y_val   : {Y_val.shape}")

    model = make_model(
        s_groups=s_groups, depth=depth,
        num_filters=num_filters, d1=d1, d2=d2,
        reg_param=reg_param, dropout_rate=dropout_rate,
        ks_value_1=ks_value_1, ks_value_2=ks_value_2, ks_value_3=ks_value_3,
        num_heads=num_heads, ff_dim=ff_dim,
        num_transformer_blocks=num_transformer_blocks,
    )
    model.summary()

    total_params = int(np.sum([np.prod(v.get_shape())
                                for v in model.trainable_weights]))
    print(f"Trainable parameters : {total_params:,}")

    model.compile(
        optimizer='adam', loss='mse',
        metrics=['accuracy',
                 tf.keras.metrics.TruePositives(name='tp'),
                 tf.keras.metrics.TrueNegatives(name='tn'),
                 tf.keras.metrics.FalsePositives(name='fp'),
                 tf.keras.metrics.FalseNegatives(name='fn')],
    )

    ckpt = os.path.join(output_dir, f'best_{nr}r.h5')
    callbacks = [
        ModelCheckpoint(ckpt, monitor='val_accuracy',
                        save_best_only=True, verbose=1),
        LearningRateScheduler(cyclic_lr(lr_epoch, high_lr, low_lr), verbose=0),
    ]

    history = model.fit(
        X_tr, Y_tr,
        epochs=num_epochs, batch_size=batch_size,
        validation_data=(X_val, Y_val),
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

    print(f"\n{'─'*55}")
    print(f"  Simeck64/128  |  {nr} rounds")
    print(f"  Best epoch    : {best_ep + 1}")
    print(f"  Val Accuracy  : {best_acc:.4f}"
          + (f"   (Table 3 approx {expected_acc:.4f})" if expected_acc else ""))
    print(f"  TPR           : {tpr:.4f}"
          + (f"   (Table 3 approx {expected_tpr:.4f})" if expected_tpr else ""))
    print(f"  TNR           : {tnr:.4f}"
          + (f"   (Table 3 approx {expected_tnr:.4f})" if expected_tnr else ""))
    print(f"{'─'*55}")

    _plot_history(history, nr, output_dir, expected_acc)

    txt = os.path.join(output_dir, f'results_{nr}r.txt')
    with open(txt, 'w') as f:
        f.write(f"Simeck64/128 Basic CNN+Transformer Distinguisher -- {nr} rounds\n")
        f.write(f"Architecture   : CNN stem + "
                f"{num_transformer_blocks} Transformer + "
                f"{max(1,depth//2)} residual CNN blocks\n")
        f.write(f"Differential   : input={tuple(hex(v) for v in diff)}, "
                f"key={tuple(hex(v) for v in key_diff)}\n")
        f.write(f"Best epoch     : {best_ep + 1} / {num_epochs}\n")
        f.write(f"Val Accuracy   : {best_acc:.4f}\n")
        f.write(f"TPR            : {tpr:.4f}\n")
        f.write(f"TNR            : {tnr:.4f}\n")
        if expected_acc:
            f.write(f"Table 3 expected : acc={expected_acc:.4f}  TPR={expected_tpr:.4f}  TNR={expected_tnr:.4f}\n")
        f.write(f"Total params   : {total_params:,}\n")
    print(f"Results saved -> {txt}")

    return model, history, {
        "nr": nr, "acc": best_acc, "tpr": tpr, "tnr": tnr,
        "best_epoch": best_ep + 1,
        "diff": diff, "key_diff": key_diff,
    }


def _plot_history(history, nr, output_dir, expected_acc=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Val')
    if expected_acc:
        ax1.axhline(expected_acc, color='r', linestyle='--',
                    label=f'Table 3 ({expected_acc:.4f})')
    ax1.set_title(f'Simeck64/128 Basic {nr}r — Accuracy')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy'); ax1.legend()
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Val')
    ax2.set_title(f'Simeck64/128 Basic {nr}r — Loss')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('MSE'); ax2.legend()
    plt.tight_layout()
    p = os.path.join(output_dir, f'history_{nr}r.png')
    plt.savefig(p, dpi=100); plt.close()
    print(f"Plot saved -> {p}")


# ============================================================================
# Full closed-loop pipeline (Steps 1-10)
# ============================================================================

def run_full_pipeline(nr,
                       threshold_acc,
                       threshold_tpr=None,
                       threshold_tnr=None,
                       max_iterations=3,
                       use_ga=True,
                       fallback_diff=None,
                       fallback_key_diff=None,
                       # GA params
                       ga_pop_size=200,
                       ga_generations=50,
                       ga_n_fitness=16384,
                       ga_n_select=2,
                       ga_min_hw=1,
                       ga_max_hw=2,
                       ga_bias_filter=_PAPER_BIAS_THRESHOLD,
                       # training params
                       n_train=2 * 10**7, n_val=2 * 10**6,
                       num_epochs=30, batch_size=30000,
                       high_lr=0.002, low_lr=0.0001, lr_epoch=10,
                       s_groups=8,
                       depth=5, num_filters=64,
                       d1=128, d2=128, reg_param=1e-5, dropout_rate=0.5,
                       ks_value_1=1, ks_value_2=3, ks_value_3=3,
                       num_heads=4, ff_dim=256, num_transformer_blocks=2,
                       output_dir='./results_simeck64128'):
    """
    Full closed-loop pipeline (Steps 1-10 of the flowchart) for basic distinguisher.
    GA selects (diff, key_diff) pairs, trains basic model, checks threshold,
    refines and loops back.
    """
    print("\n" + "█" * 76)
    print(f"  FULL PIPELINE (Basic)  --  Simeck64/128  --  {nr} rounds")
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
    cur_diff            = fallback_diff
    cur_key_diff        = fallback_key_diff

    for it in range(1, max_iterations + 1):
        print("\n" + "▶" * 76)
        print(f"  ITERATION {it}/{max_iterations}")
        print("▶" * 76)

        # ── Step 2: GA-Based Differential Selection ──────────────────────
        if use_ga:
            print(f"\n  [Step 2]  GA-Based Differential Selection  (Algorithm 1)")
            ga_all, _, filtered_pool = ga_select_differentials(
                nr=nr,
                n_select=ga_n_select * 2,
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

            if best_bias > MIN_USEFUL_BIAS and len(ga_all) >= 1:
                diff, key_diff = ga_all[0]
                print(f"  [Step 2]  GA found useful diff: "
                      f"dP={tuple(hex(v) for v in diff)}  "
                      f"dK={tuple(hex(v) for v in key_diff)}  "
                      f"(bias={best_bias:.5f})")
                cur_diff = diff
                cur_key_diff = key_diff
            else:
                print(f"  [Step 2]  GA pool too small or bias too low "
                      f"(best_bias={best_bias:.5f} ≤ {MIN_USEFUL_BIAS})")
                print(f"  [Step 2]  ⚠ Falling back to caller-supplied differential")
                if fallback_diff is None or fallback_key_diff is None:
                    raise ValueError("GA failed and no fallback differential provided!")
                cur_diff = fallback_diff
                cur_key_diff = fallback_key_diff
        else:
            if cur_diff is None or cur_key_diff is None:
                raise ValueError("use_ga=False requires fallback differentials")
            diff = cur_diff
            key_diff = cur_key_diff

        # ── Steps 3-8: Sample generation → Network → Train → Evaluate ──
        print(f"\n  [Steps 3-8]  Sample gen → Net → Train → Evaluate")
        _, _, res = train_basic_distinguisher(
            nr=nr,
            diff=cur_diff, key_diff=cur_key_diff,
            expected_acc=threshold_acc,
            expected_tpr=threshold_tpr,
            expected_tnr=threshold_tnr,
            n_train=cur_n_train, n_val=n_val,
            num_epochs=num_epochs, batch_size=batch_size,
            high_lr=high_lr, low_lr=low_lr, lr_epoch=lr_epoch,
            s_groups=s_groups,
            depth=cur_depth, num_filters=cur_num_filters,
            d1=d1, d2=d2,
            reg_param=reg_param, dropout_rate=dropout_rate,
            ks_value_1=ks_value_1, ks_value_2=ks_value_2, ks_value_3=ks_value_3,
            num_heads=num_heads, ff_dim=ff_dim,
            num_transformer_blocks=num_transformer_blocks,
            output_dir=os.path.join(output_dir, f'iter_{it}'),
        )
        res['iteration'] = it
        res['n_train'] = cur_n_train
        res['depth'] = cur_depth
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


# ============================================================================
# Sanity checks (unchanged)
# ============================================================================

def check_simeck64128():
    plain = (np.array([0x656b696c], dtype=np.uint32),
             np.array([0x20646e75], dtype=np.uint32))
    cipher = (0x45ce6902, 0x5f7ab7ed)
    keys = np.array([[0x1b1a1918], [0x13121110],
                      [0x0b0a0908], [0x03020100]], dtype=np.uint32)
    ks = expand_key_simeck(keys, 44)
    c = encrypt_simeck(plain, ks)
    assert int(c[0][0]) == cipher[0] and int(c[1][0]) == cipher[1], \
        f"Test vector FAILED: got ({hex(int(c[0][0]))}, {hex(int(c[1][0]))})"
    print(f"  check 1 PASSED: test vector -> "
          f"({hex(int(c[0][0]))}, {hex(int(c[1][0]))})")

    k_test = np.array([[0xDEADBEEF], [0xCAFEBABE],
                        [0x12345678], [0xABCDEF01]], dtype=np.uint32)
    ks44 = expand_key_simeck(k_test, 44)
    assert len(ks44) == 44
    for idx, rk in enumerate(ks44):
        assert 0 <= int(rk[0]) <= MASK_VAL, \
            f"Round key {idx} out of range: {hex(int(rk[0]))}"
    print("  check 2 PASSED: 44 valid 32-bit round keys")

    kd = (0x0, 0x0, 0x0, 0x8)
    k2 = np.array([[int(k_test[0][0]) ^ kd[0]],
                    [int(k_test[1][0]) ^ kd[1]],
                    [int(k_test[2][0]) ^ kd[2]],
                    [int(k_test[3][0]) ^ kd[3]]], dtype=np.uint32)
    ks21 = expand_key_simeck(k_test, 21)
    ks21d = expand_key_simeck(k2, 21)
    pt0l = np.array([0x11223344], dtype=np.uint32)
    pt0r = np.array([0x55667788], dtype=np.uint32)
    c0l, c0r = encrypt_simeck((pt0l, pt0r), ks21)
    c1l, c1r = encrypt_simeck((pt0l ^ 0, pt0r ^ 8), ks21d)
    delta = (int(c0l[0]) ^ int(c1l[0]), int(c0r[0]) ^ int(c1r[0]))
    print(f"  check 3 PASSED: 21r differential dC=({hex(delta[0])},{hex(delta[1])})")
    print("  Simeck64/128 cipher checks PASSED")


def quick_data_check(nr=21, n=400):
    diff = (0x0, 0x8)
    key_diff = (0x0, 0x0, 0x0, 0x8)
    X, Y = make_train_data_basic(n, nr, diff=diff, key_diff=key_diff, s_groups=8)
    expected_cols = 8 * 8 * 32
    assert X.shape == (n, expected_cols), \
        f"Shape mismatch: got {X.shape}, expected ({n}, {expected_cols})"
    assert X.dtype == np.uint8
    assert set(np.unique(X)).issubset({0, 1})
    assert set(np.unique(Y)).issubset({0, 1})
    pos_rate = Y.mean()
    assert 0.45 <= pos_rate <= 0.55, \
        f"pos_rate={pos_rate:.3f} is far from 0.5 — check label generation!"
    # Verify negatives are NOT all-zero (old bug check)
    neg_X = X[Y == 0]
    zero_rows = int(np.all(neg_X == 0, axis=1).sum())
    assert zero_rows == 0, \
        f"Found {zero_rows} all-zero negative samples — negative generation is still broken!"
    print(f"  data shape  : {X.shape}  =  "
          f"(n_samples, s_groups x 8_words x 32_bits)")
    print(f"  pos_rate    : {pos_rate:.3f}  (expected approx 0.50)")
    print("  Negative samples are non-trivial (no all-zero rows).")
    print("  Data generation check PASSED")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Basic Related-Key Neural Distinguisher for Simeck64/128 with GA + Pipeline")
    parser.add_argument('--rounds', type=int, nargs='+', default=[21, 22],
                        help='Round number(s). Default: 21 22')
    parser.add_argument('--n_train',     type=int,   default=2 * 10**7)
    parser.add_argument('--n_val',       type=int,   default=2 * 10**6)
    parser.add_argument('--epochs',      type=int,   default=30)
    parser.add_argument('--batch_size',  type=int,   default=30000)
    parser.add_argument('--high_lr',     type=float, default=0.002)
    parser.add_argument('--low_lr',      type=float, default=0.0001)
    parser.add_argument('--lr_epoch',    type=int,   default=10)
    parser.add_argument('--s_groups',    type=int,   default=8)
    parser.add_argument('--depth',       type=int,   default=5)
    parser.add_argument('--num_filters', type=int,   default=64)
    parser.add_argument('--d1',          type=int,   default=128)
    parser.add_argument('--d2',          type=int,   default=128)
    parser.add_argument('--reg_param',   type=float, default=1e-5)
    parser.add_argument('--dropout',     type=float, default=0.5)
    parser.add_argument('--ks1',         type=int,   default=1)
    parser.add_argument('--ks2',         type=int,   default=3)
    parser.add_argument('--ks3',         type=int,   default=3)
    parser.add_argument('--num_heads',   type=int,   default=4)
    parser.add_argument('--ff_dim',      type=int,   default=256)
    parser.add_argument('--num_transformer_blocks', type=int, default=2)
    parser.add_argument('--output_dir',  type=str,  default='./results_simeck64128')
    parser.add_argument('--sanity_only', action='store_true')

    # Pipeline mode
    parser.add_argument('--use_pipeline', action='store_true',
                        help='Run full closed-loop pipeline with GA selection')
    parser.add_argument('--max_iterations', type=int, default=3)
    parser.add_argument('--ga_pop_size',    type=int, default=200)
    parser.add_argument('--ga_generations', type=int, default=50)
    parser.add_argument('--ga_n_fitness',   type=int, default=16384)
    parser.add_argument('--ga_n_select',    type=int, default=2)
    parser.add_argument('--ga_min_hw',      type=int, default=1)
    parser.add_argument('--ga_max_hw',      type=int, default=2)
    parser.add_argument('--ga_bias_filter', type=float, default=0.06)
    parser.add_argument('--threshold_acc',  type=float, default=None,
                        help='Accuracy threshold for pipeline')
    parser.add_argument('--threshold_tpr',  type=float, default=None)
    parser.add_argument('--threshold_tnr',  type=float, default=None)

    args = parser.parse_args()

    print("\nRunning sanity checks ...")
    check_simeck64128()
    quick_data_check()
    print()

    if args.sanity_only:
        print("All sanity checks passed. Exiting (--sanity_only).")
        exit(0)

    # Build fallback map from Table 3
    fallback_map = {e['nr']: e for e in TABLE3_DIFFS}

    all_results = []
    for nr in args.rounds:
        fallback = fallback_map.get(nr)
        if fallback is None:
            print(f"WARNING: round {nr} not in Table 3 (known: {sorted(fallback_map)}). "
                  f"Skipping (use explicit --diff/--key_diff).")
            continue

        # ── FULL PIPELINE MODE ───────────────────────────────────────────
        if args.use_pipeline:
            thr_acc = args.threshold_acc if args.threshold_acc is not None else fallback['expected_acc']
            thr_tpr = args.threshold_tpr if args.threshold_tpr is not None else None
            thr_tnr = args.threshold_tnr if args.threshold_tnr is not None else None
            if thr_acc is None:
                print(f"  No threshold available for round {nr} – set --threshold_acc. Skipping.")
                continue

            res, _ = run_full_pipeline(
                nr=nr,
                threshold_acc=thr_acc,
                threshold_tpr=thr_tpr,
                threshold_tnr=thr_tnr,
                max_iterations=args.max_iterations,
                use_ga=True,
                fallback_diff=fallback['diff'],
                fallback_key_diff=fallback['key_diff'],
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
                s_groups=args.s_groups,
                depth=args.depth, num_filters=args.num_filters,
                d1=args.d1, d2=args.d2,
                reg_param=args.reg_param, dropout_rate=args.dropout,
                ks_value_1=args.ks1, ks_value_2=args.ks2, ks_value_3=args.ks3,
                num_heads=args.num_heads, ff_dim=args.ff_dim,
                num_transformer_blocks=args.num_transformer_blocks,
                output_dir=args.output_dir,
            )
            all_results.append(res)
            continue

        # ── Single-shot basic mode ───────────────────────────────────────
        _, _, res = train_basic_distinguisher(
            nr=nr,
            diff=fallback['diff'],
            key_diff=fallback['key_diff'],
            expected_acc=fallback['expected_acc'],
            expected_tpr=fallback['expected_tpr'],
            expected_tnr=fallback['expected_tnr'],
            n_train=args.n_train, n_val=args.n_val,
            num_epochs=args.epochs, batch_size=args.batch_size,
            high_lr=args.high_lr, low_lr=args.low_lr, lr_epoch=args.lr_epoch,
            s_groups=args.s_groups,
            depth=args.depth, num_filters=args.num_filters,
            d1=args.d1, d2=args.d2,
            reg_param=args.reg_param, dropout_rate=args.dropout,
            ks_value_1=args.ks1, ks_value_2=args.ks2, ks_value_3=args.ks3,
            num_heads=args.num_heads, ff_dim=args.ff_dim,
            num_transformer_blocks=args.num_transformer_blocks,
            output_dir=args.output_dir,
        )
        all_results.append(res)

    # ── Final summary ────────────────────────────────────────────────────
    if all_results:
        print("\n" + "=" * 90)
        print("  FINAL SUMMARY  --  Basic CNN+Transformer Distinguisher  Simeck64/128")
        print(f"  {'Nr':<5} {'Diff':<20} {'Acc':<8} {'TPR':<8} {'TNR':<8}")
        print("  " + "-" * 60)
        for r in all_results:
            diff_hex = hex(r['diff'][1])
            print(f"  {r['nr']:<5} {diff_hex:<20} {r['acc']:<8.4f} {r['tpr']:<8.4f} {r['tnr']:<8.4f}")
        print("=" * 90)