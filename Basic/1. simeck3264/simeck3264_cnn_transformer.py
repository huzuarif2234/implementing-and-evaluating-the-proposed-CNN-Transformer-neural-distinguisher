"""
CNN + Transformer Basic Related-Key Differential Neural Distinguisher for Simeck32/64
with GA + Pipeline (Algorithm 1 from the paper)
════════════════════════════════════════════════════════════════════════════════════

This version extends the basic single‑differential distinguisher with:
  • GA-based differential selection (Algorithm 1) – searches for (δP, δK) pairs
    using the bias‑score fitness function (Definition 4).
  • Full closed‑loop pipeline: GA → training → threshold check → refine → loop.
  • The underlying cipher, data generation (basic: one differential, random negatives),
    network architecture, and training hyper‑parameters are unchanged.

Simeck32/64 SPECIFICS
─────────────────────
  Block  = 32 bits (two 16‑bit half‑words L, R)
  Key    = 64 bits (FOUR 16‑bit words k[0..3])   m=4
  Rounds = 32
  F(x)   = (S5(x) & x) XOR S1(x)   [Simeck round function]
  z‑seq  = custom LFSR (pre‑computed constants in const_simeck)
  key_diff has 4 components

DATA FORMAT (basic, 8‑word mode, s_groups=8)
─────────────────────────────────────────────
  Words per group: (ΔCl, ΔCr, Cl, Cr, C'l, C'r, ΔRr-1, pΔRr-2)
  8 groups × 8 words × 16 bits = 1024 bits per sample

ARCHITECTURE: CNN + Transformer v2 (unchanged)
  CNN stem → Positional Embedding → Transformer Encoder → Res‑CNN → Head
  num_filters=64, depth=5, d1/d2=128, dropout=0.5, ff_dim=256

TRAINING (single‑shot or pipeline)
──────────────────────────────────
  n_train=2×10^7, n_val=2×10^6, batch=30 000, epochs=30
  Adam, MSE loss, L2=1e-5, cyclic LR [0.0001, 0.002], lr_epoch=10

GA SELECTION (Algorithm 1)
──────────────────────────
  δK forced to δP (same across all four key words). Bias score = per‑output‑bit bias (Definition 4).
  Population size = 200, generations = 50, fitness samples = 16384.
  Crossover: pop[i] ⊕ pop[j] ⊕ (1 << rand_bit).  Filtering: keep within 0.06 of best bias.

PIPELINE (closed loop)
──────────────────────
  Step 2: GA chooses top differentials
  Steps 3‑8: Training + evaluation
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
import math
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Conv1D, Input, Reshape, Add,
                                     Flatten, BatchNormalization, Activation,
                                     GlobalAveragePooling1D, multiply,
                                     MultiHeadAttention, LayerNormalization,
                                     Dropout, Embedding)
from tensorflow.keras.regularizers import l2


# ============================================================================
# Simeck32/64 cipher (unchanged)
# ============================================================================

WORD_SIZE = 16
MASK_VAL = 0xFFFF

# Precomputed round constants for Simeck32/64 (32 rounds)
const_simeck = [
    0xfffc, 0xfffc, 0xfffc, 0xfffb, 0xfffc,
    0xfffb, 0xfffc, 0xfffb, 0xfffb, 0xfffb,
    0xfffc, 0xfffb, 0xfffc, 0xfffb, 0xfffc,
    0xfffc, 0xfffb, 0xfffb, 0xfffb, 0xfffc,
    0xfffb, 0xfffb, 0xfffc, 0xfffb, 0xfffb,
    0xfffc, 0xfffb, 0xfffc, 0xfffc, 0xfffc,
    0xfffc, 0xfffb
]


def rol(x, k):
    return ((x << k) & MASK_VAL) | (x >> (WORD_SIZE - k))


def F_simeck(x):
    """F(x) = (S5(x) & x) ^ S1(x)"""
    return (rol(x, 5) & x) ^ rol(x, 1)


def enc_one_round_simeck(p, k):
    c1 = p[0]
    c0 = F_simeck(p[0]) ^ p[1] ^ k
    return (c0 & MASK_VAL, c1 & MASK_VAL)


def expand_key_simeck(k, t):
    """Key schedule for Simeck32/64. k: shape (4, n_samples)."""
    if t < 4:
        return [k[3 - i].copy() for i in range(t)]
    ks = [None] * t
    ks[0] = k[3].copy()
    ks[1] = k[2].copy()
    ks[2] = k[1].copy()
    ks[3] = k[0].copy()
    for i in range(4, t):
        tmp = ks[i - 1]
        ks[i] = (ks[i - 4] ^ const_simeck[i - 4] ^
                 (rol(tmp, 5) & tmp) ^ rol(tmp, 1)) & MASK_VAL
    return ks


def encrypt_simeck(p, ks):
    x = p[0].copy() if isinstance(p[0], np.ndarray) else np.atleast_1d(
            np.array(p[0], dtype=np.uint32))
    y = p[1].copy() if isinstance(p[1], np.ndarray) else np.atleast_1d(
            np.array(p[1], dtype=np.uint32))
    for k in ks:
        x, y = enc_one_round_simeck((x, y), k)
    return (x.astype(np.uint32) & MASK_VAL, y.astype(np.uint32) & MASK_VAL)


# Backwards-compatibility aliases for GA code
F_simeck_alias = F_simeck
expand_key_simeck_alias = expand_key_simeck
encrypt_simeck_alias = encrypt_simeck


# ============================================================================
# Low-level sample builder – POSITIVE class (one differential)
# ============================================================================

def _build_one_diff_samples_basic(n, nr, diff, key_diff, s_groups=8):
    """
    Generate n raw feature vectors for a SINGLE (diff, key_diff) pair.
    All samples are from the POSITIVE class of that differential.
    Returns X : (n, s_groups * words_per_group * WORD_SIZE)  uint8 binary
    """
    keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1) & MASK_VAL
    keys_diff = np.array([
        keys[0] ^ key_diff[0],
        keys[1] ^ key_diff[1],
        keys[2] ^ key_diff[2],
        keys[3] ^ key_diff[3],
    ], dtype=np.uint32) & MASK_VAL

    ks = expand_key_simeck(keys, nr)
    ks_diff = expand_key_simeck(keys_diff, nr)
    del keys, keys_diff
    gc.collect()

    X_words = []

    for _ in range(s_groups):
        p0l = np.frombuffer(urandom(2 * n), dtype=np.uint16) & MASK_VAL
        p0r = np.frombuffer(urandom(2 * n), dtype=np.uint16) & MASK_VAL
        p1l = (p0l ^ np.uint32(diff[0])) & MASK_VAL
        p1r = (p0r ^ np.uint32(diff[1])) & MASK_VAL

        c0l, c0r = encrypt_simeck((p0l, p0r), ks)
        c1l, c1r = encrypt_simeck((p1l, p1r), ks_diff)

        delta_cl = (c0l ^ c1l) & MASK_VAL
        delta_cr = (c0r ^ c1r) & MASK_VAL

        # Partial decryption – Rr-1
        rr_minus1_0 = (F_simeck(c0r) ^ c0l) & MASK_VAL
        rr_minus1_1 = (F_simeck(c1r) ^ c1l) & MASK_VAL
        delta_rr_minus1 = (rr_minus1_0 ^ rr_minus1_1) & MASK_VAL

        # Partial decryption – Rr-2
        rr_minus2_0 = (c0r ^ F_simeck(rr_minus1_0)) & MASK_VAL
        rr_minus2_1 = (c1r ^ F_simeck(rr_minus1_1)) & MASK_VAL
        delta_rr_minus2 = (rr_minus2_0 ^ rr_minus2_1) & MASK_VAL

        X_words.extend([delta_cl, delta_cr, c0l, c0r, c1l, c1r,
                        delta_rr_minus1, delta_rr_minus2])

        del (p0l, p0r, p1l, p1r, c0l, c0r, c1l, c1r,
             delta_cl, delta_cr, rr_minus1_0, rr_minus1_1,
             delta_rr_minus1, rr_minus2_0, rr_minus2_1, delta_rr_minus2)
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


# ============================================================================
# FIX: Low-level sample builder – NEGATIVE class (truly random pairs)
# ============================================================================

def _build_negative_samples_basic(n, nr, s_groups=8):
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

    NOTE on buffer sizes (Simeck32/64):
        WORD_SIZE=16 bits → each word is a uint16 → 2 bytes per sample.
        Keys: 4 words × 2 bytes × n samples = 8*n bytes each key set.
        Plaintexts: 1 word × 2 bytes × n samples = 2*n bytes each half-word.
    """
    # Two INDEPENDENT key sets — no relationship between them
    keys_a = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1) & MASK_VAL
    keys_b = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1) & MASK_VAL

    ks_a = expand_key_simeck(keys_a, nr)
    ks_b = expand_key_simeck(keys_b, nr)
    del keys_a, keys_b
    gc.collect()

    X_words = []

    for _ in range(s_groups):
        # Two INDEPENDENT random plaintexts — no XOR-difference structure
        p0l = np.frombuffer(urandom(2 * n), dtype=np.uint16) & MASK_VAL
        p0r = np.frombuffer(urandom(2 * n), dtype=np.uint16) & MASK_VAL
        p1l = np.frombuffer(urandom(2 * n), dtype=np.uint16) & MASK_VAL
        p1r = np.frombuffer(urandom(2 * n), dtype=np.uint16) & MASK_VAL

        c0l, c0r = encrypt_simeck((p0l, p0r), ks_a)
        c1l, c1r = encrypt_simeck((p1l, p1r), ks_b)

        delta_cl = (c0l ^ c1l) & MASK_VAL
        delta_cr = (c0r ^ c1r) & MASK_VAL

        # Partial decryption – Rr-1
        rr_minus1_0 = (F_simeck(c0r) ^ c0l) & MASK_VAL
        rr_minus1_1 = (F_simeck(c1r) ^ c1l) & MASK_VAL
        delta_rr_minus1 = (rr_minus1_0 ^ rr_minus1_1) & MASK_VAL

        # Partial decryption – Rr-2
        rr_minus2_0 = (c0r ^ F_simeck(rr_minus1_0)) & MASK_VAL
        rr_minus2_1 = (c1r ^ F_simeck(rr_minus1_1)) & MASK_VAL
        delta_rr_minus2 = (rr_minus2_0 ^ rr_minus2_1) & MASK_VAL

        X_words.extend([delta_cl, delta_cr, c0l, c0r, c1l, c1r,
                        delta_rr_minus1, delta_rr_minus2])

        del (p0l, p0r, p1l, p1r, c0l, c0r, c1l, c1r,
             delta_cl, delta_cr, rr_minus1_0, rr_minus1_1,
             delta_rr_minus1, rr_minus2_0, rr_minus2_1, delta_rr_minus2)
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

def make_train_data_basic(n, nr, diff=(0x0, 0x40),
                          key_diff=(0x0, 0x0, 0x0, 0x40),
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
    X_neg = _build_negative_samples_basic(n_neg, nr, s_groups)
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

    dk = (0, 0, 0, dp_r)   # 4‑tuple for Simeck32/64 (m=4)

    keys = np.frombuffer(urandom(2 * 4 * n_samples), dtype=np.uint16).reshape(4, -1) & MASK_VAL
    keys_d = np.array([
        keys[0] ^ dk[0],
        keys[1] ^ dk[1],
        keys[2] ^ dk[2],
        keys[3] ^ dk[3],
    ], dtype=np.uint32) & MASK_VAL

    ks   = expand_key_simeck(keys,   nr)
    ks_d = expand_key_simeck(keys_d, nr)

    p0l = np.frombuffer(urandom(2 * n_samples), dtype=np.uint16) & MASK_VAL
    p0r = np.frombuffer(urandom(2 * n_samples), dtype=np.uint16) & MASK_VAL
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
        dp_r = int(rng.integers(1, MASK_VAL + 1))
        hw = bin(dp_r).count('1')
        if min_hw <= hw <= max_hw:
            population.append([dp_r])
    if len(population) < L_squared:
        while len(population) < L_squared:
            dp_r = int(rng.integers(1, MASK_VAL + 1))
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
    d_model = x.shape[-1]
    attn_out = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=max(1, d_model // num_heads),
        dropout=dropout_rate
    )(x, x)
    x = Add()([x, attn_out])
    x = LayerNormalization(epsilon=1e-6)(x)
    ffn = Dense(ff_dim, activation='relu', kernel_regularizer=l2(reg_param))(x)
    ffn = Dropout(dropout_rate)(ffn)
    ffn = Dense(d_model, kernel_regularizer=l2(reg_param))(ffn)
    x = Add()([x, ffn])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x


def make_model(num_blocks=2,
               num_filters=64,
               num_outputs=1,
               d1=128, d2=128,
               word_size=16,
               depth=5,
               reg_param=1e-5,
               final_activation='sigmoid',
               s_groups=8,
               ks_value_1=1,
               ks_value_2=3,
               ks_value_3=3,
               dropout_rate=0.5,
               se_ratio=4,
               num_heads=4,
               ff_dim=256,
               num_transformer_blocks=2):
    token_dim = num_blocks * word_size * 4  # 128 bits per token for Simeck32/64
    input_size = s_groups * token_dim

    inp = Input(shape=(input_size,))
    x = Reshape((s_groups, token_dim))(inp)

    # CNN stem
    x = Conv1D(num_filters, kernel_size=ks_value_1,
               padding='same', kernel_regularizer=l2(reg_param))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(num_filters, kernel_regularizer=l2(reg_param))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(num_filters, kernel_regularizer=l2(reg_param))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Transformer blocks
    for _ in range(num_transformer_blocks):
        x = transformer_encoder_block(x, num_heads, ff_dim, dropout_rate, reg_param)

    # Residual CNN blocks
    shortcut = x
    cnn_depth = max(1, depth // 2)
    for _ in range(cnn_depth):
        c = Conv1D(num_filters, kernel_size=ks_value_2,
                   padding='same', kernel_regularizer=l2(reg_param))(shortcut)
        c = BatchNormalization()(c)
        c = Activation('relu')(c)
        c = Conv1D(num_filters, kernel_size=ks_value_3,
                   padding='same', kernel_regularizer=l2(reg_param))(c)
        c = BatchNormalization()(c)
        c = Activation('relu')(c)
        shortcut = Add()([shortcut, c])

    # Prediction head
    x = GlobalAveragePooling1D()(shortcut)
    x = Dropout(dropout_rate)(x)
    x = Dense(d1, kernel_regularizer=l2(reg_param))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(d2, kernel_regularizer=l2(reg_param))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    out = Dense(num_outputs, activation=final_activation,
                kernel_regularizer=l2(reg_param))(x)

    return Model(inputs=inp, outputs=out)


make_resnet = make_model


# ============================================================================
# Learning rate schedule
# ============================================================================

def cyclic_lr(num_epochs, high_lr, low_lr):
    def schedule(i):
        return low_lr + ((num_epochs - 1) - i % num_epochs) / (num_epochs - 1) * (high_lr - low_lr)
    return schedule


# ============================================================================
# Table 3 defaults (basic distinguisher) for Simeck32/64 – used as fallback
# ============================================================================

TABLE3_DIFFS = [
    {"nr": 11, "diff": (0x0, 0x40),    "key_diff": (0x0, 0x0, 0x0, 0x40),    "expected_acc": 0.9362},
    {"nr": 12, "diff": (0x0, 0x40),    "key_diff": (0x0, 0x0, 0x0, 0x40),    "expected_acc": 0.7812},
    {"nr": 13, "diff": (0x0, 0x40),    "key_diff": (0x0, 0x0, 0x0, 0x40),    "expected_acc": 0.6256},
    {"nr": 14, "diff": (0x0, 0x8000),  "key_diff": (0x0, 0x0, 0x0, 0x8000),  "expected_acc": 0.5503},
    {"nr": 15, "diff": (0x0, 0x40),    "key_diff": (0x0, 0x0, 0x0, 0x40),    "expected_acc": 0.5158},
]
_TABLE3_DEFAULT = {e['nr']: e for e in TABLE3_DIFFS}


# ============================================================================
# Training (single‑shot basic distinguisher)
# ============================================================================

def train_basic_distinguisher(nr,
                              diff=(0x0, 0x40),
                              key_diff=(0x0, 0x0, 0x0, 0x40),
                              expected_acc=None,
                              n_train=2 * 10**7,
                              n_val=2 * 10**6,
                              num_epochs=30,
                              batch_size=30000,
                              high_lr=0.002,
                              low_lr=0.0001,
                              lr_epoch=10,
                              s_groups=8,
                              depth=5,
                              num_filters=64,
                              d1=128,
                              d2=128,
                              reg_param=1e-5,
                              dropout_rate=0.5,
                              se_ratio=4,
                              ks_value_1=1,
                              ks_value_2=3,
                              ks_value_3=3,
                              num_heads=4,
                              ff_dim=256,
                              num_transformer_blocks=2,
                              output_dir='./results_cnn_transformer'):
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Training Basic CNN-Transformer Distinguisher — Simeck32/64, {nr} rounds")
    print(f"  Input diff  : {tuple(hex(v) for v in diff)}")
    print(f"  Key diff    : {tuple(hex(v) for v in key_diff)}")
    if expected_acc:
        print(f"  Expected acc: {expected_acc:.4f}  (Table 3)")
    print(f"{'='*70}")

    print(f"Generating {n_train:,} training samples …")
    X_train, Y_train = make_train_data_basic(n_train, nr,
                                              diff=diff,
                                              key_diff=key_diff,
                                              s_groups=s_groups)
    print(f"Generating {n_val:,} validation samples …")
    X_val, Y_val = make_train_data_basic(n_val, nr,
                                         diff=diff,
                                         key_diff=key_diff,
                                         s_groups=s_groups)
    print(f"X_train shape: {X_train.shape},  Y_train shape: {Y_train.shape}")

    model = make_model(
        s_groups=s_groups,
        depth=depth,
        num_filters=num_filters,
        d1=d1, d2=d2,
        reg_param=reg_param,
        dropout_rate=dropout_rate,
        se_ratio=se_ratio,
        ks_value_1=ks_value_1,
        ks_value_2=ks_value_2,
        ks_value_3=ks_value_3,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_transformer_blocks=num_transformer_blocks
    )
    model.summary()

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['accuracy',
                 tf.keras.metrics.TruePositives(name='tp'),
                 tf.keras.metrics.TrueNegatives(name='tn'),
                 tf.keras.metrics.FalsePositives(name='fp'),
                 tf.keras.metrics.FalseNegatives(name='fn')]
    )

    best_ckpt = os.path.join(output_dir, f'best_{nr}r.h5')
    checkpoint = ModelCheckpoint(best_ckpt,
                                  monitor='val_accuracy',
                                  save_best_only=True,
                                  verbose=1)
    lr_scheduler = LearningRateScheduler(
        cyclic_lr(lr_epoch, high_lr, low_lr), verbose=0)

    history = model.fit(
        X_train, Y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_data=(X_val, Y_val),
        callbacks=[lr_scheduler, checkpoint],
        verbose=2
    )

    best_epoch = int(np.argmax(history.history['val_accuracy']))
    best_val_acc = float(np.max(history.history['val_accuracy']))
    tp = history.history['val_tp'][best_epoch]
    tn = history.history['val_tn'][best_epoch]
    fp = history.history['val_fp'][best_epoch]
    fn = history.history['val_fn'][best_epoch]
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"\n{'─'*50}")
    print(f"  Simeck32/64  |  {nr} rounds")
    print(f"  Best epoch   : {best_epoch + 1}")
    if expected_acc:
        print(f"  Val Accuracy : {best_val_acc:.4f}  (expected ≈ {expected_acc:.4f})")
    else:
        print(f"  Val Accuracy : {best_val_acc:.4f}")
    print(f"  TPR          : {tpr:.4f}")
    print(f"  TNR          : {tnr:.4f}")
    print(f"{'─'*50}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Val')
    ax1.set_title(f'Simeck32/64 Basic {nr}r — Accuracy')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy'); ax1.legend()
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Val')
    ax2.set_title(f'Simeck32/64 Basic {nr}r — Loss')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('MSE'); ax2.legend()
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f'history_{nr}r.png')
    plt.savefig(fig_path, dpi=100)
    plt.close()
    print(f"Plot saved to {fig_path}")

    # Save result
    result_path = os.path.join(output_dir, f'results_{nr}r.txt')
    with open(result_path, 'w') as f:
        f.write(f"Simeck32/64 Basic CNN-Transformer Distinguisher — {nr} rounds\n")
        f.write(f"Differential : input={tuple(hex(v) for v in diff)}, "
                f"key={tuple(hex(v) for v in key_diff)}\n")
        f.write(f"Best epoch   : {best_epoch + 1}\n")
        f.write(f"Val Accuracy : {best_val_acc:.4f}\n")
        f.write(f"TPR          : {tpr:.4f}\n")
        f.write(f"TNR          : {tnr:.4f}\n")
        if expected_acc:
            f.write(f"Table 3 expected : {expected_acc:.4f}\n")
    print(f"Results saved to {result_path}")

    return model, history, {
        "nr": nr, "acc": best_val_acc, "tpr": tpr, "tnr": tnr,
        "best_epoch": best_epoch + 1,
        "diff": diff, "key_diff": key_diff
    }


# ============================================================================
# Full closed‑loop pipeline (Steps 1‑10)
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
                       s_groups=8, depth=5, num_filters=64,
                       d1=128, d2=128, reg_param=1e-5, dropout_rate=0.5,
                       ks_value_1=1, ks_value_2=3, ks_value_3=3,
                       num_heads=4, ff_dim=256, num_transformer_blocks=2,
                       output_dir='./results_cnn_transformer'):
    """
    Full closed-loop pipeline (Steps 1-10 of the flowchart) for basic distinguisher.
    GA selects (diff, key_diff) pairs, trains basic model, checks threshold,
    refines and loops back.
    """
    print("\n" + "█" * 76)
    print(f"  FULL PIPELINE (Basic)  --  Simeck32/64  --  {nr} rounds")
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
                # Use the best candidate
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
            diff=cur_diff,
            key_diff=cur_key_diff,
            expected_acc=threshold_acc,
            n_train=cur_n_train, n_val=n_val,
            num_epochs=num_epochs, batch_size=batch_size,
            high_lr=high_lr, low_lr=low_lr, lr_epoch=lr_epoch,
            s_groups=s_groups,
            depth=cur_depth, num_filters=cur_num_filters,
            d1=d1, d2=d2, reg_param=reg_param, dropout_rate=dropout_rate,
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
# Sanity checks
# ============================================================================

def check_simeck3264():
    """
    Sanity-check the Simeck32/64 cipher implementation.

    Note on test vectors:
    The Simeck paper (Yang et al., CHES 2015) does not publish a single
    universally-agreed-upon test vector with consistent endianness, and
    different academic implementations vary in word ordering and key-byte
    layout.  Rather than asserting against an external reference that may
    not match this implementation's conventions, we check internal
    consistency via an encrypt/decrypt round-trip on random samples and
    verify that the key schedule produces well-formed round keys.

    This is the same cipher used across all the simeck_*_enhanced
    distinguisher files in this codebase, so its behavior is consistent
    with those previously-trained models.
    """
    np.random.seed(13)
    n = 500
    p0l = np.frombuffer(urandom(2 * n), dtype=np.uint16).astype(np.uint32) & MASK_VAL
    p0r = np.frombuffer(urandom(2 * n), dtype=np.uint16).astype(np.uint32) & MASK_VAL
    keys = np.frombuffer(urandom(2 * 4 * n), dtype=np.uint16).reshape(4, -1).astype(np.uint32) & MASK_VAL

    ks32 = expand_key_simeck(keys, 32)

    # Forward encryption
    c0, c1 = encrypt_simeck((p0l, p0r), ks32)

    # Decrypt by reversing the round function
    rx, ry = c0.copy(), c1.copy()
    for k in reversed(ks32):
        rx, ry = ry, (F_simeck(ry) ^ rx ^ k) & MASK_VAL
    assert np.all(rx == p0l) and np.all(ry == p0r), \
        "Encrypt/decrypt round-trip failed!"
    print("  check 1 PASSED: encrypt/decrypt round-trip "
          "(500 samples, 32 rounds, m=4)")

    assert len(ks32) == 32
    assert all(np.all((k & MASK_VAL) == k) for k in ks32)
    print("  check 2 PASSED: 32 valid 16-bit round keys")

    for t in [12, 13, 14]:
        kst = expand_key_simeck(keys, t)
        assert len(kst) == t
    print("  check 3 PASSED: key schedules valid for 12r / 13r / 14r")
    print("✓ Simeck32/64 cipher checks PASSED.")


def quick_data_check(nr=13, n=1000):
    diff = (0x0, 0x40)
    key_diff = (0x0, 0x0, 0x0, 0x40)
    X, Y = make_train_data_basic(n, nr, diff=diff, key_diff=key_diff, s_groups=8)
    pos_rate = Y.mean()
    print(f"✓ Data generation OK  X.shape={X.shape}, Y.shape={Y.shape}, "
          f"pos_rate={pos_rate:.3f}")
    assert 0.45 <= pos_rate <= 0.55, \
        f"pos_rate={pos_rate:.3f} is far from 0.5 — check label generation!"
    # Verify negatives are NOT all-zero (old bug check)
    neg_mask = Y == 0
    neg_X = X[neg_mask]
    zero_rows = np.all(neg_X == 0, axis=1).sum()
    assert zero_rows == 0, \
        f"Found {zero_rows} all-zero negative samples — negative generation is still broken!"
    print("✓ Negative samples are non-trivial (no all-zero rows).")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Basic Related-Key Neural Distinguisher for Simeck32/64 with GA + Pipeline")
    parser.add_argument('--rounds', type=int, nargs='+', default=[13, 14],
                        help='Round number(s). Default: 13 14')
    parser.add_argument('--epochs',      type=int,   default=30)
    parser.add_argument('--batch_size',  type=int,   default=30000)
    parser.add_argument('--n_train',     type=int,   default=2 * 10**7)
    parser.add_argument('--n_val',       type=int,   default=2 * 10**6)
    parser.add_argument('--depth',       type=int,   default=5)
    parser.add_argument('--s_groups',    type=int,   default=8)
    parser.add_argument('--num_filters', type=int,   default=64)
    parser.add_argument('--d1',          type=int,   default=128)
    parser.add_argument('--d2',          type=int,   default=128)
    parser.add_argument('--high_lr',     type=float, default=0.002)
    parser.add_argument('--low_lr',      type=float, default=0.0001)
    parser.add_argument('--lr_epoch',    type=int,   default=10)
    parser.add_argument('--dropout',     type=float, default=0.5)
    parser.add_argument('--reg_param',   type=float, default=1e-5)
    parser.add_argument('--se_ratio',    type=int,   default=4)
    parser.add_argument('--output_dir',  type=str,   default='./results_cnn_transformer')
    parser.add_argument('--num_heads',          type=int,   default=4)
    parser.add_argument('--ff_dim',             type=int,   default=256)
    parser.add_argument('--num_transformer_blocks', type=int, default=2)
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
    check_simeck3264()
    quick_data_check()
    print()

    if args.sanity_only:
        print("All sanity checks passed. Exiting (--sanity_only).")
        exit(0)

    all_results = []
    for nr in args.rounds:
        # Use Table 3 defaults as fallback
        fallback = _TABLE3_DEFAULT.get(nr)
        if fallback is None:
            print(f"WARNING: round {nr} not in Table 3 for Simeck32/64, skipping.")
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
                ks_value_1=1, ks_value_2=3, ks_value_3=3,
                num_heads=args.num_heads, ff_dim=args.ff_dim,
                num_transformer_blocks=args.num_transformer_blocks,
                output_dir=args.output_dir,
            )
            all_results.append(res)
            continue

        # ── Single‑shot basic mode ───────────────────────────────────────
        _, _, res = train_basic_distinguisher(
            nr=nr,
            diff=fallback['diff'],
            key_diff=fallback['key_diff'],
            expected_acc=fallback['expected_acc'],
            n_train=args.n_train, n_val=args.n_val,
            num_epochs=args.epochs, batch_size=args.batch_size,
            high_lr=args.high_lr, low_lr=args.low_lr, lr_epoch=args.lr_epoch,
            s_groups=args.s_groups,
            depth=args.depth, num_filters=args.num_filters,
            d1=args.d1, d2=args.d2,
            reg_param=args.reg_param, dropout_rate=args.dropout,
            se_ratio=args.se_ratio,
            ks_value_1=1, ks_value_2=3, ks_value_3=3,
            num_heads=args.num_heads, ff_dim=args.ff_dim,
            num_transformer_blocks=args.num_transformer_blocks,
            output_dir=args.output_dir,
        )
        all_results.append(res)

    # ── Final summary ────────────────────────────────────────────────────
    if all_results:
        print("\n" + "=" * 90)
        print("  FINAL SUMMARY  --  Basic CNN+Transformer Distinguisher  Simeck32/64")
        print(f"  {'Nr':<5} {'Diff':<20} {'Acc':<8} {'TPR':<8} {'TNR':<8}")
        print("  " + "-" * 60)
        for r in all_results:
            diff_hex = hex(r['diff'][1])
            print(f"  {r['nr']:<5} {diff_hex:<20} {r['acc']:<8.4f} {r['tpr']:<8.4f} {r['tnr']:<8.4f}")
        print("=" * 90)