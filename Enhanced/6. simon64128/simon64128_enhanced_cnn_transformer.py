"""
CNN + Transformer  Enhanced Related-Key Differential Neural Distinguisher
for Simon64/128  --  MULTI-DIFFERENTIAL (POLYHEDRAL) VERSION
═══════════════════════════════════════════════════════════════════════════
Paper : "A Multi-Differential Approach to Enhance Related-Key Neural
         Distinguishers"  (Xue Yuan & Qichun Wang)

TARGET  ──  Table 6  (enhanced distinguishers for Simon64/128)
  Round  Input / Key Differences                         Acc    TPR    TNR
  ─────  ──────────────────────────────────────────────  ─────  ─────  ─────
  26r    pos: (0x0,0x80000000) & (0x0,0x40000000)        0.999  0.999  0.999
         neg: (0x0,0x20)       & (0x0,0x40)
  27r    pos: (0x0,0x80000000) & (0x0,0x40000000)        0.755  0.759  0.751
         neg: (0x0,0x20)       & (0x0,0x40)

  (key_diff has 4 components for Simon64/128, m=4)

KEY DIFFERENCE VS BASIC VERSION
────────────────────────────────
Basic version (Table 4):
  • single differential  (diff, key_diff)
  • label: 1 = encrypted with (diff,key_diff), 0 = random pair

Enhanced version (Table 6, this file):
  • FOUR differentials:  dp0/dk0, dp1/dk1  → positive samples
                         dp2/dk2, dp3/dk3  → negative samples
  • label: 1 = sample satisfies one of (dp0,dk0) or (dp1,dk1)
           0 = sample satisfies one of (dp2,dk2) or (dp3,dk3)
  • Eliminates positive/negative confusion that exists when neg = random
  • Gives the network 2× the useful feature signal per sample

SIMON64/128 SPECIFICS
─────────────────────
  Block  = 64 bits  (two 32-bit half-words L, R)
  Key    = 128 bits  (FOUR 32-bit words k[0..3])   m=4
  Rounds = 44
  F(x)   = (S1(x) AND S8(x)) XOR S2(x)
  key_diff has 4 components

DATA FORMAT  (10-word mode, extra_words=2)
──────────────────────────────────────────
  Words per group: (dCl, dCr, Cl, Cr, C'l, C'r, dRr-1, dRr-2, dRr-3, dRr-4)
  8 groups × 10 words × 32 bits = 2560 bits per sample

ARCHITECTURE  (CNN + Transformer v2)
────────────────────────────────────────────────────────────────
  CNN stem → Positional Embedding → 3× Transformer → 5× Res-CNN+SENet → Head
  num_filters=128, depth=10, d1/d2=256, dropout=0.3, ff_dim=512

TRAINING
────────
  n_train=2×10^7, n_val=2×10^6, batch=30 000, epochs=40
  Adam, MSE loss, L2=1e-5, cyclic LR  [0.0001, 0.002]  period=40
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


# ═══════════════════════════════════════════════════════════════════════════
# Simon64/128 cipher (correct implementation with hardcoded constants)
# ═══════════════════════════════════════════════════════════════════════════

WORD_SIZE = 32
MASK_VAL  = 0xFFFFFFFF

# Hardcoded constants – taken from the correct Simon64/128 reference implementation
# (these are the exact constants used in simon64128_rkdiff.py and pass the test vector)
CONST_SIMON64_128 = [
    0xfffffffd, 0xfffffffd, 0xfffffffc, 0xfffffffd, 0xfffffffd, 0xfffffffc,
    0xfffffffd, 0xfffffffd, 0xfffffffd, 0xfffffffc, 0xfffffffd, 0xfffffffc,
    0xfffffffd, 0xfffffffd, 0xfffffffc, 0xfffffffc, 0xfffffffc, 0xfffffffd,
    0xfffffffd, 0xfffffffc, 0xfffffffc, 0xfffffffd, 0xfffffffc, 0xfffffffd,
    0xfffffffd, 0xfffffffd, 0xfffffffd, 0xfffffffc, 0xfffffffc, 0xfffffffc,
    0xfffffffc, 0xfffffffc, 0xfffffffc, 0xfffffffd, 0xfffffffc, 0xfffffffc,
    0xfffffffd, 0xfffffffc, 0xfffffffc, 0xfffffffc
]
assert len(CONST_SIMON64_128) == 40   # Simon64/128 needs exactly 40 constants for 44 rounds


def _rand_words(n):
    """Cryptographically random uint64 array of length n, masked to 32 bits."""
    raw = np.frombuffer(urandom(4 * n), dtype=np.uint32).astype(np.uint64)
    return raw & MASK_VAL


def rol(x, k):
    return ((x << k) & MASK_VAL) | (x >> (WORD_SIZE - k))


def ror(x, k):
    return (x >> k) | ((x << (WORD_SIZE - k)) & MASK_VAL)


def F_simon64(x):
    """Simon64 round function: F(x) = (S1(x) AND S8(x)) XOR S2(x)."""
    return (rol(x, 1) & rol(x, 8)) ^ rol(x, 2)


def enc_one_round_simon64(p, k):
    c1 = p[0]
    c0 = (F_simon64(p[0]) ^ p[1] ^ k) & MASK_VAL
    return (c0, c1)


def expand_key_simon64128(k, t):
    """
    Correct key schedule for Simon64/128 (matches official specification).
    k: shape (4, n), dtype uint64 (values masked to 32 bits).
    Returns list of t round-key arrays, each shape (n,).
    """
    if t < 4:
        return [k[3 - i].copy() if isinstance(k[3 - i], np.ndarray) else k[3 - i]
                for i in range(t)]
    ks = [None] * t
    ks[0] = k[3].copy() if isinstance(k[3], np.ndarray) else k[3]
    ks[1] = k[2].copy() if isinstance(k[2], np.ndarray) else k[2]
    ks[2] = k[1].copy() if isinstance(k[1], np.ndarray) else k[1]
    ks[3] = k[0].copy() if isinstance(k[0], np.ndarray) else k[0]
    for i in range(t - 4):
        # CORRECT: tmp = ror(ks[i+3], 3) ^ ks[i+1]   (not ks[i+2])
        tmp = ror(ks[i + 3], 3) ^ ks[i + 1]
        ks[i + 4] = (CONST_SIMON64_128[i] ^ ks[i] ^ tmp ^ ror(tmp, 1)) & MASK_VAL
    return ks


def encrypt_simon64128(p, ks):
    x = p[0].copy() if isinstance(p[0], np.ndarray) else np.atleast_1d(
            np.array(p[0], dtype=np.uint64))
    y = p[1].copy() if isinstance(p[1], np.ndarray) else np.atleast_1d(
            np.array(p[1], dtype=np.uint64))
    for k in ks:
        x, y = enc_one_round_simon64((x, y), k)
    return (x & MASK_VAL, y & MASK_VAL)


# Backwards-compatibility aliases for GA code (keeps names generic)
F_simeck = F_simon64
expand_key_simeck = expand_key_simon64128
encrypt_simeck = encrypt_simon64128


# ═══════════════════════════════════════════════════════════════════════════
# Low-level sample builder (one differential, one label class)
# ═══════════════════════════════════════════════════════════════════════════

def _build_one_diff_samples(n, nr, diff, key_diff, s_groups=8, extra_words=2):
    """
    Generate n raw feature vectors for a SINGLE (diff, key_diff) pair.
    All samples are from the POSITIVE class of that differential.

    Returns X : (n, s_groups * words_per_group * WORD_SIZE)  uint8 binary
    """
    keys = np.stack([_rand_words(n) for _ in range(4)], axis=0)
    keys_diff = np.array([
        keys[0] ^ key_diff[0],
        keys[1] ^ key_diff[1],
        keys[2] ^ key_diff[2],
        keys[3] ^ key_diff[3],
    ], dtype=np.uint64) & MASK_VAL

    ks      = expand_key_simeck(keys,      nr)
    ks_diff = expand_key_simeck(keys_diff, nr)
    del keys, keys_diff
    gc.collect()

    X_words = []

    for _ in range(s_groups):
        p0l = _rand_words(n)
        p0r = _rand_words(n)
        p1l = (p0l ^ np.uint64(diff[0])) & MASK_VAL
        p1r = (p0r ^ np.uint64(diff[1])) & MASK_VAL

        c0l, c0r = encrypt_simeck((p0l, p0r), ks)
        c1l, c1r = encrypt_simeck((p1l, p1r), ks_diff)

        delta_cl = (c0l ^ c1l) & MASK_VAL
        delta_cr = (c0r ^ c1r) & MASK_VAL

        rL0_1 = (F_simeck(c0r) ^ c0l) & MASK_VAL
        rL1_1 = (F_simeck(c1r) ^ c1l) & MASK_VAL
        d_rL1 = (rL0_1 ^ rL1_1) & MASK_VAL

        rL0_2 = (c0r ^ F_simeck(rL0_1)) & MASK_VAL
        rL1_2 = (c1r ^ F_simeck(rL1_1)) & MASK_VAL
        d_rL2 = (rL0_2 ^ rL1_2) & MASK_VAL

        X_words.extend([delta_cl, delta_cr, c0l, c0r, c1l, c1r, d_rL1, d_rL2])

        if extra_words >= 2:
            rL0_3 = (rL0_1 ^ F_simeck(rL0_2)) & MASK_VAL
            rL1_3 = (rL1_1 ^ F_simeck(rL1_2)) & MASK_VAL
            d_rL3 = (rL0_3 ^ rL1_3) & MASK_VAL
            rL0_4 = (rL0_2 ^ F_simeck(rL0_3)) & MASK_VAL
            rL1_4 = (rL1_2 ^ F_simeck(rL1_3)) & MASK_VAL
            d_rL4 = (rL0_4 ^ rL1_4) & MASK_VAL
            X_words.extend([d_rL3, d_rL4])

        del (p0l, p0r, p1l, p1r, c0l, c0r, c1l, c1r,
             delta_cl, delta_cr, rL0_1, rL1_1, d_rL1, rL0_2, rL1_2, d_rL2)
        gc.collect()

    del ks, ks_diff
    gc.collect()

    M  = len(X_words)
    WS = WORD_SIZE
    Z  = np.zeros((M * WS, n), dtype=np.uint8)
    for i in range(M * WS):
        wi      = i // WS
        bit_pos = WS - (i % WS) - 1
        Z[i]    = (X_words[wi] >> bit_pos) & 1
    del X_words
    gc.collect()
    return Z.T


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 : GA-Based Differential Selection (Algorithm 1 from the paper)
# ═══════════════════════════════════════════════════════════════════════════
#
# Implementation exactly follows the enhanced simon3264 version, adapted for
# 32‑bit words and the correct Simon64/128 cipher.
#
# KEY DESIGN DECISIONS matching the paper exactly:
#
# 1. CHROMOSOME ENCODING
#    Each chromosome encodes only the PLAINTEXT difference δP as two
#    half-words [dp_l, dp_r].  The KEY difference δK is ALWAYS set equal to
#    δP (replicated across all four key words):
#        dk = (dp_l, dp_r, dp_l, dp_r)   [Simon64/128, m=4]
#    This enforces the paper's central hypothesis (Conjecture 1 / Section 3):
#    "effective differential combinations are characterised by equal input
#    and key differentials."  Algorithm 1 line 3: key_population ← plain_population.
#
# 2. FITNESS FUNCTION  — per-bit BIAS SCORE (Definition 4)
#    b̃_t(δP, δK) = (1/n) Σ_{j=0}^{n-1} |0.5 − (1/t) Σ_i [(E_Ki(Xi) ⊕
#                   E_{Ki⊕δK}(Xi⊕δP))_j]|
#    This is the sum of per-output-bit biases.
#
# 3. POPULATION SIZE & TRUNCATION
#    The paper keeps exactly 32 chromosomes at all times (current_population
#    ← first 32 elements).  `pop_size` controls the initial L² pool; after
#    the first sort the working set is always 32.
#
# 4. CROSSOVER + MUTATION (Algorithm 1, lines 11-12)
#    candidate = pop[i] ⊕ pop[j] ⊕ (m << rand(0, n-1))
#    where m=1 (a single set bit) and n = WORD_SIZE (32 for Simon64/128).
#    This simultaneously recombines two parents AND flips one random bit.
#
# 5. OUTPUT FILTERING  (Section 3, paragraph after Table 2)
#    After 50 iterations keep all chromosomes whose bias score is within
#    0.06 of the best score.
#
# ═══════════════════════════════════════════════════════════════════════════

_PAPER_WORKING_POP = 32
_PAPER_BIAS_THRESHOLD = 0.06


def _bias_score(dp_l, dp_r, nr, n_samples):
    """
    Compute the empirical related-key bias score b̃_t(δP, δK) per Definition 4.

    δK is forced equal to δP across all four key words (paper's constraint).
    Score = (1/WORD_SIZE) * Σ_j |0.5 - (fraction of samples where output bit j
                                          differs between the two encryptions)|

    Parameters
    ----------
    dp_l, dp_r : int   — left and right half-words of the plaintext difference δP
    nr         : int   — number of encryption rounds
    n_samples  : int   — number of random (P, K) pairs to sample (≡ t in paper)

    Returns
    -------
    float  — bias score in [0, 0.5]
    """
    # Paper (Tables 3 & 5, Section 3): dp_l is always 0x0; only dp_r carries
    # the difference.  Key diff format: (0, 0, 0, dp_r).
    dp_l = 0
    dp_r = int(dp_r) & MASK_VAL

    # Paper key difference: delta_K = delta_P expressed as (0, 0, 0, dp_r)
    # matching the (t2, t1, t0, k0) Simon key-word order.
    dk = (0, 0, 0, dp_r)

    # Degenerate zero-difference -- skip
    if dp_r == 0:
        return 0.0

    # Sample random keys and plaintexts
    keys = np.stack([_rand_words(n_samples) for _ in range(4)], axis=0)
    keys_d = np.array([keys[i] ^ dk[i] for i in range(4)],
                      dtype=np.uint64) & MASK_VAL

    ks   = expand_key_simeck(keys,   nr)
    ks_d = expand_key_simeck(keys_d, nr)

    p0l = _rand_words(n_samples)
    p0r = _rand_words(n_samples)
    p1l = (p0l ^ dp_l) & MASK_VAL
    p1r = (p0r ^ dp_r) & MASK_VAL

    c0l, c0r = encrypt_simeck((p0l, p0r), ks)
    c1l, c1r = encrypt_simeck((p1l, p1r), ks_d)

    # XOR of output pairs  — shape (n_samples,) for each half-word
    diff_l = (c0l ^ c1l) & MASK_VAL   # left  output difference
    diff_r = (c0r ^ c1r) & MASK_VAL   # right output difference

    # Per-bit bias: |0.5 - fraction_of_samples_where_bit_j_is_1|
    bias_sum = 0.0
    for bit in range(WORD_SIZE):
        mask = np.uint64(1 << bit)
        frac_l = float(np.count_nonzero(diff_l & mask)) / n_samples
        frac_r = float(np.count_nonzero(diff_r & mask)) / n_samples
        bias_sum += abs(0.5 - frac_l)
        bias_sum += abs(0.5 - frac_r)

    # Normalise by total number of output bits (2 × WORD_SIZE for a 64-bit block)
    return bias_sum / (2.0 * WORD_SIZE)


def _chrom_bias(chrom, nr, n_samples):
    """Wrapper: evaluate bias score for a chromosome [dp_r] (dp_l fixed to 0)."""
    return _bias_score(0, int(chrom[0]), nr, n_samples)


def _generate_initial_population(L_squared, plain_bits, min_hw, max_hw, rng):
    """
    Algorithm 1 lines 1-4: generate L² random chromosomes with Hamming weight
    in [min_hw, max_hw].  Each chromosome is a 2-word plaintext difference
    [dp_l, dp_r] (key difference is always derived as δK = δP).

    Parameters
    ----------
    L_squared  : int  — size of the initial population pool
    plain_bits : int  — bit-width of dp_r (WORD_SIZE for the cipher variant)
    min_hw     : int  — minimum Hamming weight of the difference
    max_hw     : int  — maximum Hamming weight of the difference
    rng        : numpy Generator

    Returns
    -------
    population : ndarray, shape (L_squared, 2), dtype uint32
                 Rows are [dp_l, dp_r] chromosomes.
    """
    # Paper: dp_l is always 0; only dp_r (a single WORD_SIZE-bit value) is
    # evolved.  Chromosomes are therefore 1-word: [dp_r].
    population = []
    attempts   = 0
    max_attempts = L_squared * 200

    while len(population) < L_squared and attempts < max_attempts:
        attempts += 1
        dp_r = int(rng.integers(1, MASK_VAL + 1))
        hw   = bin(dp_r).count('1')
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

    Finds effective (δP, δK) pairs for Simon64/128 over `nr` rounds.
    δK is always kept equal to δP (paper's key constraint).

    Parameters
    ----------
    nr            : int   — number of encryption rounds
    n_select      : int   — how many differentials to return (from filtered pool)
    pop_size      : int   — L²: size of the initial random population pool
    n_generations : int   — number of evolutionary iterations (paper: 50)
    n_fitness     : int   — samples used to evaluate the bias score (paper: t)
    min_hw        : int   — minimum Hamming weight of initial differences (paper: 1)
    max_hw        : int   — maximum Hamming weight of initial differences (paper: 2)
    bias_filter   : float — keep candidates within this margin of the best
                            bias score (paper: 0.06)
    seed          : int or None
    verbose       : bool

    Returns
    -------
    selected      : list of (diff_2tuple, key_diff_4tuple) — top `n_select`
                    from the 0.06-filtered pool, ordered by bias score
    best_history  : list of float — best bias score per generation
    filtered_pool : list of (diff_2tuple, key_diff_4tuple, bias_score) —
                    full set of candidates within the 0.06 threshold
    """
    rng = np.random.default_rng(seed)

    if verbose:
        print(f"\n  [GA / Algorithm 1]  nr={nr}  pop={pop_size}  "
              f"gens={n_generations}  t={n_fitness}  HW=[{min_hw},{max_hw}]")
        print(f"  Constraint: δK ≡ δP  (paper Section 3)")

    # ── Algorithm 1, lines 1-5 ───────────────────────────────────────────
    plain_population = _generate_initial_population(
        pop_size, WORD_SIZE, min_hw, max_hw, rng)

    starting_bias = np.array(
        [_chrom_bias(c, nr, n_fitness) for c in plain_population])

    order = np.argsort(-starting_bias)
    current_population = plain_population[order[:_PAPER_WORKING_POP]].copy()
    current_bias       = starting_bias[order[:_PAPER_WORKING_POP]].copy()

    if verbose:
        print(f"  Initial pool: {pop_size} → kept top {_PAPER_WORKING_POP}  "
              f"(best bias = {float(current_bias.max()):.5f})")

    best_history = [float(current_bias.max())]

    # ── Algorithm 1, lines 7-14: main evolutionary loop ─────────────────
    for gen in range(n_generations):
        n_pop = len(current_population)

        # Build ALL pairwise candidates: pop[i] ⊕ pop[j] ⊕ (1 << rand_bit)
        candidates = []
        for i in range(n_pop):
            for j in range(i + 1, n_pop):
                rand_bit = int(rng.integers(0, WORD_SIZE))
                cand_r = (
                    int(current_population[i][0])
                    ^ int(current_population[j][0])
                    ^ (1 << rand_bit)
                ) & MASK_VAL
                if cand_r != 0:
                    candidates.append([cand_r])

        candidates = np.array(candidates, dtype=np.uint32)

        cand_bias = np.array(
            [_chrom_bias(c, nr, n_fitness) for c in candidates])

        order = np.argsort(-cand_bias)
        current_population = candidates[order[:_PAPER_WORKING_POP]].copy()
        current_bias       = cand_bias[order[:_PAPER_WORKING_POP]].copy()

        gen_best = float(current_bias.max())
        best_history.append(gen_best)

        if verbose:
            print(f"  [GA]  iter {gen+1:>2}/{n_generations}   "
                  f"best bias = {gen_best:.5f}   "
                  f"mean bias = {float(current_bias.mean()):.5f}   "
                  f"candidates = {len(candidates)}")

    # ── Output filtering (Section 3, 0.06-close rule) ────────────────────
    best_bias = float(current_bias.max())
    threshold = best_bias - bias_filter

    filtered_pool = []
    seen = set()
    for i in range(len(current_population)):
        c  = current_population[i]
        bs = float(current_bias[i])
        if bs < threshold:
            continue
        dp_r = int(c[0]) & MASK_VAL
        if dp_r == 0:
            continue
        if dp_r in seen:
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


# ═══════════════════════════════════════════════════════════════════════════
# Enhanced multi-differential data generator  (POLYHEDRAL METHOD)
# ═══════════════════════════════════════════════════════════════════════════

def make_train_data_enhanced(n, nr,
                              pos_diffs,
                              neg_diffs,
                              s_groups=8,
                              extra_words=2):
    """
    Build the ENHANCED (polyhedral) training set for Simon64/128.

    Parameters
    ──────────
    n          : total number of samples (half positive, half negative)
    nr         : number of rounds
    pos_diffs  : list of (diff_tuple, key_diff_tuple) for POSITIVE class
    neg_diffs  : list of (diff_tuple, key_diff_tuple) for NEGATIVE class
    s_groups   : number of ciphertext-pair groups per sample (default 8)
    extra_words: extra partial-decryption words (0 or 2, default 2)

    Returns
    ───────
    X : (n, s_groups * words_per_group * WORD_SIZE)  uint8
    Y : (n,)  uint8  labels  {0, 1}
    """
    t_pos = len(pos_diffs)
    t_neg = len(neg_diffs)
    assert t_pos >= 1 and t_neg >= 1, "Need at least 1 pos and 1 neg differential"

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

    Y_pos = np.ones(len(X_pos),  dtype=np.uint8)
    Y_neg = np.zeros(len(X_neg), dtype=np.uint8)
    X = np.vstack([X_pos, X_neg])
    Y = np.concatenate([Y_pos, Y_neg])
    del X_pos, X_neg, Y_pos, Y_neg
    gc.collect()

    idx = np.random.permutation(len(Y))
    return X[idx], Y[idx]


# ═══════════════════════════════════════════════════════════════════════════
# Neural network  (CNN + Transformer v2 — identical to base file)
# ═══════════════════════════════════════════════════════════════════════════

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
    d_model  = int(x.shape[-1])
    key_dim  = max(1, d_model // num_heads)
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
    x   = Add(name=f'{name}_ffn_add')([x, ffn])
    x   = LayerNormalization(epsilon=1e-6, name=f'{name}_ln2')(x)
    return x


def make_model(s_groups=8,
               words_per_group=10,
               word_size=32,
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
    """
    CNN + Transformer v2.
    Default: 8 groups × 10 words × 32 bits = 2560 bits per sample.
    depth=10 → cnn_depth = depth//2 = 5 Res-CNN+SENet blocks.
    """
    token_dim  = words_per_group * word_size
    input_size = s_groups * token_dim

    inp = Input(shape=(input_size,), name='input')
    x   = Reshape((s_groups, token_dim), name='reshape')(inp)

    # Module 1: CNN stem
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
    pos_emb = Embedding(s_groups, num_filters, name='pos_emb')(
        tf.range(s_groups)
    )
    pos_emb = tf.expand_dims(pos_emb, axis=0)
    x = Add(name='pos_add')([x, pos_emb])

    # Module 2a: Transformer Encoder
    for ti in range(num_transformer_blocks):
        x = transformer_encoder_block(
            x, num_heads=num_heads, ff_dim=ff_dim,
            dropout_rate=dropout_rate, reg_param=reg_param,
            name=f'tr{ti}',
        )

    # Module 2b: Residual CNN + SENet blocks
    shortcut  = x
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

    # Module 3: Prediction head
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
                 name='CNN_Transformer_Simon64_128_Enhanced_v2')


# ═══════════════════════════════════════════════════════════════════════════
# Learning rate schedule
# ═══════════════════════════════════════════════════════════════════════════

def cyclic_lr(n, high_lr, low_lr):
    def schedule(i):
        return low_lr + ((n - 1) - i % n) / (n - 1) * (high_lr - low_lr)
    return schedule


# ═══════════════════════════════════════════════════════════════════════════
# Table 6 targets  (enhanced distinguishers for Simon64/128)
# ═══════════════════════════════════════════════════════════════════════════

TABLE6_SIMON64128 = [
    {
        "nr"          : 26,
        "pos_diffs"   : [
            ((0x0, 0x80000000), (0x0, 0x0, 0x0, 0x80000000)),
            ((0x0, 0x40000000), (0x0, 0x0, 0x0, 0x40000000)),
        ],
        "neg_diffs"   : [
            ((0x0, 0x20),       (0x0, 0x0, 0x0, 0x20)),
            ((0x0, 0x40),       (0x0, 0x0, 0x0, 0x40)),
        ],
        "expected_acc": 0.999,
        "expected_tpr": 0.999,
        "expected_tnr": 0.999,
        "source"      : "Table6",
    },
    {
        "nr"          : 27,
        "pos_diffs"   : [
            ((0x0, 0x80000000), (0x0, 0x0, 0x0, 0x80000000)),
            ((0x0, 0x40000000), (0x0, 0x0, 0x0, 0x40000000)),
        ],
        "neg_diffs"   : [
            ((0x0, 0x20),       (0x0, 0x0, 0x0, 0x20)),
            ((0x0, 0x40),       (0x0, 0x0, 0x0, 0x40)),
        ],
        "expected_acc": 0.755,
        "expected_tpr": 0.759,
        "expected_tnr": 0.751,
        "source"      : "Table6",
    },
]

_TABLE6_DEFAULT = {e['nr']: e for e in TABLE6_SIMON64128}


# ═══════════════════════════════════════════════════════════════════════════
# Training (single-shot mode)
# ═══════════════════════════════════════════════════════════════════════════

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
        output_dir='./results_simon64128_enhanced',
):
    """
    Train the ENHANCED (polyhedral) distinguisher for Simon64/128.

    pos_diffs / neg_diffs are each a list of (diff_2tuple, key_diff_4tuple).
    """
    os.makedirs(output_dir, exist_ok=True)
    words_per_group = 8 + extra_words

    print(f"\n{'='*76}")
    print(f"  Simon64/128  ENHANCED CNN+Transformer v2  --  {nr} rounds")
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
    if expected_acc is not None:
        print(f"  target: acc={expected_acc:.4f}  "
              f"TPR={expected_tpr:.4f}  TNR={expected_tnr:.4f}")
    print(f"{'='*76}")

    print(f"\nGenerating {n_train:,} ENHANCED training samples ...")
    X_tr, Y_tr = make_train_data_enhanced(
        n_train, nr,
        pos_diffs=pos_diffs, neg_diffs=neg_diffs,
        s_groups=s_groups, extra_words=extra_words,
    )
    print(f"Generating {n_val:,} ENHANCED validation samples ...")
    X_val, Y_val = make_train_data_enhanced(
        n_val, nr,
        pos_diffs=pos_diffs, neg_diffs=neg_diffs,
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

    pos_tag  = f"p{hex(pos_diffs[0][0][1])}"
    neg_tag  = f"n{hex(neg_diffs[0][0][1])}"
    diff_tag = f"{pos_tag}_{neg_tag}"
    ckpt     = os.path.join(output_dir, f'best_{nr}r_{diff_tag}.h5')
    callbacks = [
        ModelCheckpoint(ckpt, monitor='val_accuracy',
                        save_best_only=True, verbose=1),
        LearningRateScheduler(cyclic_lr(lr_epoch, high_lr, low_lr), verbose=0),
    ]

    Y_tr_f32  = Y_tr.astype(np.float32)
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

    best_ep  = int(np.argmax(history.history['val_accuracy']))
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
        print(f"  Target:              acc={expected_acc:.4f}  "
              f"TPR={expected_tpr:.4f}  TNR={expected_tnr:.4f}")
        if best_acc >= expected_acc:
            print("  *** TABLE 6 TARGET REACHED ***")
        else:
            print(f"  Gap: {best_acc - expected_acc:+.4f}")
    print(f"{'─'*64}")

    _plot_history(history, nr, diff_tag, output_dir, expected_acc)

    txt = os.path.join(output_dir, f'results_{nr}r_{diff_tag}.txt')
    with open(txt, 'w') as f:
        f.write(f"Simon64/128  CNN+Transformer v2 ENHANCED  --  {nr} rounds\n")
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
            f.write(f"target_acc={expected_acc:.4f}  "
                    f"target_TPR={expected_tpr:.4f}  "
                    f"target_TNR={expected_tnr:.4f}\n")
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
    ax1.plot(history.history['accuracy'],     label='Train')
    ax1.plot(history.history['val_accuracy'], label='Val')
    if expected_acc is not None:
        ax1.axhline(expected_acc, color='r', linestyle='--',
                    label=f'Target ({expected_acc:.4f})')
    ax1.set_title(f'Simon64/128 Enhanced  {nr}r  Accuracy')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy'); ax1.legend()
    ax2.plot(history.history['loss'],     label='Train')
    ax2.plot(history.history['val_loss'], label='Val')
    ax2.set_title(f'Simon64/128 Enhanced  {nr}r  MSE Loss')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('MSE'); ax2.legend()
    plt.tight_layout()
    p = os.path.join(output_dir, f'history_{nr}r_{diff_tag}.png')
    plt.savefig(p, dpi=100); plt.close()
    print(f"Plot → {p}")


# ═══════════════════════════════════════════════════════════════════════════
# Full closed‑loop pipeline (Steps 1‑10 from the flowchart)
# ═══════════════════════════════════════════════════════════════════════════

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
                       output_dir='./results_simon64128_enhanced'):
    """
    Full closed-loop pipeline (Steps 1-10 of the flowchart).

    Returns the final result dict and a list of per-iteration results.
    """
    print("\n" + "█" * 76)
    print(f"  FULL PIPELINE  --  Simon64/128  --  {nr} rounds")
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


# ═══════════════════════════════════════════════════════════════════════════
# Sanity checks
# ═══════════════════════════════════════════════════════════════════════════

def check_simon64128():
    """Verify Simon64/128 correctness using the official test vector."""
    # Official test vector from Simon specification
    plain = (0x656b696c, 0x20646e75)
    key = np.array([[0x1b1a1918], [0x13121110], [0x0b0a0908], [0x03020100]], dtype=np.uint64)
    cipher_correct = (0x44c8fc20, 0xb9dfa07a)

    ks = expand_key_simon64128(key, 44)
    c0, c1 = encrypt_simon64128(plain, ks)
    assert c0 == cipher_correct[0] and c1 == cipher_correct[1], \
        f"Test vector failed: got ({hex(c0)},{hex(c1)}) expected ({hex(cipher_correct[0])},{hex(cipher_correct[1])})"
    print("  check 0 PASSED: official test vector (44 rounds)")

    # Additional random round-trip tests
    np.random.seed(13)
    n    = 500
    p0l  = _rand_words(n)
    p0r  = _rand_words(n)
    keys = np.stack([_rand_words(n) for _ in range(4)], axis=0)

    ks44     = expand_key_simon64128(keys, 44)
    c0l, c0r = encrypt_simon64128((p0l, p0r), ks44)
    rx, ry = c0l.copy(), c0r.copy()
    for k in reversed(ks44):
        rx, ry = ry, (F_simon64(ry) ^ rx ^ k) & MASK_VAL
    assert np.all(rx == p0l) and np.all(ry == p0r), "Decrypt mismatch!"
    print("  check 1 PASSED: encrypt/decrypt round-trip (500 samples, 44 rounds, m=4)")

    assert len(ks44) == 44
    assert all(np.all((k & MASK_VAL) == k) for k in ks44)
    print("  check 2 PASSED: 44 valid 32-bit round keys")

    for t in [26, 27]:
        kst = expand_key_simon64128(keys, t)
        assert len(kst) == t
        assert all(np.all((k & MASK_VAL) == k) for k in kst)
    print("  check 3 PASSED: key schedules valid for 26r / 27r")

    diff     = (0x0, 0x80000000)
    key_diff = (0x0, 0x0, 0x0, 0x80000000)
    keys2    = np.array([
        keys[0] ^ key_diff[0], keys[1] ^ key_diff[1],
        keys[2] ^ key_diff[2], keys[3] ^ key_diff[3],
    ], dtype=np.uint64) & MASK_VAL
    ks26  = expand_key_simon64128(keys,  26)
    ks26d = expand_key_simon64128(keys2, 26)
    c0l26, c0r26 = encrypt_simon64128((p0l, p0r), ks26)
    c1l26, c1r26 = encrypt_simon64128(
        ((p0l ^ diff[0]) & MASK_VAL, (p0r ^ diff[1]) & MASK_VAL), ks26d)
    dL = (c0l26 ^ c1l26) & MASK_VAL
    dR = (c0r26 ^ c1r26) & MASK_VAL
    print(f"  check 4 PASSED: 26r Table6 diff sample  "
          f"dL={hex(int(dL[0]))}  dR={hex(int(dR[0]))}")

    print("  Simon64/128 cipher checks PASSED")


def quick_data_check_enhanced(nr=26, n=200, extra_words=2):
    """Quick sanity check for the enhanced data generator."""
    pos_diffs = [
        ((0x0, 0x80000000), (0x0, 0x0, 0x0, 0x80000000)),
        ((0x0, 0x40000000), (0x0, 0x0, 0x0, 0x40000000)),
    ]
    neg_diffs = [
        ((0x0, 0x20),       (0x0, 0x0, 0x0, 0x20)),
        ((0x0, 0x40),       (0x0, 0x0, 0x0, 0x40)),
    ]
    X, Y = make_train_data_enhanced(n, nr,
                                     pos_diffs=pos_diffs,
                                     neg_diffs=neg_diffs,
                                     s_groups=8,
                                     extra_words=extra_words)
    wpg      = 8 + extra_words
    expected = (n, 8 * wpg * WORD_SIZE)
    assert X.shape == expected, f"Shape mismatch: {X.shape} vs {expected}"
    assert X.dtype == np.uint8
    assert set(np.unique(X)).issubset({0, 1})
    print(f"  data shape  : {X.shape}  "
          f"(8 groups × {wpg} words × {WORD_SIZE} bits)")
    print(f"  pos_rate    : {Y.mean():.3f}  (expected ~0.50)")
    print("  Enhanced data generation check PASSED")


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def _parse_hex_tuple(s, n_words):
    parts = [p.strip() for p in s.split(',')]
    if len(parts) != n_words:
        raise ValueError(
            f"Expected {n_words} comma-separated hex values, got: {s!r}")
    return tuple(int(p, 16) for p in parts)


def _parse_diff_list(args_list, n_words):
    """
    Parse a list of strings like ["0x0,0x80000000", "0x0,0x40000000"] into
    a list of tuples [(0x0, 0x80000000), (0x0, 0x40000000)].
    """
    return [_parse_hex_tuple(s, n_words) for s in args_list]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            'CNN+Transformer v2  ENHANCED Multi-Differential Distinguisher\n'
            'for Simon64/128  (Table 6 of the paper)\n'
            '\n'
            'Two modes:\n'
            '  (a) Single-shot (default): trains once with the given diffs\n'
            '      (or Table 6 diffs if available).\n'
            '  (b) Full pipeline (--use_pipeline): runs the closed loop from\n'
            '      the flowchart -- GA selects diffs (Step 2), trains and\n'
            '      evaluates (Steps 3-8), checks threshold (Step 9), and\n'
            '      refines & retries (Step 10) until the threshold is met or\n'
            '      --max_iterations is exhausted.\n'
            '\n'
            'Examples:\n'
            '  # Both Table 6 rounds (default single-shot):\n'
            '  python simon64128_enhanced_cnn_transformer.py --rounds 26 27\n\n'
            '  # GA pipeline for an unknown round:\n'
            '  python simon64128_enhanced_cnn_transformer.py --rounds 28 \\\n'
            '      --use_pipeline --threshold_acc 0.55 --max_iterations 3\n\n'
            '  # Custom diffs, single-shot\n'
            '  python simon64128_enhanced_cnn_transformer.py --rounds 27 \\\n'
            '      --pos_diffs "0x0,0x80000000" "0x0,0x40000000" \\\n'
            '      --neg_diffs "0x0,0x20" "0x0,0x40" \\\n'
            '      --pos_key_diffs "0x0,0x0,0x0,0x80000000" "0x0,0x0,0x0,0x40000000" \\\n'
            '      --neg_key_diffs "0x0,0x0,0x0,0x20" "0x0,0x0,0x0,0x40"\n\n'
            '  # Sanity checks only:\n'
            '  python simon64128_enhanced_cnn_transformer.py --sanity_only\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--rounds',      type=int, nargs='+', default=[26, 27],
                        help='Round number(s). Default: 26 27')
    parser.add_argument('--pos_diffs',     type=str, nargs='+', default=None,
                        help='Positive input diffs as "0xHH,0xHH" each.')
    parser.add_argument('--neg_diffs',     type=str, nargs='+', default=None,
                        help='Negative input diffs as "0xHH,0xHH" each.')
    parser.add_argument('--pos_key_diffs', type=str, nargs='+', default=None,
                        help='Positive key diffs as "0xHH,0xHH,0xHH,0xHH" each '
                             '(4 words for Simon64/128).')
    parser.add_argument('--neg_key_diffs', type=str, nargs='+', default=None,
                        help='Negative key diffs as "0xHH,0xHH,0xHH,0xHH" each.')
    parser.add_argument('--n_train',     type=int,   default=2 * 10**7)
    parser.add_argument('--n_val',       type=int,   default=2 * 10**6)
    parser.add_argument('--extra_words', type=int,   default=2,
                        help='Extra partial-dec words/group. 0=8w/2560b, 2=10w/3200b')
    parser.add_argument('--epochs',      type=int,   default=40)
    parser.add_argument('--batch_size',  type=int,   default=30000)
    parser.add_argument('--high_lr',     type=float, default=0.002)
    parser.add_argument('--low_lr',      type=float, default=0.0001)
    parser.add_argument('--lr_epoch',    type=int,   default=40)
    parser.add_argument('--s_groups',    type=int,   default=8)
    parser.add_argument('--depth',       type=int,   default=10)
    parser.add_argument('--num_filters', type=int,   default=128)
    parser.add_argument('--d1',          type=int,   default=256)
    parser.add_argument('--d2',          type=int,   default=256)
    parser.add_argument('--reg_param',   type=float, default=1e-5)
    parser.add_argument('--dropout',     type=float, default=0.3)
    parser.add_argument('--ks1',         type=int,   default=1)
    parser.add_argument('--ks2',         type=int,   default=3)
    parser.add_argument('--ks3',         type=int,   default=3)
    parser.add_argument('--num_heads',              type=int, default=4)
    parser.add_argument('--ff_dim',                 type=int, default=512)
    parser.add_argument('--num_transformer_blocks', type=int, default=3)
    parser.add_argument('--se_ratio',               type=int, default=4)
    parser.add_argument('--output_dir',  type=str,
                        default='./results_simon64128_enhanced')
    parser.add_argument('--sanity_only', action='store_true')

    # ── Full pipeline mode (Steps 1-10 with GA + refinement loop) ───────
    parser.add_argument('--use_pipeline', action='store_true',
        help='Run the full closed-loop pipeline: GA-based differential '
             'selection + automatic refinement loop until threshold or '
             'max_iterations is reached.')
    parser.add_argument('--max_iterations', type=int, default=3,
        help='Max refinement iterations when --use_pipeline is set.')
    parser.add_argument('--ga_pop_size',    type=int, default=200,
        help='L^2: initial random population pool size (paper Algorithm 1).')
    parser.add_argument('--ga_generations', type=int, default=50,
        help='Number of evolutionary iterations (paper: 50).')
    parser.add_argument('--ga_n_fitness',   type=int, default=16384,
        help='Samples t used to evaluate the bias score (Definition 4).')
    parser.add_argument('--ga_n_select',    type=int, default=2,
        help='Differentials to pick from the 0.06-filtered pool for each of '
             'pos and neg sets (total = 2 x ga_n_select).')
    parser.add_argument('--ga_min_hw',      type=int, default=1,
        help='Minimum Hamming weight of initial differences (paper: 1).')
    parser.add_argument('--ga_max_hw',      type=int, default=2,
        help='Maximum Hamming weight of initial differences (paper: 2).')
    parser.add_argument('--ga_bias_filter', type=float, default=0.06,
        help='Keep candidates within this margin of best bias score (paper: 0.06).')
    parser.add_argument('--threshold_acc', type=float, default=None,
        help='Accuracy threshold for Step 9 (pipeline mode). Required for '
             'pipeline mode unless you know the round\'s expected target.')
    parser.add_argument('--threshold_tpr', type=float, default=None)
    parser.add_argument('--threshold_tnr', type=float, default=None)

    args = parser.parse_args()

    print("\nRunning sanity checks ...")
    check_simon64128()
    quick_data_check_enhanced(extra_words=args.extra_words)
    print()

    if args.sanity_only:
        print("All sanity checks passed.")
        exit(0)

    have_manual = (args.pos_diffs is not None and args.neg_diffs is not None
                   and args.pos_key_diffs is not None
                   and args.neg_key_diffs is not None)

    if have_manual:
        cli_pos_plain = _parse_diff_list(args.pos_diffs,     2)
        cli_neg_plain = _parse_diff_list(args.neg_diffs,     2)
        cli_pos_key   = _parse_diff_list(args.pos_key_diffs, 4)
        cli_neg_key   = _parse_diff_list(args.neg_key_diffs, 4)
        if len(cli_pos_plain) != len(cli_pos_key):
            raise ValueError("--pos_diffs and --pos_key_diffs must have the "
                             "same number of entries.")
        if len(cli_neg_plain) != len(cli_neg_key):
            raise ValueError("--neg_diffs and --neg_key_diffs must have the "
                             "same number of entries.")
        cli_pos_diffs = list(zip(cli_pos_plain, cli_pos_key))
        cli_neg_diffs = list(zip(cli_neg_plain, cli_neg_key))
    else:
        cli_pos_diffs = None
        cli_neg_diffs = None

    all_results = []
    for nr in args.rounds:
        if cli_pos_diffs is not None:
            pos_diffs    = cli_pos_diffs
            neg_diffs    = cli_neg_diffs
            expected_acc = None
            expected_tpr = None
            expected_tnr = None
        elif nr in _TABLE6_DEFAULT:
            e            = _TABLE6_DEFAULT[nr]
            pos_diffs    = e['pos_diffs']
            neg_diffs    = e['neg_diffs']
            expected_acc = e['expected_acc']
            expected_tpr = e['expected_tpr']
            expected_tnr = e['expected_tnr']
        else:
            print(f"\nERROR: round {nr} not in Table 6 "
                  f"(known rounds: {sorted(_TABLE6_DEFAULT)}).")
            print(f"  Supply all four --pos_diffs / --neg_diffs / "
                  f"--pos_key_diffs / --neg_key_diffs explicitly.")
            print(f"  Skipping round {nr}.")
            continue

        # ── FULL PIPELINE MODE  (Steps 1-10 with GA + refinement loop) ──
        if args.use_pipeline:
            thr_acc = args.threshold_acc if args.threshold_acc is not None else expected_acc
            thr_tpr = args.threshold_tpr if args.threshold_tpr is not None else expected_tpr
            thr_tnr = args.threshold_tnr if args.threshold_tnr is not None else expected_tnr
            if thr_acc is None:
                print(f"  No threshold available for round {nr} - "
                      f"set --threshold_acc explicitly. Skipping.")
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

        # ── Single-shot mode (original behaviour) ───────────────────────
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
        print("  FINAL SUMMARY  --  CNN+Transformer v2 ENHANCED  Simon64/128  (Table 6)")
        print(f"  {'Nr':<5} {'Pos-diff0':<14} {'Neg-diff0':<14} "
              f"{'Acc':<8} {'TPR':<8} {'TNR':<8} "
              f"{'Target':<8} {'T.TPR':<8} {'T.TNR':<8} {'Reached?'}")
        print("  " + "-" * 90)
        for r in all_results:
            t6 = _TABLE6_DEFAULT.get(r['nr'])
            pd0 = hex(r['pos_diffs'][0][0][1])
            nd0 = hex(r['neg_diffs'][0][0][1])
            if t6:
                reached = "YES ***" if r['acc'] >= t6['expected_acc'] else "no"
                print(f"  {r['nr']:<5} {pd0:<14} {nd0:<14} "
                      f"{r['acc']:<8.4f} {r['tpr']:<8.4f} {r['tnr']:<8.4f} "
                      f"{t6['expected_acc']:<8.4f} {t6['expected_tpr']:<8.4f} "
                      f"{t6['expected_tnr']:<8.4f} {reached}")
            else:
                print(f"  {r['nr']:<5} {pd0:<14} {nd0:<14} "
                      f"{r['acc']:<8.4f} {r['tpr']:<8.4f} {r['tnr']:<8.4f} "
                      f"{'N/A':<8} {'N/A':<8} {'N/A':<8} (custom)")
        print("=" * 90)