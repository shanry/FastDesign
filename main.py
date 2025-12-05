import os
import sys
import time
import copy
from datetime import datetime
import random
import json
import heapq
import argparse

from multiprocessing import Pool

import numpy as np
import pandas as pd
from pyparsing import Dict

from utils.vienna import position_ed_pd_mfe, position_ed_ned_mfe, mfe, position_ed_pd_mfe_v0
from utils.structure import (
    struct_dist,
    dotbracket2target,
    dotbracket2constraint_vienna,
)
from utils.structure import len_motif, pairs_match
from utils.constants import P1, P2, U1, U2

from decimal import Decimal, getcontext

getcontext().prec = 200

import multiprocessing

multiprocessing.set_start_method("fork")

from utils.motif import MotifNode, decompose, get_selected_motifs


name2pair = {
    "cg": ["CG", "GC"],
    "cggu": ["CG", "GC", "GU", "UG"],
    "cgau": ["CG", "GC", "AU", "UA"],
    "all": ["CG", "GC", "AU", "UA", "GU", "UG"],
}


nuc_others = {"A": "CGU", "C": "AGU", "U": "ACG", "G": "ACU"}


nuc_pair_others = {
    "AU": ["UA", "CG", "GC", "UG", "GU"],
    "UA": ["AU", "CG", "GC", "UG", "GU"],
    "CG": ["AU", "UA", "GC", "UG", "GU"],
    "GC": ["AU", "UA", "CG", "UG", "GU"],
    "GU": ["AU", "UA", "CG", "GC", "UG"],
    "UG": ["AU", "UA", "CG", "GC", "GU"],
}

nuc_all = ["A", "C", "G", "U"]
nuc_pair_all = ["AU", "UA", "CG", "GC", "UG", "GU"]

# global parameters for SAMFEO
STAY = 2000
STOP = Decimal("0.01")
EPSILON_r = Decimal("1e-4")
MAX_REPEAT = 1000
FREQ_PRINT = 10
COUNT_DESIGN = 200
LOG = False

# setts for cubic pruning
REDESIGN = True
RESCORE = True

# parameters for parallelization
WORKER_COUNT = 10
BATCH_SIZE = 20


# global parameters for input and output
path_motifs = "data/easy_motifs.txt"
size_motifs = 3
selected_motifs = None
OUTPUT_DIR = None
META_DATA = dict()

# global random seed
seed_np = None


class RNAStructure:

    def __init__(
        self, seq, score, v=None, v_list=None
    ):  # v_list: positional NED, v: objective value, socore: used for priority queue
        self.seq = seq
        self.score = score
        self.v = v
        self.v_list = v_list

    def __gt__(self, other):
        return self.score > other.score

    def __lt__(self, other):
        return self.score < other.score

    def __eq__(self, other):
        return self.seq == other.seq

    def __ge__(self, other):
        return self.score >= other.score

    def __le__(self, other):
        return self.score <= other.score

    def __str__(self):
        return f"{self.seq}: {self.score: .4e}"

    def __repr__(self):
        return f"RNAStructure('{self.seq}', {self.score})"

    def __hash__(self):
        return hash(self.seq)


class Design:

    def __init__(self, sequence, structure):
        self.sequence = sequence
        self.structure = structure

        self.type = None # "motif" or "whole structure"

        # RNA attributes with respect to the target structure / motif
        self.dist = None  # structure distance
        self.prob = None  # structure probability
        self.ned = None  # normalized ensemble defect
        self.ensemble_defect_list = None

        # meaning of score depends on the context
        self.score = None

    @property
    def probability_defect(self):
        if self.prob is None:
            raise AttributeError("prob is not set")
        return 1 - self.prob

    @property
    def ensemble_defect(self):
        if self.ned is None:
            raise AttributeError("ned is not set")
        return self.ned

    def set_score(self, obj_name, length):
        if obj_name == "probability_defect":
            if self.prob is None:
                raise ValueError("self.prob is not set")
            self.score = Decimal.log10(self.prob)
        elif obj_name == "ensemble_defect":
            if self.ned is None:
                raise ValueError("self.ned is not set")
            self.score = (1 - self.ned) * length
        else:
            assert False

    def __str__(self):
        return f"{self.sequence}\t{self.structure}"

    def __repr__(self):
        return f"Design('{self.sequence}', '{self.structure}')"
    

def combine_sequences(motif, seq_list):

    # create a list of sequences from the combined scores
    dotbrk_list = list(motif)  # convert motif to a list of characters
    # for sum_score, seq_list in k_best:
    x_self_one = seq_list.pop(0)  # take the first sequence from the combined scores
    x_self_one_list = list(x_self_one)  # convert the sequence to a list of characters
    x_combined_list = []
    # replace * with the designs of children
    for i, char in enumerate(dotbrk_list):
        if char == "*":
            x_combined_list.append(
                seq_list.pop(0)
            )  # take the first design from the list
            # pop three times from x_self_list
            assert (
                x_self_one_list.pop(0) == "A"
            )  # remove the first character from x_self_list
            assert (
                x_self_one_list.pop(0) == "A"
            )  # remove the second character from x_self_list
            assert (
                x_self_one_list.pop(0) == "A"
            )  # remove the third character from x_self_list
        elif char in "().":
            x_combined_list.append(
                x_self_one_list.pop(0)
            )  # keep the original character from the sequence
        else:
            assert (
                char == "5" or char == "3"
            ), f"Unexpected character in unit_list: {char}"
    x_combined = "".join(x_combined_list)  # combine the list into a string

    return x_combined


def add_constraint(seq, constraint):
    assert len(seq) == len(constraint)
    seq_new = list(seq)
    for i, s in enumerate(constraint):
        if s == "x":
            seq_new[i] = "A"
    return "".join(seq_new)


def init_with_pair(t, pos_pairs, pairs_init):
    rna = list("." * len(t))
    assert len(rna) == len(t)
    for i, s in enumerate(t):
        if s == ".":
            rna[i] = "A"
            if name_pair == "all":
                rna[i] = np.random.choice(["A", "C", "G", "U"])
        elif s == "(":
            j = pos_pairs[i]
            pair = np.random.choice(pairs_init)
            rna[i] = pair[0]
            rna[j] = pair[1]
        elif s == ")":
            pass
        else:
            raise ValueError(
                f"the value of structure at position: {i} is not right: {s}!"
            )
    return "".join(rna)


# targeted initilization
def init_k(target, pos_pairs, k):
    print(f"name_pair: {name_pair}")
    pair_pool = name2pair[name_pair]
    print(f"pair_pool: {pair_pool}")
    init_0 = init_with_pair(target, pos_pairs, pair_pool)
    p_list = [init_0]
    # if too few pairs then use 'cggu', however this may never happen
    if k > len(pair_pool) ** (len(pos_pairs) / 2) and len(pair_pool) < 4:
        pair_pool = name2pair["cggu"]
    # the max number of initial sequences is: len(pair_pool)**(len(pos_pairs)/2)
    while len(p_list) < min(k, len(pair_pool) ** (len(pos_pairs) / 2)):
        init_i = init_with_pair(target, pos_pairs, pair_pool)
        if init_i not in p_list:
            p_list.append(init_i)
    return p_list


# targeted initilization with constraint
def init_k_constrained(target, constraint, pos_pairs, k):
    print(f"name_pair: {name_pair}")
    pair_pool = name2pair[name_pair]
    print(f"pair_pool: {pair_pool}")
    init_0 = init_with_pair(target, pos_pairs, pair_pool)
    init_0 = add_constraint(init_0, constraint)
    p_list = [init_0]
    # if too few pairs then use 'cggu', however this may never happen
    if k > len(pair_pool) ** (len(pos_pairs) / 2) and len(pair_pool) < 4:
        pair_pool = name2pair["cggu"]
    # the max number of initial sequences is: len(pair_pool)**(len(pos_pairs)/2)
    while len(p_list) < min(k, len(pair_pool) ** (len(pos_pairs) / 2)):
        init_i = init_with_pair(target, pos_pairs, pair_pool)
        init_i = add_constraint(init_i, constraint)
        if init_i not in p_list:
            p_list.append(init_i)
    return p_list


def mutate_pair(nuc_i, nuc_j, exclude=False):
    pair_ij = nuc_i + nuc_j
    return (
        np.random.choice(nuc_pair_others[pair_ij])
        if exclude
        else np.random.choice(nuc_pair_all)
    )


def mutate_unpair(nuc_i, exclude=False):
    return (
        np.random.choice(list(nuc_others[nuc_i]))
        if exclude
        else np.random.choice(nuc_all)
    )


# traditional mutation
def mutate_tradition(seq, pairs, v, v_list, T, pairs_dg=None):
    v_list = [v / T for v in v_list]
    probs = np.exp(v_list) / sum(np.exp(v_list))
    index = np.random.choice(list(range(len(seq))), p=probs)
    seq_next = [nuc for nuc in seq]
    if index in pairs:
        i = min(index, pairs[index])
        j = max(index, pairs[index])
        pair_ij = seq[i] + seq[j]
        pair_new = np.random.choice(nuc_pair_others[pair_ij])
        seq_next[i] = pair_new[0]
        seq_next[j] = pair_new[1]
    else:
        c = np.random.choice(list(nuc_others[seq[index]]))
        assert c != seq[index]
        seq_next[index] = c
    return "".join(seq_next)


# structured mutation
def mutate_structured(seq, pairs, v, v_list, T):
    v_list = [v / T for v in v_list]
    probs = np.exp(v_list) / sum(np.exp(v_list))
    index = np.random.choice(list(range(len(seq))), p=probs)
    pairs_mt = []
    unpairs_mt = []

    if index in pairs:
        i = min(index, pairs[index])
        j = max(index, pairs[index])
        pairs_mt.append((i, j))
        if j - 1 in pairs and pairs[j - 1] == i + 1:
            pairs_mt.append((pairs[j - 1], j - 1))
            if i + 2 not in pairs and j - 2 not in pairs:
                unpairs_mt.append(i + 2)
                unpairs_mt.append(j - 2)
        if i + 1 not in pairs and j - 1 not in pairs:
            unpairs_mt.append(i + 1)
            unpairs_mt.append(j - 1)
    else:
        unpairs_mt.append(index)
        if index - 1 in pairs and pairs[index - 1] > index:
            pairs_mt.append((index - 1, pairs[index - 1]))
            if pairs[index - 1] - 1 not in pairs:
                unpairs_mt.append(pairs[index - 1] - 1)
        elif index + 1 in pairs and pairs[index + 1] < index:
            pairs_mt.append((pairs[index + 1], index + 1))
            if pairs[index + 1] + 1 not in pairs:
                unpairs_mt.append(pairs[index + 1] + 1)

    assert len(pairs_mt) <= 2, pairs_mt
    assert len(unpairs_mt) <= 2, unpairs_mt

    # one pair
    if len(pairs_mt) == 1:
        pairs_selected_index = np.random.choice(range(len(P1)))
        pairs_selected = P1[pairs_selected_index]
    else:  # two pair
        pairs_selected_index = np.random.choice(range(len(P2)))
        pairs_selected = P2[pairs_selected_index]

    # one unpair
    if len(unpairs_mt) == 1:
        unpairs_selected_index = np.random.choice(range(len(U1)))
        unpairs_selected = U1[unpairs_selected_index]
    else:  # two unpair
        unpairs_selected_index = np.random.choice(range(len(U2)))
        unpairs_selected = U2[unpairs_selected_index]

    nuc_list = list(seq)
    for pos_pair, pair in zip(pairs_mt, pairs_selected):
        nuc_list[pos_pair[0]] = pair[0]
        nuc_list[pos_pair[1]] = pair[1]
    for pos_unpair, unpair in zip(unpairs_mt, unpairs_selected):
        nuc_list[pos_unpair] = unpair
    return "".join(nuc_list)


def samfeo(
    motif, f, steps, k, t=1, check_mfe=True, sm=True, freq_print=FREQ_PRINT, initial_list=None, seed=None
):
    # assert seed is not None
    if seed is not None:
        np.random.seed(seed)
        print(f"samfeo seed: {seed}")
    else:
        print("samfeo seed: None")
    # check if it is an external motif
    target = None
    if motif[0] == "5" and motif[-1] == "3":
        motif_extended = "(" + motif[1:-1] + ")"
        target = dotbracket2target(motif_extended)[
            1:-1
        ]  # remove the external 5' and 3' ends
    else:
        target = dotbracket2target(motif)
    constraint = dotbracket2constraint_vienna(motif)
    print(f"target structure: {target}")
    print(f"constraint\t: {constraint}")
    start_time = time.time()
    # global seed_np
    # np.random.seed(seed_np)
    # print(f'seed_np: {seed_np}')
    if sm:
        mutate = mutate_structured
    else:
        mutate = mutate_tradition
    print(
        f"steps: {steps}, t: {t}, k: {k}, structured mutation: {sm}, ensemble objective: {f.__name__}"
    )

    # targeted initilization
    pairs = pairs_match(target)
    if initial_list is None:
        initial_list = init_k_constrained(target, constraint, pairs, k)
    history = set()
    k_best = []
    best_many = []
    log = []
    dist_list = []
    dist_best = None
    seq_list = []  # pair of sequences from improved mutation
    mfe_list = []
    umfe_list = []
    count_umfe = 0
    ned_best = (1, None)
    for sequence in initial_list:
        positional_ensemble_defect, objective, suboptimal_structure_list = f(
            sequence, target, constraint
        )  # suboptimal_structure_list: (multiple) MFE structures by subopt of ViennaRNA
        rna_struct = RNAStructure(seq=sequence, score=-objective, v=objective, v_list=positional_ensemble_defect)
        rna_struct.dist = min(
            [struct_dist(target, suboptimal_structure) for suboptimal_structure in suboptimal_structure_list]
        )
        rna_struct.subcount = len(suboptimal_structure_list)
        k_best.append(rna_struct)
        history.add(rna_struct.seq)
        design = Design(sequence, target)
        design.dist = rna_struct.dist
        if f == position_ed_pd_mfe:
            design.prob = Decimal(1.0) - Decimal(objective)
        elif f == position_ed_ned_mfe:
            design.ned = Decimal(objective)
        else:
            assert False
        design.ensemble_defect_list = positional_ensemble_defect
        best_many.append(design)
        # record the best NED
        ned_sequence = np.mean(positional_ensemble_defect)
        if ned_sequence <= ned_best[0]:
            ned_best = (ned_sequence, sequence)

    # priority queue
    heapq.heapify(k_best)
    for i, rna_struct in enumerate(k_best):
        print(i, rna_struct)
        log.append(rna_struct.v)
        if rna_struct.dist == 0:  # MFE solution
            mfe_list.append(rna_struct.seq)
        if rna_struct.dist == 0 and rna_struct.subcount == 1:  # UMFE solution
            dist_list.append(-2)
            umfe_list.append(rna_struct.seq)
            count_umfe += 1
        else:
            dist_list.append(rna_struct.dist)
        if not dist_best or rna_struct.dist <= dist_best[0]:
            dist_best = (rna_struct.dist, rna_struct.seq)

    # log of lowest objective value at eachs iterations
    objective_min = min(log)
    iter_min = 0
    log_min = [objective_min]
    for i in range(steps):
        # sequence selection
        # score_list = [rna_struct.score/Decimal(t*2) for rna_struct in k_best] # objective values
        # probs_boltzmann_1 = np.exp(score_list)/sum(np.exp(score_list)) # boltzmann distribution
        score_array = np.array(
            [float(rna_struct.score) / (t * 2) for rna_struct in k_best], dtype=float
        )
        exp_scores = np.exp(score_array)
        probs_boltzmann_1 = exp_scores / exp_scores.sum()
        try:
            p = np.random.choice(k_best, p=probs_boltzmann_1)
        except Exception as e:
            print("score_array:", score_array)
            p = np.random.choice(k_best)
            raise e
        # position sampling and mutation
        seq_next = mutate(p.seq, pairs, p.v, p.v_list, t)
        seq_next = add_constraint(seq_next, constraint)
        num_repeat = 0
        while seq_next in history:
            num_repeat += 1
            if num_repeat > len(target) * MAX_REPEAT:
                break
            try:
                p = np.random.choice(k_best, p=probs_boltzmann_1)
            except Exception as e:
                print(e)
                print("probs_boltzmann_1:", probs_boltzmann_1)
                p = np.random.choice(k_best)
            seq_next = mutate(p.seq, pairs, p.v, p.v_list, t)
            seq_next = add_constraint(seq_next, constraint)
        if num_repeat > len(target) * MAX_REPEAT:
            print(f"num_repeat: {num_repeat} > {len(target)*MAX_REPEAT} for {motif}")
            break
        history.add(seq_next)
        positional_ensemble_defect_next, objective_next, suboptimal_structure_list_next = f(
            seq_next, target, constraint
        )
        dist_next = min([struct_dist(target, suboptimal_structure) for suboptimal_structure in suboptimal_structure_list_next])
        design = Design(seq_next, target)
        design.dist = dist_next
        if f == position_ed_pd_mfe:
            design.prob = Decimal(1.0) - Decimal(objective_next)
        elif f == position_ed_ned_mfe:
            design.ned = Decimal(objective_next)
        else:
            assert False
        design.ensemble_defect_list = positional_ensemble_defect_next
        best_many.append(design)
        # mfe and umfe solutions as byproducts
        umfe = False
        if check_mfe:
            dist = dist_next
            if dist == 0:
                mfe_list.append(seq_next)
                if len(suboptimal_structure_list_next) == 1:
                    umfe = True
                    umfe_list.append(seq_next)
        else:
            dist = len(target)  # set a dummy dist
        if not umfe:
            dist_list.append(dist)
        else:
            dist = -2  # set a dummy dist for UMFE
            dist_list.append(dist)
            count_umfe += 1
        # update the best distance
        if not dist_best or dist <= dist_best[0]:
            dist_best = (dist, seq_next)

        # compare with best ned
        ned_next = np.mean(positional_ensemble_defect_next)
        if ned_next <= ned_best[0]:
            ned_best = (ned_next, seq_next)

        # update priority queue(multi-frontier)
        rna_struct_next = RNAStructure(seq_next, -objective_next, objective_next, positional_ensemble_defect_next)
        if rna_struct_next.v <= p.v:
            seq_list.append(
                (p.seq, rna_struct_next.seq, str(p.v), str(rna_struct_next.v))
            )

        if len(k_best) < k:
            heapq.heappush(k_best, rna_struct_next)
        elif rna_struct_next > k_best[0]:  # min-heap
            heapq.heappushpop(k_best, rna_struct_next)
        if objective_next <= objective_min:
            iter_min = i

        # update log
        objective_min = min(objective_min, objective_next)
        log_min.append(objective_min)
        log.append(objective_next)
        assert len(dist_list) == len(log)
        # assert len(seq_list) == len(log)

        # output information during iteration
        if (i + 1) % freq_print == 0:
            improve = objective_min - log_min[-freq_print]
            if check_mfe:
                print(
                    f"iter: {i+1: 5d}\t value: {objective_min: .4e}\t mfe count: {len(mfe_list): 5d}\t umfe count: {count_umfe}\t best iter: {iter_min} improve: {improve:.2e}"
                )
            else:
                print(
                    f"iter: {i+1: 5d}\t value: {objective_min: .4e}\t best iter: {iter_min} improve: {improve:.4e}"
                )

        EPSILON_r = Decimal("1e-4")
        # stop if convergency condition is satisfied
        if (
            f == position_ed_pd_mfe
            and i > 500
            and (
                objective_min < STOP
                or (
                    len(log_min) > STAY
                    and objective_min - log_min[-STAY] > abs(EPSILON_r * objective_min)
                )
            )
        ):
            break
        if (
            f == position_ed_ned_mfe
            and i > 500
            and (
                objective_min < STOP
                or (
                    len(log_min) > STAY
                    and objective_min - log_min[-STAY] > abs(EPSILON_r * objective_min)
                )
            )
        ):
            break
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    log = [str(decimal) for decimal in log]
    best_many = sorted(best_many, key=lambda x: getattr(x, objective_name), reverse=False)[:COUNT_DESIGN]
    return (
        k_best,
        log,
        mfe_list,
        umfe_list,
        dist_best,
        ned_best,
        seq_list,
        elapsed_time,
        best_many
    )


def samfeo_structure(target, f, steps, k, t=1, check_mfe=True, sm=True, freq_print=FREQ_PRINT, initial_list=None):
    start_time = time.time()
    global seed_np
    np.random.seed(seed_np)
    print(f"seed_np: {seed_np}")
    if sm:
        mutate = mutate_structured
    else:
        mutate = mutate_tradition
    print(
        f"steps: {steps}, t: {t}, k: {k}, structured mutation: {sm}, ensemble objective: {f.__name__}"
    )

    # targeted initilization
    pairs = pairs_match(target)
    print('initial_list:', initial_list)
    if initial_list is None:
        initial_list = init_k(target, pairs, k)
    history = set()
    k_best = []
    # best_many = []
    log = []
    dist_list = []  # deprecated
    mfe_list = []
    umfe_list = []
    count_umfe = 0
    ned_best = (1, None)
    prob_best = (1, None)
    dist_best = (len(target), None)
    for sequence in initial_list:
        defect_list, prob_defect, y_mfe_list = position_ed_pd_mfe(
            sequence, target
        )  # defect_list: positional NED, objective: objective value, y_mfe_list: (multiple) MFE structures of sequence
        defect_list_decimal = [Decimal(str(defect_perposition)) for defect_perposition in defect_list]
        ned_sequence = sum(defect_list_decimal) / len(defect_list_decimal)
        if objective_name == "ensemble_defect":
            value = ned_sequence
        elif objective_name == "probability_defect":
            value = prob_defect
        else:
            assert False, "objective function not supported"
        rna_struct = RNAStructure(seq=sequence, score=-value, v=value, v_list=defect_list)
        rna_struct.dist = min(
            [struct_dist(target, y_mfe) for y_mfe in y_mfe_list]
        )  # ss: secondary structure
        rna_struct.subcount = len(y_mfe_list)
        k_best.append(rna_struct)
        history.add(rna_struct.seq)
        if ned_sequence <= ned_best[0]:
            ned_best = (ned_sequence, sequence)
        if prob_defect <= prob_best[0]:
            prob_best = (prob_defect, sequence)

    # priority queue
    heapq.heapify(k_best)
    for i, rna_struct in enumerate(k_best):
        print(i, rna_struct)
        log.append(-rna_struct.score)
        if rna_struct.dist == 0:  # MFE solution
            mfe_list.append(rna_struct.seq)
        if rna_struct.dist == 0 and rna_struct.subcount == 1:  # UMFE solution
            rna_struct.dist = -2  # set a dummy dist for UMFE
            dist_list.append(rna_struct.dist)
            umfe_list.append(rna_struct.seq)
            count_umfe += 1
        else:
            dist_list.append(rna_struct.dist)

        if dist_best[1] is None or rna_struct.dist <= dist_best[0]:
            dist_best = (rna_struct.dist, rna_struct.seq)

    # log of lowest objective value at eachs iterations
    v_min = min(log)
    iter_min = 0
    log_min = [v_min]
    for i in range(steps):
        score_array = np.array(
            [float(rna_struct.score) / (t * 2) for rna_struct in k_best], dtype=float
        )
        score_max = np.max(score_array)
        exp_scores = np.exp(score_array - score_max)
        sum_exp = exp_scores.sum()
        if not np.isfinite(sum_exp) or sum_exp <= 0:
            probs_boltzmann_1 = np.full(len(k_best), 1.0 / len(k_best), dtype=float)
        else:
            probs_boltzmann_1 = exp_scores / sum_exp
        try:
            p = np.random.choice(k_best, p=probs_boltzmann_1)
        except Exception as e:
            print("score_array:", score_array)
            print("probs_boltzmann_1:", probs_boltzmann_1)
            p = np.random.choice(k_best)

        # position sampling and mutation
        seq_next = mutate(p.seq, pairs, p.v, p.v_list, t)
        num_repeat = 0
        while seq_next in history:
            num_repeat += 1
            if num_repeat > len(target) * MAX_REPEAT:
                break
            try:
                p = np.random.choice(k_best, p=probs_boltzmann_1)
            except Exception as e:
                print(e)
                # print structure
                print("Value Error at structure:")
                print(probs_boltzmann_1)
                print(target)
                print("score_list", score_array)
                raise e
            seq_next = mutate(p.seq, pairs, p.v, p.v_list, t)
        if num_repeat > len(target) * MAX_REPEAT:
            print(f"num_repeat: {num_repeat} > {len(target)*MAX_REPEAT}")
            break
        history.add(seq_next)

        # evaluation new sequence
        # defect_list_next, objective_next, y_mfe_list = f(seq_next, target)
        defect_list_next, prob_defect_next, y_mfe_list_next = position_ed_pd_mfe(
            seq_next, target
        )  # defect_list: positional NED, prob_defect_next: prob defect, y_mfe_list: (multiple) MFE structures of sequence
        # ned_next = float(np.mean(defect_list_next))
        defect_list_next_decimal = [Decimal(str(defect_perposition)) for defect_perposition in defect_list_next]
        ned_next = sum(defect_list_next_decimal) / len(defect_list_next_decimal)
        dist = min([struct_dist(target, y_mfe) for y_mfe in y_mfe_list_next])

        # mfe and umfe solutions as byproducts
        umfe = False
        if check_mfe:
            if dist == 0:
                mfe_list.append(seq_next)
                if len(y_mfe_list_next) == 1:
                    umfe = True
                    umfe_list.append(seq_next)
        # else:
        #     dist = len(target)  # set a dummy dist
        if not umfe:
            dist_list.append(dist)
        else:
            dist = -2  # set a dummy dist for UMFE
            dist_list.append(dist)
            count_umfe += 1

        # update dist_best
        if dist_best[1] is None or dist <= dist_best[0]:
            dist_best = (dist, seq_next)

        # compare with best ned
        if ned_next <= ned_best[0]:
            ned_best = (ned_next, seq_next)
        if prob_defect_next <= prob_best[0]:
            prob_best = (prob_defect_next, seq_next)

        # update priority queue(multi-frontier)
        # if f == position_ed_ned_mfe:
        #     value_next = objective_next - 1
        #     assert abs(float(objective_next) - np.mean(defect_list_next)) < 1e-5, f"objective {objective_next}, mean defect {np.mean(defect_list_next)}"
        # elif f == position_ed_pd_mfe_v0:
        #     value_next = objective_next
        if objective_name == "ensemble_defect":
            value_next = ned_next
            # assert abs(float(objective_next) - np.mean(defect_list_next)) < 1e-5, f"objective {objective_next}, mean defect {np.mean(defect_list_next)}"
        elif objective_name == "probability_defect":
            value_next = prob_defect_next
        else:
            assert False, "objective function not supported!"
        rna_struct_next = RNAStructure(seq_next, -value_next, value_next, defect_list_next)

        if len(k_best) < k:
            heapq.heappush(k_best, rna_struct_next)
        elif rna_struct_next > k_best[0]:
            heapq.heappushpop(k_best, rna_struct_next)
        if value_next <= v_min:
            iter_min = i
        
        # if f == position_ed_ned_mfe:
        #     assert abs(ned_best[0] - (1 - max(k_best).score))

        # update log
        v_min = min(v_min, -rna_struct_next.score)
        log_min.append(v_min)
        log.append(-rna_struct_next.score)
        assert len(dist_list) == len(log)

        # output information during iteration
        if (i + 1) % freq_print == 0:
            improve = v_min - log_min[-freq_print]
            if check_mfe:
                print(
                    f"iter: {i+1: 5d}\t value: {v_min: .4e}\t mfe count: {len(mfe_list): 5d}\t umfe count: {count_umfe}\t best iter: {iter_min} improve: {improve:.2e}"
                )
            else:
                print(
                    f"iter: {i+1: 5d}\t value: {v_min: .4e}\t best iter: {iter_min} improve: {improve:.4e}"
                )

        # stop if convergency condition is satisfied
        global STOP, EPSILON_r, STAY
        STOP = float(STOP)
        # EPSILON_r = float(EPSILON_r)
        
        assert f == position_ed_pd_mfe_v0 or f == position_ed_ned_mfe, "objective function not supported"
        if f == position_ed_pd_mfe_v0 and (
            v_min < STOP - 1
            or (len(log_min) > STAY and v_min - log_min[-STAY] > abs(EPSILON_r * v_min))
        ):
            break
        if f == position_ed_ned_mfe and (
            v_min < STOP - 1
            or (len(log_min) > STAY and v_min - log_min[-STAY] > abs(EPSILON_r * v_min))
        ):
            break
    # ned_best = (float(ned_best[0]), ned_best[1])
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    # best_many = sorted(best_many, key=lambda x: x.prob, reverse=True)[: args.k2]
    return k_best, log, mfe_list, umfe_list, dist_best, ned_best, prob_best, elapsed_time


# structure design with multiple processing
def design_parallel(path_txt, name, func, num_step, k, t, seed=None):
    from multiprocessing import Pool

    print("BATCH_SIZE:", BATCH_SIZE)
    print("WORKER_COUNT:", WORKER_COUNT)
    targets = []
    with open(path_txt) as f:
        for line in f:
            targets.append(line.strip())
    data = []
    if LOG:
        cols = (
            "puzzle_name",
            "structure",
            "rna",
            "objective",
            "mfe",
            "dist",
            "time",
            "log",
            "k_best",
            "ned_best",
            "prob_best"
            "dist_best",
            "mfe_list",
            "umfe_list",
        )
    else:
        cols = (
            "puzzle_name",
            "structure",
            "rna",
            "objective",
            "mfe",
            "dist",
            "time",
            "k_best",
            "ned_best",
            "prob_best",
            "dist_best",
            "mfe_list",
            "umfe_list",
        )
    label = "e" if "easy" in path_motifs else "h"
    label += str(size_motifs)
    if RESCORE:
        label += "_rescore"
    else:
        label += "_approxi"
    filename = f"{name}_{objective_name}_step{num_step}_post{args.poststep}_{suffix}_{label}_kprune{args.k_prune}_time{int(time.time())}.csv"
    filename = os.path.join(OUTPUT_DIR, filename)
    print(f"Design results will be saved to {filename}")
    for i_batch in range(0, len(targets), BATCH_SIZE):
        pool = Pool(WORKER_COUNT)
        args_map = []
        for j, target in enumerate(
            targets[i_batch : min(i_batch + BATCH_SIZE, len(targets))]
        ):
            name_target = f"{name}_{i_batch + j}"
            args_map.append((name_target, target, seed))
        print("args_map:")
        print(args_map)
        results_pool = pool.starmap(design_pipeline, args_map)
        pool.close()
        pool.join()
        # return
        for j, result in enumerate(results_pool):
            idx_puzzle = i_batch + j
            puzzle_name = f"{name}_{idx_puzzle}"
            target = targets[idx_puzzle]
            print(f"target structure {idx_puzzle}, {puzzle_name}:")
            print(target)
            k_best, log, mfe_list, umfe_list, dist_best, ned_best, prob_best, elapsed_time = result

            rna_best = max(k_best)
            seq = rna_best.seq
            obj = None
            obj = -rna_best.score
            print("RNA sequence: ")
            print(seq)
            print("ensemble objective: ", obj)
            print(target)
            ss_mfe = mfe(seq)[0]
            dist = struct_dist(target, ss_mfe)
            print(ss_mfe)
            print(f"structure distance: {dist}")
            if LOG:
                data.append(
                    [
                        puzzle_name,
                        target,
                        seq,
                        obj,
                        ss_mfe,
                        dist,
                        elapsed_time,
                        log,
                        k_best,
                        ned_best,
                        prob_best,
                        dist_best,
                        mfe_list,
                        umfe_list
                    ]
                )
            else:
                data.append(
                    [
                        puzzle_name,
                        target,
                        seq,
                        obj,
                        ss_mfe,
                        dist,
                        elapsed_time,
                        k_best,
                        ned_best,
                        prob_best,
                        dist_best,
                        mfe_list,
                        umfe_list                    ]
                )
        df = pd.DataFrame(data, columns=cols)
        df.to_csv(filename)
        print(f"Results saved to {filename}")
    return filename


def motif_design(node, seed=None):
    assert node.motif is not None, "The helix tree must have a motif."
    dotbracket = node.motif.dotbracket
    target = dotbracket2target(dotbracket)
    motif = dotbracket
    if node.motif.type == "E":
        motif = "5" + dotbracket + "3"
    print(f"target structure: {target}")
    constraint = dotbracket2constraint_vienna(motif)
    print(f"constraint: {constraint}")
    # start_time = time.time()
    # adaptive step size: 20 * motif length
    initial_list = None
    step_adaptive = args.step if len(target) < 200 else int(args.step * 1.5)
    print(f"Initial list: {initial_list}")
    (
        k_best,
        log,
        mfe_list,
        umfe_list,
        dist_list,
        ned_best,
        seq_list,
        elapsed_time,
        best_many,
    ) = samfeo(
        motif,
        f_obj,
        step_adaptive,
        k=args.k,
        t=args.t,
        check_mfe=not args.nomfe,
        sm=not args.nosm,
        initial_list=initial_list,
        seed=seed
    )  # rna and ensemble defect
    node.design_list = best_many  # store the design results in the node
    assert len(node.design_list[0].sequence) == len(
        target
    ), f"Length of sequence {len(node.design_list[0].sequence)} does not match motif length {len(node.motif)}"

    # recursive design motifs of children
    for child in node.children:
        motif_design(child, seed)


def objective2score(objective, length=None):
    if f_obj == position_ed_pd_mfe:
        score = Decimal.log10(1 - objective)
    elif f_obj == position_ed_ned_mfe:
        score = (1 - objective) * length
    else:
        assert False, "objective function not supported"
    return score


def rescore(motif_node: MotifNode, sequence_combined: str):
    # sequence_combined = combine_sequences(motif_node.motif.dotbracket, sequence_list)
    # evaluate the combined sequence and get prob and dist
    if motif_node.parent is None:
        y_sub = motif_node.structure
        y_constrain = "." * len(sequence_combined)  # no constraint
    else:
        assert motif_node.motif.type == 'p'  # pseudo pair & outside boundary
        pair_boundary = motif_node.motif.children[0].pair
        print(f"pair_boundary: {pair_boundary}")
        # extract substructure
        y_sub = motif_node.structure[pair_boundary[0] : pair_boundary[1] + 1] 
        y_constrain = (
            "(" + "." * (len(sequence_combined) - 2) + ")"
        )  # constraint for the substructure, only the first and last characters are paired
    print(f"y_sub: {y_sub}, y_constrain: {y_constrain}")

    # prob_x_combined, suboptimal_structures = prob_subopt(sequence_combined, y_sub, y_constrain)
    _, obj_x_combined, suboptimal_structure_list = f_obj(sequence_combined, y_sub, y_constrain)
    score_real = objective2score(obj_x_combined, len(motif_node))  # calculate the folding probability
    dist_real = min(
        [struct_dist(y_sub, structure) for structure in suboptimal_structure_list]
    )  # calculate the distance of the best design
    if motif_node.parent is None and dist_real == 0 and len(suboptimal_structure_list) == 1:
        dist_real = -2  # unique MFE for global structure
    return score_real, dist_real


def combine_designs_kbest_pruning(
    node,
    seed=None
):  # get k-best designs by combining k-best designs from children, pruning according to score and distance
    from itertools import product
    assert node.design_list, "The node must have a design."
    design_self = node.design_list
    x_sample = design_self[0].sequence  # take the first design from the node
    print("x_sample:", x_sample)
    target = dotbracket2target(node.motif.dotbracket)
    print("target:", target)
    assert len(x_sample) == len_motif(target), f"Length of sequence {len(x_sample)} does not match motif length {len_motif(target)}"
    motif_list = [node.motif.dotbracket]
    for design in design_self:
        # design.score = Decimal.log10(design.prob)
        design.set_score(objective_name, len(node))
    if not node.children:  # leaf node
        return design_self
    else:
        ranks_all = []  # list of all combinations of designs from the children
        ranks_all.append(design_self)  # add the ranks of the node
        for child in node.children:
            motif_list.append(child.motif.dotbracket)
        design_list_of_children = [
            combine_designs_kbest_pruning(child, seed=seed) for child in node.children
        ]  # list of designs from the children
        for design_list_of_child in design_list_of_children:
            for design in design_list_of_child:
                assert design.score is not None, "Design score is None"
            ranks_all.append(design_list_of_child)
            print(f"designs_child samples: {design_list_of_child[:10]}")
        print("current motif:", node.motif.dotbracket)
        print("current target:", target)
        print("children motifs:")
        for motif_child in node.children:
            print(motif_child.motif.dotbracket)

        # use a heap to get the top k designs insead of sorting all combinations
        heap = []
        bad_designs = set()  # list of bad designs to be filtered out, current criterion is the re-evaluated distance is larger the the sum of distances plus 2
        design_set = set()  # to avoid duplicate designs
        size_prune = args.k_prune  # double the size of the heap to allow for pruning

        score_sum = sum(
            [rank[0].score for rank in ranks_all]
        )  # initial score for the best design
        dist_sum = sum(
            [rank[0].dist for rank in ranks_all]
        )
        score_best = score_sum  # use the sum of scores
        dist_best = dist_sum  # use the sum of distances

        sequence_combined = combine_sequences(node.motif.dotbracket, [rank[0].sequence for rank in ranks_all])
        if RESCORE:
            # combine designs and recalculate the score, score is log10(probability) or ensemble defect (unormalized)
            score_real, dist_real = rescore(node, sequence_combined)
            # whether a bad design
            if dist_real > dist_sum:
                bad_designs.add(sequence_combined)

            score_best = score_real  # use the recalculated score
            dist_best = dist_real  # use the recalculated distance

        seq_indices_best = [
            0 for _ in range(len(ranks_all))
        ]  # index of the best sequence in each rank

        # k-best designs
        k_best = []

        # bad designs
        # push the best design to heap; min heap with negative scores (lowest score -> highest probability)
        heap.append((-score_best, dist_best, seq_indices_best, sequence_combined))
        design_set.add(sequence_combined)
        while (
            heap and len(k_best) < size_prune
        ):  # while there are designs in the heap and we need more designs
            score_negative, dist_top, seq_indices, sequence_combined = heapq.heappop(
                heap
            )  # get the best design from the heap
            score_top = -score_negative  # convert the negative score back to positive
            k_best.append(
                (score_top, dist_top, sequence_combined)
            )  # add the design to the list of best designs

            # push neighboring designs to the heap
            if len(k_best) < size_prune:  # if we still need more designs
                for i in range(len(seq_indices)):
                    if seq_indices[i] + 1 < len(ranks_all[i]):
                        new_seq_indices = seq_indices.copy()
                        new_seq_indices[i] += 1  # increment the index of the i-th rank
                        # ranks_new = [ranks_all[j] for j in new_seq_indices]
                        # new_sequence_combined = combine_sequences(node.motif.dotbracket, [ranks_all[j][new_seq_indices[j]].sequence if j == i else ranks_all[j][seq_indices[j]].sequence for j in range(len(ranks_all))])
                        new_sequence_combined = combine_sequences(
                            node.motif.dotbracket,
                            [ranks_all[j][new_seq_indices[j]].sequence for j in range(len(ranks_all))],
                        )
                        if new_sequence_combined not in design_set:
                            new_score_sum = sum(
                                [ranks_all[j][new_seq_indices[j]].score for j in range(len(ranks_all))]
                            )
                            new_dist_sum = sum(
                                [ranks_all[j][new_seq_indices[j]].dist for j in range(len(ranks_all))]
                            )
                            if RESCORE:
                                # combine designs and recalculate the score
                                new_score_real, new_dist_real = rescore(node, new_sequence_combined)
                                if new_dist_real > new_dist_sum:
                                    bad_designs.add(new_sequence_combined)
                                    print(f"Bad design found: {new_sequence_combined} with dist_real {new_dist_real} > dist_sum {new_dist_sum}")
                                new_score = new_score_real
                                new_dist = new_dist_real
                            else:
                                new_score = new_score_sum
                                new_dist = new_dist_sum

                            # push the new design to the heap
                            # if new_sequence_combined not in design_set:
                            design_set.add(new_sequence_combined)
                            heapq.heappush(
                                heap, (-new_score, new_dist, new_seq_indices, new_sequence_combined)
                            )  # push the new design to the heap with negative score

        k_best_filtered = []
        if not RESCORE:
            # rescore and filter bad designs
            print("Rescoring and filtering bad designs...")
            for i in range(len(k_best)):
                score, dist, seq = k_best[i]
                print(f"k_best {i}: seq: {seq}, score: {score}, dist: {dist}")
                score_real, dist_real = rescore(node, seq)
                k_best[i] = (score_real, dist_real, seq)
                if dist_real > dist:
                    bad_designs.add(seq)
                    print(f"Bad design found: {seq} with dist_real {dist_real} > dist {dist}")

        # filter out the bad designs
        for score, dist, seq in k_best:
            if seq not in bad_designs:
                design = Design(sequence=seq, structure=node.motif.dotbracket)
                design.score = score
                design.dist = dist
                k_best_filtered.append(design)

        # print("k_best: ", k_best)
        # print('k_best_filtered:', k_best_filtered)
        if k_best_filtered:
            k_best_filtered.sort(key=lambda x: (-x.score, x.dist))
            print("return k_best_filtered")
            return k_best_filtered[:args.k_prune]
        else:
            if REDESIGN:
                # redesign
                k_redesign = 5
                k_best.sort(key=lambda x: (-x[0], x[1]))
                k_best = k_best[:k_redesign]
                initial_list = [seq for _, _, seq in k_best]
                step_redesign = 500
                if node.parent is None:
                    y_sub = node.structure
                else:
                    pair_boundary = node.motif.children[0].pair
                    y_sub = node.structure[pair_boundary[0] : pair_boundary[1] + 1]
                print(f"y_sub for redesign: {y_sub}")
                motif_y_sub = y_sub  # use the substructure as the motif for redesign
                if node.parent is None:  # redesign the whole motif
                    motif_y_sub = "5" + y_sub + "3"  # add 5' and 3' ends
                print(f"Initial list for redesign: {initial_list}")
                (
                    k_best,
                    log,
                    mfe_list,
                    umfe_list,
                    dist_list,
                    ned_best,
                    seq_list,
                    elapsed_time,
                    best_many,
                ) = samfeo(
                    motif_y_sub,
                    f_obj,
                    steps=step_redesign,
                    k=k_redesign,
                    t=args.t,
                    check_mfe=not args.nomfe,
                    sm=not args.nosm,
                    initial_list=initial_list,
                    seed=seed,
                )  # rna and ensemble defect
                # add scores
                for best in best_many:
                    # best.score = Decimal.log10(best.prob)
                    best.set_score(objective_name, len(y_sub))
                return best_many
            else:
                print("return k_best, no redesign")
                # print("k_best:", k_best)
                k_best_raw = []
                for score, dist, seq in k_best:
                    item = Design(sequence=seq, structure=node.motif.dotbracket)
                    # item.prob = Decimal(10) ** score
                    item.score = score
                    item.dist = dist
                    k_best_raw.append(item)
                k_best_raw.sort(key=lambda x: (-x.score, x.dist))
                return k_best_raw[:args.k_prune]
                # return k_best.sort(key=lambda x: (-x[0], x[1]))[:args.k_prune]  # else should do redesign, needs to be implemented later


def design_pipeline(name, y, seed=None):  # divide, conquer and combine
    start_time = time.time()
    print(f"test seed: {seed}")

    # divide
    motif_tree = decompose(y, selected_motifs)
    for motif_node in motif_tree.preorder():
        motif_node.structure = y

    # conquer
    motif_design(motif_tree, seed=seed)

    # combine
    design_list = combine_designs_kbest_pruning(
        motif_tree,
        seed=seed
    )

    # collect results
    # result_multi = dict()
    assert len(design_list) > 0, "design_list should not be empty"
    design_list.sort(key=lambda x: (-x.score, x.dist))  # sort by score and distance

    # redesign via samfeo_structure
    step_for_whole_structure = args.poststep
    k_for_whole_structure = 5 if step_for_whole_structure == 0 else 10
    # if step_for_whole_structure > 0:  # redesign the best designs for the whole structure
    initial_list = []
    print("initial_list for redesign:")
    for i in range(min(k_for_whole_structure, len(design_list))):
        initial_list.append(design_list[i].sequence)
        print(initial_list[-1])

    end_time = time.time()
    time_span = end_time - start_time
    func_objective = position_ed_pd_mfe_v0 if objective_name == "probability_defect" else position_ed_ned_mfe
    k_best, log, mfe_list, umfe_list, dist_best, ned_best, prob_best, elapsed_time = samfeo_structure(
        y,
        func_objective,
        steps=step_for_whole_structure,
        k=k_for_whole_structure,
        t=args.t,
        check_mfe=not args.nomfe,
        sm=not args.nosm,
        initial_list=initial_list,
    )
    elapsed_time += time_span
    return k_best, log, mfe_list, umfe_list, dist_best, ned_best, prob_best, elapsed_time


def test_design():
    # Prepare minimal global args/objective configuration when imported under pytest
    from types import SimpleNamespace
    global args, objective_name, f_obj, name_pair
    if 'args' not in globals():
        args = SimpleNamespace(
            object="pd",
            # parameters inherited from samfeo, should be fixed
            k=10,
            t=1,
            nomfe=False,
            nosm=False,
            stay=2000,
            init="cg",
            log=False,
            # parameters for divide-conquer-and-combine
            motif_path="data/easy_motifs.txt",
            motif_size=3,
            k_prune=20,
            step=1000,
            poststep=200,
            step_redesign=500,
            # parameters for parallel processing, no parallel here for testing
            worker_count=1,
            batch_size=1,
        )
        name_pair = args.init
        f_obj = position_ed_pd_mfe
        objective_name = "probability_defect"

        configure_global()

    y = ".....(.((((..(((((........)))))..(((.(((.(((((.....))))).((((....)))).))))))..)))).)...................."
    print("test structure:", y)

    k_best, log, mfe_list, umfe_list, dist_best, ned_best, prob_best, elapsed_time = design_pipeline("test", y, seed=seed_np)

    rna_best = max(k_best)
    obj = -rna_best.score
    print("ensemble objective: ", obj)
    seq = rna_best.seq
    assert seq == "AGAAACGGCCCAAGGACCGAACUGAAGGUCCAAGCGAGCCAGCGCCGAAAAGGCGCAGCCCUCAUGGGCAGGCCGCAAGGGUGGGAAAAAAAAAAGAAAAGAGA", f"seq is not right: {seq}"


def online_design(structure):
    k_best, log, mfe_list, umfe_list, dist_best, ned_best, prob_best, elapsed_time = design_pipeline("test", structure, seed=seed_np)
    print("k_best:", k_best)
    ned_best = float(ned_best[0]), ned_best[1]
    prob_best = float(1 - prob_best[0]), prob_best[1]
    results = {
                "target": structure,
                "mfe_list": mfe_list,
                "umfe_list": umfe_list,
                "ned_best": ned_best,
                "prob_best": prob_best,
                "dist_best": dist_best,
                "time": elapsed_time
            }
    print("design results:")
    for key, value in results.items():
        if "list" in key:
            # print count
            print(f"size of {key}: {len(value)}")
        else:
            print(f"{key}: {value}")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    id_random = random.randint(0, int(1e7))
    filename = f"results_{timestamp}_{id_random}.json"
    with open(filename, "w") as f:
        json.dump(results, f)
    print(f"Results saved to {filename}")


def configure_global():
    global seed_np
    seed_np = 2020
    np.random.seed(seed_np)

    global selected_motifs, path_motifs, size_motifs, f_obj, LOG, STAY, WORKER_COUNT, BATCH_SIZE
    path_motifs = args.motif_path
    size_motifs = args.motif_size
    selected_motifs = get_selected_motifs(path_motifs, size_motifs)
    print(f"Total {len(selected_motifs)} motifs loaded from {path_motifs} with size >= {size_motifs}")
    LOG = args.log
    STAY = args.stay
    WORKER_COUNT = args.worker_count
    BATCH_SIZE = WORKER_COUNT * 2 if args.batch_size is None else args.batch_size

    global name_pair, objective_name, name_input
    name_pair = args.init
    objective_name = None
    if args.object == "ned":  # normalized ensemble defect
        f_obj = position_ed_ned_mfe
        objective_name = "ensemble_defect"
    elif args.object == "pd":  # probability defect
        f_obj = position_ed_pd_mfe
        objective_name = "probability_defect"
    else:
        raise ValueError("the objective in not correct!")

    global META_DATA
    META_DATA['args'] = vars(args)
    META_DATA['objective_name'] = objective_name


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default="")
    parser.add_argument("-b", "--object", type=str, default="pd")
    # parameters inherited from samfeo, should be fixed
    parser.add_argument("--k", type=int, default=10)  # size of frontier for samfeo algorithm
    parser.add_argument("--t", type=float, default=1)  # temperature for mutation
    parser.add_argument("--stay", type=int, default=2000)
    parser.add_argument("--init", type=str, default="cg")
    parser.add_argument("--nomfe", action="store_true")
    parser.add_argument("--nosm", action="store_true")
    # parameters for divide-conquer-and-combine
    parser.add_argument("--k_prune", type=int, default=90)  # k for cube pruning
    parser.add_argument("--step", type=int, default=5000)  # number of step for designing each leaf node
    parser.add_argument("--poststep", type=int, default=0)  # number of steps to refine the root node
    parser.add_argument("--step_redesign", type=int, default=500)  # number of step for redesigning internal nodes when needed
    parser.add_argument("--motif_path", type=str, default="data/easy_motifs.txt")  # path to easy-to-design motifs
    parser.add_argument("--motif_size", type=int, default=3)  # minimum motif cardinality
    # parameters for parallel processing
    parser.add_argument("--worker_count", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=None)
    # other parameters
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--log", action="store_true")
    parser.add_argument("-o", "--online", action="store_true")

    args = parser.parse_args()
    print("args:")
    print(args)

    configure_global()

    if args.online:
        print("online mode:")
        for line in sys.stdin:
            target = line.strip()
            online_design(structure=target)

    if args.path:
        time_str = time.strftime("%Y%m%d_%H%M%S")
        OUTPUT_DIR = f"results/output_{time_str}"
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Design results will be saved to {OUTPUT_DIR}")
        path_meta = os.path.join(OUTPUT_DIR, "meta_data.json")
        META_DATA['runs'] = []
        name_input = os.path.splitext(os.path.basename(args.path))[0]
        for i in range(args.repeat):
            seed_np = 2020 + (i + args.start) * 2021
            np.random.seed(seed_np)
            suffix = f"{i+args.start}"
            path_output = design_parallel(
                args.path,
                name_input,
                f_obj,
                args.step,
                k=args.k,
                t=args.t,
                seed=seed_np,
            )
            META_DATA['runs'].append({'seed': seed_np, 'output_file': path_output})
            # save meta data
            with open(path_meta, 'w') as f_meta:
                json.dump(META_DATA, f_meta, indent=4)
        print(f"Meta data saved to {path_meta}")
