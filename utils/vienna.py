import os
import sys
import numpy as np
from utils.structure import extract_pairs

from decimal import Decimal, getcontext

import RNA

getcontext().prec = 200


def base_pair_probs(seq, sym=False, scale=True, energy=None):
    fc = RNA.fold_compound(seq)
    if scale:
        if energy is not None:
            fc.exp_params_rescale(energy)
        else:
            _, mfe = fc.mfe()
            fc.exp_params_rescale(mfe)
    fc.pf()
    bpp = np.array(fc.bpp())[1:, 1:]
    if sym:
        bpp += bpp.T
        unpair = 1 - np.sum(bpp, axis=1)
        bpp[range(len(bpp)), range(len(bpp))] = unpair
    return bpp


def ensemble_defect(seq, ss, scale=True):
    fc = RNA.fold_compound(seq)
    if scale:
        energy = fc.eval_structure(ss)
        fc.exp_params_rescale(energy)
    fc.pf()
    fc.bpp()
    ed = fc.ensemble_defect(ss)
    return ed


def position_defect(seq, ss, scale=True):
    energy = None
    if scale:
        fc = RNA.fold_compound(seq)
        energy = fc.eval_structure(ss)
    bpp = base_pair_probs(seq, sym=True, scale=scale, energy=energy)
    pairs = extract_pairs(ss)
    defect_pos = [1 - bpp[i, j] for i, j in enumerate(pairs)]
    return defect_pos


def position_defect_mfe(seq, ss, scale=True):
    fc = RNA.fold_compound(seq)
    tuples_of_energy_and_structure = subopt(seq)["tuples_of_energy_and_structure"]
    mfe = tuples_of_energy_and_structure[0][0]
    suboptimal_structure_list = [
        structure for (_, structure) in tuples_of_energy_and_structure
    ]
    if scale:
        # energy = fc.eval_structure(ss)
        fc.exp_params_rescale(mfe)
    fc.pf()
    bpp = np.array(fc.bpp())[1:, 1:]
    sym = True
    if sym:
        bpp += bpp.T
        unpair = 1 - np.sum(bpp, axis=1)
        bpp[range(len(bpp)), range(len(bpp))] = unpair
    pairs = extract_pairs(ss)
    defect_pos = [1 - bpp[i, j] for i, j in enumerate(pairs)]

    return defect_pos, suboptimal_structure_list


def position_ed_pd(seq, ss, scale=True):
    fc = RNA.fold_compound(seq)
    if scale:
        # _, mfe = fc.mfe()
        energy = fc.eval_structure(ss)
        fc.exp_params_rescale(energy)
    fc.pf()
    bpp = np.array(fc.bpp())[1:, 1:]
    sym = True
    if sym:
        bpp += bpp.T
        unpair = 1 - np.sum(bpp, axis=1)
        bpp[range(len(bpp)), range(len(bpp))] = unpair
    pairs = extract_pairs(ss)
    defect_pos = [1 - bpp[i, j] for i, j in enumerate(pairs)]
    pr = fc.pr_structure(ss)
    pd = 1.0 - pr
    return defect_pos, pd


def position_ed_pd_mfe(seq, ss, constraint=None, scale=True):
    fc = RNA.fold_compound(seq)
    if constraint is not None:
        opts = RNA.CONSTRAINT_DB_DEFAULT | RNA.CONSTRAINT_DB_ENFORCE_BP
        fc.hc_add_from_db(constraint, opts)
    tuples_of_energy_and_structure = subopt(seq, constraint)[
        "tuples_of_energy_and_structure"
    ]
    mfe = tuples_of_energy_and_structure[0][0]
    suboptimal_structure_list = [
        structure for (_, structure) in tuples_of_energy_and_structure
    ]
    if scale:
        fc.exp_params_rescale(mfe)
    fc.pf()
    bpp = np.array(fc.bpp())[1:, 1:]
    sym = True
    if sym:
        bpp += bpp.T
        unpair = 1 - np.sum(bpp, axis=1)
        bpp[range(len(bpp)), range(len(bpp))] = unpair
    pairs = extract_pairs(ss)
    positional_ensemble_defect = [1 - bpp[i, j] for i, j in enumerate(pairs)]
    pr = fc.pr_structure(ss)
    prob_defect = Decimal(1.0) - Decimal(str(pr))  # probability defect
    return positional_ensemble_defect, prob_defect, suboptimal_structure_list


def position_ed_pd_mfe_v0(seq, ss, scale=True):
    fc = RNA.fold_compound(seq)
    ss_mfe_list = subopt(seq)["tuples_of_energy_and_structure"]  # a list of (mfe, structure) tuples
    mfe = ss_mfe_list[0][0]  # minimum free energy
    ss_list = [ss_mfe[1] for ss_mfe in ss_mfe_list]
    if scale:
        fc.exp_params_rescale(mfe)
    fc.pf()
    bpp = np.array(fc.bpp())[1:, 1:]
    sym = True
    if sym:
        bpp += bpp.T
        unpair = 1 - np.sum(bpp, axis=1)
        bpp[range(len(bpp)), range(len(bpp))] = unpair
    pairs = extract_pairs(ss)
    defect_pos = [1 - bpp[i, j] for i, j in enumerate(pairs)]
    pr = fc.pr_structure(ss)
    pd = -pr  # negative probability, should be 1 - pr, but to avoid precision issue, use -pr here and convert to Decimal later
    pd = Decimal(str(pd))
    return defect_pos, pd, ss_list


def prob_subopt(seq, structure, constraint=None, scale=True):
    fc = RNA.fold_compound(seq)
    if constraint is not None:
        opts = RNA.CONSTRAINT_DB_DEFAULT | RNA.CONSTRAINT_DB_ENFORCE_BP
        fc.hc_add_from_db(constraint, opts)
    tuples_of_energy_and_structure = subopt(seq, constraint)[
        "tuples_of_energy_and_structure"
    ]
    mfe = tuples_of_energy_and_structure[0][0]
    suboptimal_structure_list = [
        structure for (_, structure) in tuples_of_energy_and_structure
    ]
    if scale:
        fc.exp_params_rescale(mfe)
    fc.pf()
    pr = fc.pr_structure(structure)
    return Decimal(str(pr)), suboptimal_structure_list


def position_ed_ned_mfe(seq, ss, constraint=None, scale=True):
    fc = RNA.fold_compound(seq)
    if constraint is not None:
        opts = RNA.CONSTRAINT_DB_DEFAULT | RNA.CONSTRAINT_DB_ENFORCE_BP
        fc.hc_add_from_db(constraint, opts)
    tuples_of_energy_and_structure = subopt(seq, constraint)[
        "tuples_of_energy_and_structure"
    ]
    mfe = tuples_of_energy_and_structure[0][0]
    suboptimal_structure_list = [
        structure for (_, structure) in tuples_of_energy_and_structure
    ]
    if scale:
        fc.exp_params_rescale(mfe)
    fc.pf()
    bpp = np.array(fc.bpp())[1:, 1:]
    sym = True
    if sym:
        bpp += bpp.T
        unpair = 1 - np.sum(bpp, axis=1)
        bpp[range(len(bpp)), range(len(bpp))] = unpair
    pairs = extract_pairs(ss)
    positional_ensemble_defect = [1 - bpp[i, j] for i, j in enumerate(pairs)]
    ned = float(np.mean(positional_ensemble_defect))
    ned = Decimal(str(ned))
    return positional_ensemble_defect, ned, suboptimal_structure_list


def energy(seq, ss):
    fc = RNA.fold_compound(seq)
    return fc.eval_structure(ss)


def mfe(seq, constraint=None):
    fc = RNA.fold_compound(seq)
    if constraint is not None:
        opts = RNA.CONSTRAINT_DB_DEFAULT | RNA.CONSTRAINT_DB_ENFORCE_BP
        fc.hc_add_from_db(constraint, opts)
    ss = fc.mfe()
    return ss


def prob(seq, ss, scale=True, constraint=None):
    fc = RNA.fold_compound(seq)
    if scale:
        energy = fc.eval_structure(ss)
        fc.exp_params_rescale(energy)
    if constraint is not None:
        opts = RNA.CONSTRAINT_DB_DEFAULT | RNA.CONSTRAINT_DB_ENFORCE_BP
        fc.hc_add_from_db(constraint, opts)
    fc.pf()
    pr = fc.pr_structure(ss)
    return pr


def partition_function(seq, scale=True, constraint=None):
    fc = RNA.fold_compound(seq)
    if constraint is not None:
        opts = RNA.CONSTRAINT_DB_DEFAULT | RNA.CONSTRAINT_DB_ENFORCE_BP
        fc.hc_add_from_db(constraint, opts)
    if scale:
        mfe = fc.mfe()[1]
        fc.exp_params_rescale(mfe)
    pf = fc.pf()
    return pf


def prob_defect(seq, ss):
    return 1 - prob(seq, ss)


# Print a subopt result as FASTA record
def print_subopt_result(structure, energy, data):
    if structure is not None:
        # print(">subopt {:d}".format(data['counter']))
        # print("{}\n{} [{:6.2f}]".format(data['sequence'], structure, energy))
        data["tuples_of_energy_and_structure"].append((energy, structure))
        # increase structure counter
        data["counter"] = data["counter"] + 1


def subopt(seq, constraint=None, e=0):
    result = {"counter": 0, "sequence": seq, "tuples_of_energy_and_structure": []}
    fc = RNA.fold_compound(seq)
    if constraint is not None:
        opts = RNA.CONSTRAINT_DB_DEFAULT | RNA.CONSTRAINT_DB_ENFORCE_BP
        fc.hc_add_from_db(constraint, opts)
    fc.subopt_cb(e, print_subopt_result, result)
    result["tuples_of_energy_and_structure"].sort()
    return result


if __name__ == "__main__":
    if len(sys.argv) > 1:
        rna = sys.argv[1]
    else:
        rna = "AAAAAAAAAACCGCAAAAGCGGGGCCUAAUGGCCGCGGAAUCCGC"
    ss_mfe = mfe(rna)[0]
    bpp = base_pair_probs(rna, sym=True)
    defect = ensemble_defect(rna, ss_mfe)
    defect_pos = position_defect(rna, ss_mfe)
    pr = prob(rna, ss_mfe)
    rna4subopt = "AAUAGGUUUGGUCCUAGCCUUUCUAUUAACUCUUAGUAGGAUUACACAUGCAAGCAUCCCCGCCCCAGUGAGUCACCCUCUAAAUCACCACGAUCAAAAGGAACAAGCAUCAAGUACGCAGAAAUGCAGCUCAAAACGCUUAGCCUAGCCACACCCCCACGGGAGACAGCAGUGAUAAACCUUUAGCAAUAAACGAAAGUUUAACUAAGCCAUACUAACCCCAGGGUUGGUCAAUUUCGUGCCAGCCACCGCGGUCACACGAUUAACCCAAGCCAAUAGAAAUCGGCGUAAAGAGUGUUUUAGAUCAAUCCCCCAAUAAAGCUAAAAUUCACCUGAGUUGUAAAAAACUCCAGCUGAUAUAAAAUAAACUACGAAAGUGGCUUUAAUAUAUCUGAACACACAAUAGCUAGGACCCAAACUGGGAUUAGAUACCCCACUAUGCCUAGCCCUAAACUUCAACAGUUAAAUUAACAAGACUGCUCGCCAGAACACUACGAGCCACAGCUUAAAACUCAAAGGACCUGGCGGUGCUUCACAUCCUUCUAGAGGAGCCUGUUCUGUAAUCGAUAAACCCCGAUCAACCUCACCACCUCUUGCUCAGCCUAUAUACCGCCAUCUUCAGCAAACCCUGACGAAGGCCACAAAGUAAGCACAAGUACCCACGUAAAGACGUUAGGUCAAGGUGUAGCCCAUGAGGUGGCAAGAAAUGGGCUACAUUUUCUACUUCAGAAAACUACGAUAACCCUUAUGAAACCUAAGGGUAGAAGGUGGAUUUAGCAGUAAACUAAGAGUAGAGUGCUUAGUUGAACAGGGCCCUGAAGCGCGUACACACCGCCCGUCACCCUCCUCAAGUAUACUUCAAAGGACAUUUAACUAAAACCCCUACGCAUCUAUAUAGAGGAGAUAAGUCGUAACAUGGUAAGUGUACUGGAAAGUGCACUUGGACGAAC"
    result_suboptimal = subopt(rna4subopt, None)
    defect_pos_2, pr_2 = position_ed_pd(rna, ss_mfe)
    print("rna:", rna)
    print("mfe:", ss_mfe)
    print("prb:", pr)
    print("pr2:", pr_2)
    print("bpp:", bpp)
    print("Vinenna NED:", defect)
    print("Scratch NED:", sum(defect_pos) / len(defect_pos))
    print("Position ED:", sum(defect_pos_2) / len(defect_pos_2))
    # print("subopt:", result_suboptimal)

    x = "UAAUGGCCGCGGAAUCCGC"
    c = "(.................)"

    pf = partition_function(x, scale=False, constraint=None)

    print(len(x), len(c))
    print("Partition function:", pf)

    print("----------------------")

    x = "GGACCCGAAAGGCGCAC"
    x = "GGCGUAAAAAUGUCCAC"
    y = "((.(((....)).)).)"
    c = "(....(xxxx).....)"
    # c = "(...............)"
    print("Sequence:", x)
    print("Structure:", y)

    energy_x_y = energy(x, y)
    print("Energy:", energy_x_y)

    pr = prob(x, y, scale=False, constraint=c)
    print("Probability:", pr)

    subopt_result = subopt(x, constraint=c, e=200)
    print("Subopt result:", len(subopt_result["tuples_of_energy_and_structure"]))

    pf = partition_function(x, scale=False, constraint=c)
    print("Partition function with constraint:", pf)
    
    # c = "(...............)"
    # pf = partition_function(x, scale=False, constraint=c)
    # print("Partition function with constraint:", pf)
