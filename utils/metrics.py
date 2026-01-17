"""This script reads the meta_data.json output from main.py,
summarizes ensemble metrics such as equilibrium probability and ensemble defect,
counts the number of solved puzzles / structures by MFE and uMFE (unique Minimum Free Energy) criteria,
and prints various statistics about the solutions.
Example usage:
python metrics.py --meta PATH_TO_META.json --eval

It can also perform rival search to boost MFE solutions:
python metrics.py --meta PATH_TO_META.json --rival_search
"""
import os
import json
import argparse 
import subprocess

import numpy as np
import pandas as pd

from decimal import Decimal, getcontext
getcontext().prec = 200


def rival_search(x, y, z, n_threads=None):
    input_std = f"{x}\n{y}\n{z}"
    env = os.environ.copy()
    # n_threads = n_thread_global
    if n_threads is None:
        n_threads = os.cpu_count() or 1
    env["OMP_NUM_THREADS"] = str(n_threads)
    env["OMP_DYNAMIC"] = "FALSE"         # Disable dynamic adjustment
    env["OMP_PROC_BIND"] = "TRUE"        # Bind threads to cores
    path_undesign = os.environ.get("PATH_UNDESIGN")
    binary_undesign = os.path.join(path_undesign, "bin/main")
    print("command:\n", " ".join([binary_undesign, 
                      "--alg", "2", 
                      "--max_enum", "100000000",
                      "--max_constraint", "100000",
                      "--n_sample", "500", 
                      "--max_rival", "100"]))
    print("input:\n", input_std.strip(), sep="")
    result = subprocess.run([binary_undesign, 
                             "--alg", "2", 
                             "--max_enum", "100000000",
                             "--max_constraint", "100000",
                             "--n_sample", "500", 
                             "--max_rival", "100"], 
                             input=input_std, 
                             stdout=subprocess.PIPE, 
                             text=True,
                             env=env)
    
    output_lines = result.stdout.strip().split('\n')
    last_line = output_lines[-1] if output_lines else None
    return last_line


def read_dataframes(path_list):
    df_list = []
    print()
    print("Reading files ..")
    print("-------------------------------------")
    for filename in path_list:
        print(filename, end=" ")
        df_onerun = pd.read_csv(filename)
        df_list.append(df_onerun)
        print(len(df_onerun))
    print("-------------------------------------")
    print("Done reading files.")
    return df_list


def evaluate(df_list):
    assert all([len(df) == len(df_list[0]) for df in df_list])
    num_puzzles = len(df_list[0])
    print(f"num_puzzles: {num_puzzles}")
    data = []
    matrix_mfe = np.zeros((num_puzzles, len(df_list)), dtype=int)
    matrix_umfe = np.zeros((num_puzzles, len(df_list)), dtype=int)
    # lens_structures = []
    for i in range(num_puzzles):
        structure = df_list[0].structure.iloc[i]
        # lens_structures.append(len(structure))
        puzzle_name = df_list[0].puzzle_name.iloc[i] # if "puzzle_name" in df_list[0].columns else f"Puzzle {i+1}"
        count_soved_by_mfe = 0
        count_soved_by_umfe = 0
        objective_list = []
        dist_best_list = []
        ned_best_list = []
        prob_best_list = []
        time_list = []
        for j, df_one in enumerate(df_list):
            mfe_list = eval(df_one.mfe_list.iloc[i])
            umfe_list = eval(df_one.umfe_list.iloc[i])
            if mfe_list:
                count_soved_by_mfe += 1
                matrix_mfe[i, j] = 1
            if umfe_list:
                count_soved_by_umfe += 1
                matrix_umfe[i, j] = 1
            objective_list.append((df_one.objective.iloc[i]))
            dist_best_list.append(eval(df_one.dist_best.iloc[i]))
            ned_best_list.append(eval(df_one.ned_best.iloc[i]))
            if 'prob_best' in df_one.columns:
                prob_best_list.append(eval(df_one.prob_best.iloc[i]))
            time_list.append(df_one.time.iloc[i])
        dist_best, _ = min(dist_best_list)
        ned_best, _ = min(ned_best_list)
        if 'prob_best' in df_one.columns:
            prob_best, _ = min(prob_best_list)
        time_mean = np.mean(time_list)
        if dist_best < 0:
            dist_best = 0
        if 'prob_best' in df_one.columns:
            data.append(
                [
                    i,
                    puzzle_name,
                    structure,
                    min(objective_list),
                    np.mean(objective_list),
                    count_soved_by_mfe,
                    count_soved_by_umfe,
                    count_soved_by_mfe > 0,
                    count_soved_by_umfe > 0,
                    dist_best,
                    ned_best,
                    prob_best,
                    time_mean,
                ]
            )
        else:
            data.append(
                [
                    i,
                    puzzle_name,
                    structure,
                    min(objective_list),
                    np.mean(objective_list),
                    count_soved_by_mfe,
                    count_soved_by_umfe,
                    count_soved_by_mfe > 0,
                    count_soved_by_umfe > 0,
                    dist_best,
                    ned_best,
                    time_mean,
                ]
            )
    if 'prob_best' in df_one.columns:
        df_joint = pd.DataFrame(
            data,
            columns=(
                "index",
                "puzzle_name",
                "structure",
                "objective",
                "objective_mean",
                "count_solved_mfe",
                "count_solved_umfe",
                "is_solved_mfe",
                "is_solved_umfe",
                "dist_best",
                "ned_best",
                "prob_best",
                "time_mean",
            ),
        )
    else:
        df_joint = pd.DataFrame(
            data,
            columns=(
                "index",
                "puzzle_name",
                "structure",
                "objective",
                "objective_mean",
                "count_solved_mfe",
                "count_solved_umfe",
                "is_solved_mfe",
                "is_solved_umfe",
                "dist_best",
                "ned_best",
                "time_mean",
            ),
        )
    print()
    print("MFE metrics:")
    print("-------------------------------------")
    print(f"solved by  mfe: {df_joint.is_solved_mfe.sum()}")
    print(f"solved by umfe: {df_joint.is_solved_umfe.sum()}")
    print(
        f"  mfe mean and std: {matrix_mfe.sum(axis=0).mean():.1f} {matrix_mfe.sum(axis=0).std():.1f}"
    )
    print(
        f" umfe mean and std: {matrix_umfe.sum(axis=0).mean():.1f} {matrix_umfe.sum(axis=0).std():.1f}"
    )
    print()

    print("best distance:")
    print("-------------------------------------")
    print(f"mean: {df_joint.dist_best.mean():.2f} std: {df_joint.dist_best.std():.2f}")
    print()

    print("best ned:")
    print("-------------------------------------")
    print(f"mean: {df_joint.ned_best.mean():.4f}") # std: {df_joint.prob_best.std():.4f}")
    print()

    if "prob_best" in df_one.columns:
        print("best prob:")
        print("-------------------------------------")
        print(f"mean: {1 - df_joint.prob_best.mean():.4f}") # std: {df_joint.ned_best.std():.4f}")
        print()


    print("Objective statistics:")
    print("-------------------------------------")
    print(f"objective arithmic mean: {df_joint.objective.mean():.4f}")
    print(f"1 - obj.  arithmic mean: {1 - df_joint.objective.mean():.4f}")
    if VERBOSE:
        print("probability list (1 - objective):", [f"{1 - x:.3f}" for x in df_joint.objective.tolist()])
    print()

    objective_list = df_joint.objective
    # geometric mean and std
    geometric_mean = np.exp(np.log(objective_list).mean())
    # geometric_std = np.exp(np.log(prob_list).std())
    if geometric_mean > 1e-4:
        print(f"objective geometric mean: {geometric_mean:.4f}")
    else:  # scientific notation for very small mean value
        print(f"objective geometric mean: {geometric_mean:.4e}")

    objective_complement_list = 1 - df_joint.objective
    geometric_mean_complement = np.exp(np.log(objective_complement_list).mean())
    if geometric_mean_complement > 1e-4:
        print(f"1 - obj.  geometric mean: {geometric_mean_complement:.4f}")
    else:  # scientific notation for very small mean value
        print(f"1 - obj.  geometric mean: {geometric_mean_complement:.4e}")
    print()

    print("time statistics:")
    print("-------------------------------------")
    if VERBOSE:
        print("time list (precision .1f):", [f"{x:.1f}" for x in df_joint.time_mean.tolist()])
    print(f"time cost ave: {df_joint.time_mean.mean():.2f}")
    print(f"time cost std: {df_joint.time_mean.std():.2f}")
    print()

    # save the joint dataframe to a CSV file
    df_joint.to_csv("mfe_counts.csv", index=False)
    print("Joint metrics saved to mfe_counts.csv")


def mfe_booster(df_list, num_threads=None, output_dir="."):
    def argmin_dist(x, y):
        from utils.vienna import subopt
        from utils.structure import struct_dist
        y_mfe_list = subopt(x)["tuples_of_energy_and_structure"]
        if len(y_mfe_list) == 1:
            _, y_mfe = y_mfe_list[0]
            assert y_mfe != y, f"MFE structure matches target structure, should have been an uMFE solution, {y}, {x}"
            return y_mfe, struct_dist(y, y_mfe)
        y_mfe_list = [item for item in y_mfe_list if item[1] != y]
        y_best = y_mfe_list[0][1]
        d_best = struct_dist(y, y_best)
        for _, y_mfe in y_mfe_list[1:]:
            d = struct_dist(y, y_mfe)
            if d < d_best:
                d_best = d
                y_best = y_mfe
        return y_best, d_best
    
    assert all([len(df) == len(df_list[0]) for df in df_list])
    num_puzzles = len(df_list[0])
    print(f"num_puzzles: {num_puzzles}")
    data = []
    for i in range(num_puzzles):
        # print(f"Processing puzzle {i+1}/{num_puzzles}")
        puzzle_name = df_list[0].puzzle_name.iloc[i] if "puzzle_name" in df_list[0].columns else f"Puzzle {i+1}"
        structure = df_list[0].structure.iloc[i]
        structure = df_list[0].structure.iloc[i]
        dist_best_raw_list = []
        dist_best_list = []
        for j, df_one in enumerate(df_list):
            dist_best_raw = eval(df_one.dist_best.iloc[i])
            dist_best_raw_list.append(dist_best_raw)
        dist_best_raw = min(dist_best_raw_list)
        dist_raw, x = dist_best_raw
        y_rival = ""
        time_cost = 0.0
        x_mfe = None
        x_umfe = None
        if dist_raw < 0:
            x_mfe = x
            x_umfe = x
            data.append([puzzle_name, structure, x, y_rival, dist_raw, x_mfe, x_umfe, time_cost])
            continue
        if dist_raw == 0:
            x_mfe = x
        for j, df_one in enumerate(df_list):
            x = df_one.rna.iloc[i]
            y_mfe, dist_mfe = argmin_dist(x, structure)
            dist_best_list.append((dist_mfe, x, y_mfe))
        dist_best, x, y_mfe = min(dist_best_list)
        assert dist_best >= dist_raw, f"Distance mismatch: {dist_best} >= {dist_raw}"
        y_rival = y_mfe
        result_of_rival_search_jstr = rival_search(x, structure, y_rival, n_threads=num_threads)
        print("Rival search result (json):", result_of_rival_search_jstr)
        result_of_rival_search = json.loads(result_of_rival_search_jstr)
        x_umfe = result_of_rival_search.get('umfe', x_umfe)
        x_mfe = x_umfe if x_umfe is not None else result_of_rival_search.get('mfe', x_mfe)
        time_cost = result_of_rival_search.get('time')
        if x_mfe or x_umfe:
            dist_best = 0
        if x_mfe is not None:
            assert len(x_mfe) == len(structure), f"Length mismatch between RNA and structure: {len(x_mfe)} != {len(structure)}"
        if x_umfe is not None:
            assert len(x_umfe) == len(structure), f"Length mismatch between RNA and structure: {len(x_umfe)} != {len(structure)}"
        data.append([puzzle_name, structure, x, y_rival, dist_best, x_mfe, x_umfe, time_cost])
    df_mfes = pd.DataFrame(data, columns=("puzzle_name", "structure", "x", "y_rival", "dist_best", "mfe_solution", "umfe_solution", "time"))
    # print metrics of mfe and umfe
    print("number of mfe solutions found:", len(df_mfes[df_mfes.mfe_solution.notnull()]))
    print("number of umfe solutions found:", len(df_mfes[df_mfes.umfe_solution.notnull()]))
    # save the dataframe to a CSV file
    path_csv = os.path.join(output_dir, "rival_search_results.csv")
    df_mfes.to_csv(path_csv, index=False)
    print(f"Boosted MFE evaluation results are saved to {path_csv}")


def main(args):
    if not args.meta or os.path.exists(args.meta) is False:
        print("Please provide valid meta data file using --meta")
        return
    meta_data = json.load(open(args.meta, 'r'))
    path_list = [run['output_file'] for run in meta_data['runs']]
    if args.eval:
        print("Counting MFE statistics ..")
        df_list = read_dataframes(path_list)
        evaluate(df_list)

    if args.rival_search:
        print("Performing rival search to boost MFE solutions ..")
        output_dir = os.path.dirname(args.meta)
        df_list = read_dataframes(path_list)
        mfe_booster(df_list, num_threads=args.num_threads, output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--meta", type=str, default="")
    parser.add_argument("-e", "--eval", action="store_true", help="get mfe statistics")
    parser.add_argument("--verbose", action="store_true", help="verbose output")
    parser.add_argument("-r", "--rival_search", action="store_true", help="perform rival search to boost mfe solutions")
    parser.add_argument("--num_threads", type=int, default=8, help="number of threads to use")
    args = parser.parse_args()
    print(f"args: {args}")
    VERBOSE = args.verbose

    main(args)
