# FastDesign
Fast RNA Design via Motif-level Divide-Conquer-Combine and Structure-level Rival Search

## Dependencies
```
python 3.11
pip install -r requirements.txt
```

## Test
```
pytest main.py -s
```

### Optional: Enable Rival Search (for boosting MFE-based metrics)
To enable structure-level rival search, download and build the RNA-Undesign repository:

https://github.com/shanry/RNA-Undesign/tree/main

Then set the environment variable PATH_UNDESIGN:
```
export PATH_UNDESIGN=path/to/RNA-Undesign
```

## Main Parameters

The primary command-line parameters are:

- **`--path`**:  
  Path to the input file containing target RNA secondary structures, one structure per line.  
  Examples: `data/rnasolo764.txt`, `data/eterna100.txt`.

- **`--step`**:  
  Number of optimization steps for leaf-node (motif-level) design. Default: **5000**.

- **`--poststep`**:  
  Number of optimization steps for root-node (full-structure) refinement. A flexible parameter that balances quality and efficiency.
  Default values:  
  • **0** for the *fast* version  
  • **2500** for the *full* version

- **`--repeat`**:  
  Number of repeated experiments to run. Default: **1**.

- **`--worker_count`**:  
  Number of CPU cores to use for parallel execution.

- **`--batch_size`**:  
  Number of structures processed in parallel per batch. Must be **≥ `worker_count`**.

- **`--k_prune`**:  
  Beam size for cubic pruning. For each node, the top `k_prune` candidate designs are retained during search.

## Experiment Replication
To begin, navigate to the repository directory:
```
cd FastDesign/
```

### Fast Version
Run
```
python main.py --path data/rnasolo764.txt --step 5000 --poststep 0 --batch_size 200 --worker_count 20 --repeat 5
```
This runs 5 repeated experiments and produces 5 CSV files plus a `meta_data.json` file under: 
```results/output_timestamp/```

#### Evaluation (without rival search):
```
python utils/metrics.py --meta results/output_timestamp/meta_data.json --eval
```
This will evaluate and output the metrics without rival search.

#### Boosting MFE-based metrics using rival search

```
python utils/metrics.py --meta results/output_timestamp/meta_data.json -r
```
This generates improved MFE and uMFE results and saves them to:
```results/output_timestamp/rival_serach_results.csv```

### Full Version
Run
```
python main.py --path data/rnasolo764.txt --step 5000 --poststep 2500 --batch_size 200 --worker_count 20 --repeat 5
```
As with the fast version, this generates 5 CSV files and a `meta_data.json` file in: `results/output_timestamp/`

#### Evaluation (without rival search):
```
python utils/metrics.py --meta results/output_timestamp/meta_data.json --eval
```
This will evaluate and output the metrics without rival search.

#### Boosting MFE-based metrics using rival search

```
python utils/metrics.py --meta results/output_timestamp/meta_data.json -r
```
This generates improved MFE and uMFE results and saves them to:
```results/output_timestamp/rival_serach_results.csv```