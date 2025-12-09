# FastDesign
Fast RNA Design via Motif-level Divide-Conquer-Combine and Structure-level Rival Search

## Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/shanry/FastDesign.git
    cd FastDesign
    ```

2.  **Install dependencies:**
    This project is developed using Python 3.11, and should also work for other Python versions.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Optional: Enable Rival Search**
    To enable structure-level rival search for boosting MFE-based metrics, download and build the [RNA-Undesign](https://github.com/shanry/RNA-Undesign/tree/main) repository.

    Then, set the `PATH_UNDESIGN` environment variable:
    ```bash
    export PATH_UNDESIGN=/path/to/RNA-Undesign
    ```

## Testing
```bash
pytest main.py -s
```

## Quick Start

You can run FastDesign directly from the command line.

**Fast Version:**
```bash
echo ".((((((((((..((((......((((((((.......)))))))).(((..(((((.....)))).)..)))))))....))))))))))" | python main.py --online
```

**Full Version:**
```bash
echo ".((((((((((..((((......((((((((.......)))))))).(((..(((((.....)))).)..)))))))....))))))))))" | python main.py --online --poststep 2500
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
The replicatable results (RNAsolo_full.zip, RNAsolo_fast.zip, Eterna100_full.zip, Eterna100_fast.zip) are available on [Google Drive](https://drive.google.com/drive/u/0/folders/12A2PhjEMSfbsqS9-matxi5O9RDWBVr2g).

### Fast Version
```bash
python main.py --path data/rnasolo764.txt --step 5000 --poststep 0 --batch_size 200 --worker_count 20 --repeat 5
```
This runs 5 repeated experiments and produces 5 CSV files plus a `meta_data.json` file under: 
```results/output_timestamp/```

#### Evaluation (without rival search):
```bash
python utils/metrics.py --meta results/output_timestamp/meta_data.json --eval
```
This will evaluate and output the metrics without rival search.

#### Boosting MFE-based metrics using rival search

```bash
python utils/metrics.py --meta results/output_timestamp/meta_data.json -r
```
This generates improved MFE and uMFE results and saves them to:
```results/output_timestamp/rival_serach_results.csv```

### Full Version
```bash
python main.py --path data/rnasolo764.txt --step 5000 --poststep 2500 --batch_size 200 --worker_count 20 --repeat 5
```
As with the fast version, this generates 5 CSV files and a `meta_data.json` file in: `results/output_timestamp/`

#### Evaluation (without rival search):
```bash
python utils/metrics.py --meta results/output_timestamp/meta_data.json --eval
```
This will evaluate and output the metrics without rival search.

#### Boosting MFE-based metrics using rival search

```bash
python utils/metrics.py --meta results/output_timestamp/meta_data.json -r
```
This generates improved MFE and uMFE results and saves them to:
```results/output_timestamp/rival_serach_results.csv```