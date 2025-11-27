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

## Experiment Replication
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