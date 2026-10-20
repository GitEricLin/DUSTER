"""
Average results.json over seeds:  python aggregate_results.py runs/clothing_pred_s*
"""
import json
import statistics
import sys
from glob import glob

paths = [p for arg in sys.argv[1:] for p in glob(arg)]
rows = [json.load(open(p.rstrip('/') + '/results.json')) for p in paths]
keys = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and k != 'seed']
print('runs:', len(rows))
for k in keys:
    vals = [r[k] for r in rows if k in r]
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    print('{:20s} {:9.4f} ± {:.4f}'.format(k, statistics.mean(vals), sd))
