"""
Concentration of generated explanations + reproducible example selection (Sect. 5.5).

Usage:
  python output_stats.py runs/ta_none_tr1_s1/generated.txt runs/ta_nohist_s1/generated.txt [more ...]
All files must come from DUSTER runs on the same test split (same instance order).
"""
import random
import sys
from collections import Counter


def read(path):
    gt, gen = [], []
    for block in open(path, encoding='utf-8').read().strip().split('\n\n'):
        lines = dict(l.split(': ', 1) if ': ' in l else (l.rstrip(':'), '') for l in block.split('\n'))
        gt.append(lines.get('GT', ''))
        gen.append(lines.get('GEN', ''))
    return gt, gen


runs = {p: read(p) for p in sys.argv[1:]}
for p, (gt, gen) in runs.items():
    c = Counter(gen)
    n = len(gen)
    print('\n==', p, '| n =', n, '| unique =', len(c))
    print('top-1 / top-10 / top-100 share: {:.1%} / {:.1%} / {:.1%}'.format(
        c.most_common(1)[0][1] / n, sum(v for _, v in c.most_common(10)) / n,
        sum(v for _, v in c.most_common(100)) / n))
    for s, v in c.most_common(5):
        print('  {:6.2%}  {}'.format(v / n, s))

# reproducible examples: random test instances (seed 0) where the generated sentences differ
paths = list(runs)
gt0 = runs[paths[0]][0]
n = min(len(runs[p][1]) for p in paths)
rnd = random.Random(0)
cand = [k for k in range(n) if len({runs[p][1][k] for p in paths}) > 1]
print('\n== examples (seed 0, among {} instances whose outputs differ)'.format(len(cand)))
for k in rnd.sample(cand, min(5, len(cand))):
    print('\n#{}  GT: {}'.format(k, gt0[k]))
    for p in paths:
        print('   {:<40} {}'.format(p.split('/')[-2], runs[p][1][k]))
