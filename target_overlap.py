"""
Target-overlap analysis for Sect. 5.5 / Table 5 (reviewer item 3E).

Question: does the leaky model reproduce DISTINCTIVE content of the reference sentence more often
than the as-of model?  Instead of judging from two hand-picked examples, this script
  (1) measures, over ALL test instances, how often each model's output contains at least one
      distinctive reference token, and tests leaky vs. as-of with an exact McNemar test;
  (2) selects Table 5 examples with a fixed, reportable criterion and a fixed random seed.

A reference token is "distinctive" if it is NOT
  - a stopword,
  - the aspect word fed to the model (ASPECT_IN; in the Oracle protocol this is the gold aspect),
  - among the top-K most frequent tokens of the TRAINING sentences (document frequency),
  - <unk> or non-alphabetic.
Results are reported for several K so the conclusion does not hinge on one threshold.

generated.txt files must come from runs on the same test split (main.py writes test instances in a
fixed order), e.g. ta_gold_tr1_s1, ta_leaky_gold_tr1_s1, ta_nohist_gold_tr1_s1.

Usage (Colab, from the DUSTER_V2 folder):
  python target_overlap.py \
     --asof   runs/ta_gold_tr1_s1/generated.txt \
     --leaky  runs/ta_leaky_gold_tr1_s1/generated.txt \
     --nohist runs/ta_nohist_gold_tr1_s1/generated.txt \
     --data_path /content/drive/MyDrive/TripAdvisor_Dataset/reviews_time.pickle \
     --index_dir /content/drive/MyDrive/TripAdvisor_Dataset/temporal/ \
     --out runs/target_overlap.json
"""
import argparse
import json
import math
import os
import pickle
import random
from collections import Counter

STOP = set("""
a an the and or but if then so of to in on at by for with from as into onto about over under
is are was were be been being am it its this that these those there here
i we you he she they me us him her them my our your his their mine ours yours
very really quite too also just not no nor only than more most much many some any all both each
few other such own same can will would could should may might must do does did done
have has had having get got one ones s t
""".split())


def read_generated(path):
    recs, cur = [], {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                if cur:
                    recs.append(cur)
                    cur = {}
                continue
            key, _, val = line.partition(':')
            if key in ('GT', 'CTX', 'GEN', 'ASPECT_IN'):
                cur[key] = val.strip()
    if cur:
        recs.append(cur)
    return recs


def train_doc_freq(data_path, index_dir):
    reviews = pickle.load(open(data_path, 'rb'))
    with open(os.path.join(index_dir, 'train.index')) as f:
        train_idx = [int(x) for x in f.readline().split()]
    df = Counter()
    for i in train_idx:
        df.update(set(reviews[i]['template'][2].split()))
    return df


def distinctive(gt, aspect, frequent):
    toks = set(gt.split())
    return {w for w in toks
            if w.isalpha() and w not in STOP and w != aspect and w not in frequent and w != '<unk>'}


def log_binom_cdf(k, n):
    """log P(X <= k), X ~ Bin(n, 0.5)"""
    if n == 0:
        return 0.0
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2)
            for i in range(0, k + 1)]
    m = max(logs)
    return m + math.log(sum(math.exp(x - m) for x in logs))


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * math.exp(log_binom_cdf(min(b, c), n)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--asof', required=True)
    ap.add_argument('--leaky', required=True)
    ap.add_argument('--nohist', default=None)
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--index_dir', required=True)
    ap.add_argument('--topk', type=int, nargs='+', default=[200, 500, 1000])
    ap.add_argument('--select_k', type=int, default=500, help='K used for the example-selection criterion')
    ap.add_argument('--n_examples', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='target_overlap.json')
    args = ap.parse_args()

    runs = {'asof': read_generated(args.asof), 'leaky': read_generated(args.leaky)}
    if args.nohist:
        runs['nohist'] = read_generated(args.nohist)
    n = len(runs['asof'])
    for name, r in runs.items():
        assert len(r) == n, '{} has {} instances, as-of has {}'.format(name, len(r), n)
        assert all(a['GT'] == b['GT'] for a, b in zip(runs['asof'], r)), name + ': GT order differs'
    print('test instances:', n)

    df = train_doc_freq(args.data_path, args.index_dir)
    ranked = [w for w, _ in df.most_common()]
    out = {'n_test': n, 'runs': {'asof': args.asof, 'leaky': args.leaky, 'nohist': args.nohist},
           'definition': 'distinctive = reference token that is alphabetic, not a stopword, not the aspect '
                         'fed to the model, and not among the top-K most frequent training tokens',
           'by_topk': {}}

    hits_at = {}
    for K in args.topk:
        frequent = set(ranked[:K])
        res = {}
        hits = {}
        for name, r in runs.items():
            n_with, n_hit, rec_sum, exact = 0, 0, 0.0, 0
            h = []
            for x in r:
                exact += int(x['GEN'] == x['GT'])
                d = distinctive(x['GT'], x.get('ASPECT_IN', ''), frequent)
                if not d:
                    h.append(None)
                    continue
                g = set(x['GEN'].split())
                inter = d & g
                n_with += 1
                n_hit += int(bool(inter))
                rec_sum += len(inter) / len(d)
                h.append(bool(inter))
            hits[name] = h
            res[name] = {'instances_with_distinctive_tokens': n_with,
                         'hit_rate': n_hit / max(1, n_with),
                         'mean_distinctive_recall': rec_sum / max(1, n_with),
                         'exact_match_rate': exact / n}
        b = sum(1 for a, l in zip(hits['asof'], hits['leaky']) if a is not None and l and not a)
        c = sum(1 for a, l in zip(hits['asof'], hits['leaky']) if a is not None and a and not l)
        res['mcnemar_leaky_vs_asof'] = {'leaky_only': b, 'asof_only': c, 'p_value_exact': mcnemar_exact(b, c)}
        out['by_topk'][str(K)] = res
        hits_at[K] = hits
        print('K={}: '.format(K) + ' | '.join('{} hit {:.4f} recall {:.4f}'.format(
            m, res[m]['hit_rate'], res[m]['mean_distinctive_recall']) for m in runs)
              + ' | McNemar leaky-only {} vs as-of-only {}, p={:.3g}'.format(b, c, mcnemar_exact(b, c)))

    # ---------------- example selection with a disclosed criterion ----------------
    frequent = set(ranked[:args.select_k])
    names = list(runs)
    differ = [i for i in range(n) if len({runs[m][i]['GEN'] for m in names}) == len(names)]
    pool = []
    for i in differ:
        x = runs['leaky'][i]
        d = distinctive(x['GT'], x.get('ASPECT_IN', ''), frequent)
        lk = d & set(x['GEN'].split())
        others = set()
        for m in names:
            if m != 'leaky':
                others |= set(runs[m][i]['GEN'].split())
        if lk - others:
            pool.append(i)
    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(pool, min(args.n_examples, len(pool))))
    rng2 = random.Random(args.seed)
    typical = sorted(rng2.sample(differ, min(args.n_examples, len(differ))))

    def show(i):
        x = runs['asof'][i]
        return {'test_index': i, 'aspect': x.get('ASPECT_IN', ''), 'ground_truth': x['GT'],
                'distinctive_tokens': sorted(distinctive(x['GT'], x.get('ASPECT_IN', ''), frequent)),
                **{m: runs[m][i]['GEN'] for m in names}}

    out['examples'] = {
        'criterion': ('test instances whose {} outputs all differ and whose leaky output contains at least one '
                      'distinctive reference token (K={}) absent from the other outputs; {} drawn uniformly '
                      'with random seed {}').format(len(names), args.select_k, args.n_examples, args.seed),
        'n_outputs_differ': len(differ),
        'pool_size': len(pool),
        'selected': [show(i) for i in chosen],
        'typical_random_from_differ': [show(i) for i in typical],
    }
    print('outputs differ: {} | criterion pool: {} ({:.2%} of differ)'.format(
        len(differ), len(pool), len(pool) / max(1, len(differ))))
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('saved ->', args.out)


if __name__ == '__main__':
    main()
