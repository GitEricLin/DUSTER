"""
Full (non-sampled) leakage audit for DUSTER v2.

Runs DataLoader.check_leakage() on EVERY instance of the chosen splits instead of the
1,000-instance sample that main.py uses (--check_leakage 1000). No training is done.

Put this file in the DUSTER_V2 folder (next to utils.py, bleu.py, rouge.py) and run, e.g.:

  python audit_full.py --data_path .../TripAdvisor/reviews_time.pickle \
                       --index_dir .../TripAdvisor/temporal \
                       --out runs/leakage_full_asof.json

  # optional: full counts for the leaky control (replaces the 1,000-sample numbers in Sect. 5.2)
  python audit_full.py --data_path ... --index_dir ... --history_scope leaky \
                       --splits valid test --out runs/leakage_full_leaky.json
"""
import argparse
import json
import time

from utils import DataLoader

ap = argparse.ArgumentParser()
ap.add_argument('--data_path', required=True)
ap.add_argument('--index_dir', required=True)
ap.add_argument('--vocab_size', type=int, default=20000)
ap.add_argument('--words', type=int, default=15)
ap.add_argument('--his_len', type=int, default=100)
ap.add_argument('--history_scope', choices=['auto', 'as_of', 'train', 'leaky'], default='auto')
ap.add_argument('--vocab_scope', choices=['all', 'train'], default='all')
ap.add_argument('--splits', nargs='+', choices=['train', 'valid', 'test'], default=['train', 'valid', 'test'])
ap.add_argument('--out', default='leakage_full.json')
args = ap.parse_args()

corpus = DataLoader(args.data_path, args.index_dir, args.vocab_size, seq_len=args.words,
                    his_len=args.his_len, history_scope=args.history_scope,
                    vocab_scope=args.vocab_scope)
print('history scope = {} | timestamps = {}'.format(corpus.history_scope, corpus.has_time))

out = {'history_scope': corpus.history_scope, 'his_len': args.his_len}
all_ok = True
for split in args.splits:
    t0 = time.time()
    n = len(getattr(corpus, split))
    # n = full split size -> rnd.sample() returns every instance (just permuted)
    ok, rep = corpus.check_leakage(split, n=n, seed=0)
    rep['instances_in_split'] = n
    rep['histories_checked'] = 2 * rep['checked']          # user + item history per instance
    rep['pass'] = ok
    rep['seconds'] = round(time.time() - t0, 1)
    out[split] = rep
    all_ok = all_ok and ok
    print('[{}] {} | {}'.format(split, 'PASS' if ok else 'FAIL', rep))

out['all_pass'] = all_ok
with open(args.out, 'w') as f:
    json.dump(out, f, indent=2)
print('saved ->', args.out)
