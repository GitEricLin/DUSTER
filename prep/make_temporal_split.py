"""
Global chronological split (Ji et al., TOIS 2023): every validation review is later than every
training review, and every test review is later than every validation review.

  python make_temporal_split.py --pickle reviews_time.pickle --out_dir temporal \
         --ratio 0.8 0.1 0.1 --drop_cold

Writes train.index / validation.index / test.index (same one-line format as the original splits)
and split_stats.json. Indices refer to positions in reviews_time.pickle.

--drop_cold removes validation/test records whose user or item never occurs in the training split
(ID embeddings of such entities would be untrained). Their reviews are NOT used as history either.
"""
import argparse
import datetime
import json
import os
import pickle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pickle', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--ratio', type=float, nargs=3, default=[0.8, 0.1, 0.1])
    ap.add_argument('--drop_cold', action='store_true')
    args = ap.parse_args()

    reviews = pickle.load(open(args.pickle, 'rb'))
    assert all('time' in r for r in reviews), "run add_timestamps.py first"
    n = len(reviews)
    order = sorted(range(n), key=lambda i: (reviews[i]['time'], i))
    times = [reviews[i]['time'] for i in order]
    # cut on timestamp VALUES so that no two splits share a timestamp
    t1 = times[min(n - 1, int(args.ratio[0] * n))]
    t2 = times[min(n - 1, int((args.ratio[0] + args.ratio[1]) * n))]
    train = [i for i in order if reviews[i]['time'] < t1]
    valid = [i for i in order if t1 <= reviews[i]['time'] < t2]
    test = [i for i in order if reviews[i]['time'] >= t2]

    cold = {'valid': 0, 'test': 0}
    if args.drop_cold:
        users = {reviews[i]['user'] for i in train}
        items = {reviews[i]['item'] for i in train}
        keep = lambda i: reviews[i]['user'] in users and reviews[i]['item'] in items
        v2, t2_ = [i for i in valid if keep(i)], [i for i in test if keep(i)]
        cold = {'valid': len(valid) - len(v2), 'test': len(test) - len(t2_)}
        valid, test = v2, t2_

    os.makedirs(args.out_dir, exist_ok=True)
    for name, idx in (('train', train), ('validation', valid), ('test', test)):
        with open(os.path.join(args.out_dir, name + '.index'), 'w') as f:
            f.write(' '.join(map(str, idx)))
    def day(t):   # calendar date of the first record at time-key t (TripAdvisor: key = review id)
        j = next(i for i in order if reviews[i]['time'] >= t)
        if reviews[j].get('date'):
            return reviews[j]['date']
        return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime('%Y-%m-%d')
    stats = {'n': n, 'train': len(train), 'validation': len(valid), 'test': len(test),
             'validation_start': day(t1), 'test_start': day(t2),
             'cold_start_dropped': cold,
             'fractions': [round(len(x) / n, 4) for x in (train, valid, test)]}
    json.dump(stats, open(os.path.join(args.out_dir, 'split_stats.json'), 'w'), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
