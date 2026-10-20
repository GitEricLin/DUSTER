"""
Recover review timestamps for the PETER-style reviews.pickle files by joining with the raw data.

  Amazon 2014 (McAuley):  reviewerID, asin, unixReviewTime, reviewText
      e.g. reviews_Clothing_Shoes_and_Jewelry_5.json.gz / reviews_Movies_and_TV_5.json.gz
           (or the full reviews_<category>.json.gz)  http://jmcauley.ucsd.edu/data/amazon/
  Yelp Open Dataset:      user_id, business_id, date, text      (yelp_academic_dataset_review.json)
  TripAdvisor (Li et al.): userID, hotelID, reviewURL, reviewDate, reviewText (OriginalReviews.json,
                           already in the TripAdvisor_Dataset folder). reviewDate is missing for most
                           records, so TripAdvisor is ordered by the review id in reviewURL
                           (.../ShowUserReviews-g186338-d224761-r147393627-...): ids are assigned
                           sequentially, hence chronologically. Calendar dates are kept where present and
                           interpolated from the review id elsewhere (field 'date').

Join key = (user, item). If a pair has several raw reviews, the one whose text contains the
template sentence (after normalisation) is chosen; otherwise the earliest one (counted as ambiguous).
Unmatched records are dropped. Output records keep every original field plus:
    'time'     : int, unix seconds (UTC); for TripAdvisor the review id (monotone in time)
    'date'     : 'YYYY-MM-DD' (TripAdvisor: exact when reviewDate exists, otherwise interpolated)
    'orig_idx' : index of the record in the original reviews.pickle

Usage
  python add_timestamps.py --dataset amazon --pickle reviews.pickle \
         --raw reviews_Clothing_Shoes_and_Jewelry_5.json.gz --out reviews_time.pickle
  python add_timestamps.py --dataset yelp --pickle reviews.pickle \
         --raw yelp_academic_dataset_review.json --out reviews_time.pickle
  python add_timestamps.py --dataset tripadvisor --pickle reviews.pickle \
         --raw OriginalReviews.json --out reviews_time.pickle
"""
import argparse
import bisect
import ast
import datetime
import gzip
import json
import pickle
import re
from collections import defaultdict

_norm_re = re.compile(r'[^a-z0-9]')


def norm(s):
    return _norm_re.sub('', (s or '').lower())


def open_any(path):
    return gzip.open(path, 'rt', encoding='utf-8') if path.endswith('.gz') else open(path, 'r', encoding='utf-8')


def to_unix(date_str, fmt):
    return int(datetime.datetime.strptime(date_str, fmt).replace(tzinfo=datetime.timezone.utc).timestamp())


def iter_json_lines(path):
    with open_any(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:          # Amazon 2014 "loose json" (python dict literals)
                yield ast.literal_eval(line)


def iter_json_array(path):
    """stream a big JSON array (TripAdvisor OriginalReviews.json); uses ijson if installed."""
    try:
        import ijson
        with open_any(path) as f:
            for obj in ijson.items(f, 'item'):
                yield obj
    except ImportError:
        with open_any(path) as f:
            for obj in json.load(f):
                yield obj


_rid_re = re.compile(r'-r(\d+)-')


def raw_records(dataset, path, needed_items, field_stats):
    """yield (user, item, order_key, text, date_str) for raw reviews of items in `needed_items`."""
    if dataset == 'amazon':
        for r in iter_json_lines(path):
            it = str(r.get('asin'))
            if it in needed_items:
                t = r.get('unixReviewTime')
                if t is None and r.get('reviewTime'):
                    t = to_unix(r['reviewTime'], '%m %d, %Y')
                if t is None:
                    field_stats['no_time'] += 1
                    continue
                yield r.get('reviewerID'), it, int(t), (r.get('summary', '') + ' ' + r.get('reviewText', '')), None
    elif dataset == 'yelp':
        for r in iter_json_lines(path):
            it = str(r.get('business_id'))
            if it in needed_items:
                d = r['date']
                fmt = '%Y-%m-%d %H:%M:%S' if len(d) > 10 else '%Y-%m-%d'
                yield r.get('user_id'), it, to_unix(d, fmt), r.get('text', ''), None
    elif dataset == 'tripadvisor':
        for r in iter_json_array(path):
            it = str(r.get('hotelID'))
            if it not in needed_items:
                continue
            field_stats['raw_records'] += 1
            date = r.get('reviewDate')
            field_stats['has_reviewDate'] += bool(date)
            field_stats['has_travelDate'] += bool(r.get('travelDate'))
            m = _rid_re.search(str(r.get('reviewURL') or ''))
            if not m:
                field_stats['no_review_id_in_url'] += 1
                continue
            yield r.get('userID'), it, int(m.group(1)), \
                (str(r.get('reviewHeading') or '') + ' ' + str(r.get('reviewText') or '')), \
                (str(date)[:10] if date else None)
    else:
        raise ValueError(dataset)


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for k, i in enumerate(order):
            r[i] = k
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    if n < 3:
        return None
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1 - 6 * d2 / (n * (n * n - 1))


def interpolate_dates(anchors, keys):
    """anchors: sorted [(review_id, unix)], made monotone; returns unix estimate per key."""
    xs, ys, best = [], [], float('-inf')
    for x, y in anchors:
        best = max(best, y)
        xs.append(x)
        ys.append(best)
    out = []
    for k in keys:
        p = bisect.bisect_left(xs, k)
        if p == 0:
            out.append(ys[0])
        elif p >= len(xs):
            out.append(ys[-1])
        else:
            x0, x1, y0, y1 = xs[p - 1], xs[p], ys[p - 1], ys[p]
            out.append(y0 if x1 == x0 else y0 + (y1 - y0) * (k - x0) / (x1 - x0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['amazon', 'yelp', 'tripadvisor'], required=True)
    ap.add_argument('--pickle', required=True)
    ap.add_argument('--raw', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--match', choices=['pair', 'pair+text'], default='pair+text',
                    help="pair: (user,item) only, low memory (use for the 5 GB Yelp file if RAM runs out); pair+text: if the pair is not found, match the "
                         "extracted sentence against all raw reviews of the same item")
    args = ap.parse_args()

    reviews = pickle.load(open(args.pickle, 'rb'))
    p_users = {str(r['user']) for r in reviews}
    p_items = {str(r['item']) for r in reviews}
    print('pickle records: {} | users: {} | items: {}'.format(len(reviews), len(p_users), len(p_items)))

    field_stats = defaultdict(int)
    by_item = defaultdict(list)                       # item -> [(user, key, normtext, date)]
    raw_users, n_raw = set(), 0
    needed_pairs = {(str(r['user']), str(r['item'])) for r in reviews}
    for u, it, key, text, date in raw_records(args.dataset, args.raw, p_items, field_stats):
        if args.match == 'pair' and (str(u), it) not in needed_pairs:
            continue                                   # low-memory mode: keep only needed pairs
        by_item[it].append((str(u), key, norm(text), date))
        raw_users.add(str(u))
        n_raw += 1
    diag = {'raw_reviews_used': n_raw, 'raw_field_stats': dict(field_stats),
            'items_found_in_raw': len(by_item), 'items_in_pickle': len(p_items),
            'user_id_overlap': len(p_users & raw_users), 'users_in_pickle': len(p_users)}
    print('diagnostics:', json.dumps(diag, indent=2))

    out, unmatched, ambiguous, multi, by_pair, by_text = [], 0, 0, 0, 0, 0
    for i, r in enumerate(reviews):
        u, it = str(r['user']), str(r['item'])
        pool = by_item.get(it, [])
        sent = norm(r['template'][2])
        cands = [e for e in pool if e[0] == u]
        chosen = None
        if cands:
            by_pair += 1
            if len(cands) == 1:
                chosen = cands[0]
            else:
                multi += 1
                hits = [e for e in cands if sent and sent in e[2]]
                if not hits:
                    ambiguous += 1
                    hits = cands
                chosen = min(hits, key=lambda e: e[1])
        elif args.match == 'pair+text' and sent and len(sent) >= 10:
            hits = [e for e in pool if sent in e[2]]
            if hits:
                by_text += 1
                if len({e[1] for e in hits}) > 1:
                    ambiguous += 1
                chosen = min(hits, key=lambda e: e[1])
        if chosen is None:
            unmatched += 1
            continue
        rec = dict(r)
        rec['time'] = int(chosen[1])
        rec['orig_idx'] = i
        rec['_raw_date'] = chosen[3]
        out.append(rec)

    day = lambda x: datetime.datetime.fromtimestamp(x, datetime.timezone.utc).strftime('%Y-%m-%d')
    extra = {}
    if args.dataset == 'tripadvisor':
        anchors = sorted({(e[1], to_unix(e[3], '%Y-%m-%d')) for lst in by_item.values() for e in lst if e[3]})
        extra['date_anchors'] = len(anchors)
        extra['spearman_reviewid_vs_date'] = spearman([a for a, _ in anchors], [b for _, b in anchors])
        est = interpolate_dates(anchors, [r['time'] for r in out]) if anchors else [None] * len(out)
        exact = 0
        for r, e in zip(out, est):
            if r['_raw_date']:
                r['date'] = r['_raw_date']
                exact += 1
            else:
                r['date'] = day(int(e)) if e is not None else None
        extra['records_with_exact_date'] = exact
        extra['time_unit'] = 'tripadvisor_review_id (order only); see field date'
    else:
        for r in out:
            r['date'] = day(r['time'])
    for r in out:
        r.pop('_raw_date', None)

    dates = sorted(r['date'] for r in out if r.get('date'))
    stats = {
        'dataset': args.dataset, 'records_in': len(reviews), 'records_out': len(out),
        'match_rate': len(out) / max(1, len(reviews)), 'matched_by_user_item_pair': by_pair,
        'matched_by_item_and_sentence_text': by_text, 'unmatched_dropped': unmatched,
        'pairs_with_multiple_raw_reviews': multi, 'ambiguous_resolved_by_earliest': ambiguous,
        'date_min': dates[0] if dates else None, 'date_max': dates[-1] if dates else None,
        **extra, 'diagnostics': diag,
    }
    print(json.dumps(stats, indent=2))
    pickle.dump(out, open(args.out, 'wb'))
    json.dump(stats, open(args.out + '.stats.json', 'w'), indent=2)
    print('saved ->', args.out)


if __name__ == '__main__':
    main()
