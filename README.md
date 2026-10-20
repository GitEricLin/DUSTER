# DUSTER: point-in-time evaluation of history-aware explainable recommendation

This repository contains the code for the paper *Auditing History Leakage in Explainable Recommendation via Point-in-Time Data Contracts* (under review).

DUSTER is a dual-sequence, history-aware Transformer that jointly predicts a rating and generates a short natural-language explanation. It encodes the review histories of a user and of an item separately. The code also implements the data side of the paper:

- recovering review timestamps;
- a global chronological split;
- an **as-of (point-in-time) history construction** with an automated leakage audit;
- two evaluation protocols.

## Evaluation protocols

One script, `main.py`, runs every variant; `--aspect_mode` selects the protocol:

| `--aspect_mode` | Protocol | Meaning |
|---|---|---|
| `gold` | feature-given (oracle-assisted) | the ground-truth aspect of the target review is fed to the generator at test time; report only as a reference, since it is unavailable at recommendation time |
| `pred` | decision-time | the aspect is predicted from the histories only |
| `none` | no aspect token | ablation |

## As-of history construction

For every target record, the user and item histories are built separately and contain only records that were available before it:

- the target record itself is always excluded;
- only records whose split rank is not later than the target's are visible: training targets see training records, validation targets see training and validation records, and test targets see all three;
- when timestamps exist, only records **strictly earlier** than the target are used, in chronological order, keeping the most recent `--his_len` reviews.

Before training, `main.py` checks a sample of validation and test instances (`--check_leakage`, default 1,000) and aborts if any history contains the target or a later record. `audit_full.py` runs the same check on every instance.

`--history_scope` controls the construction:

| Value | History |
|---|---|
| `auto` (default) | `as_of` if every record has a timestamp, otherwise `train` |
| `as_of` | point-in-time history as described above (requires timestamps) |
| `train` | training records only, target excluded; for random splits without timestamps |
| `leaky` | reproduces the unconstrained pipeline, in which the target can appear in its own history; **for the leakage study only** |

## Repository structure

| File | Purpose |
|---|---|
| `main.py` | training and evaluation (all protocols, ablations and fusion variants) |
| `module.py` | model: history encoders, Transformer, rating / context / explanation / aspect heads |
| `utils.py` | data loading, as-of history construction, leakage check, metrics |
| `bleu.py`, `rouge.py` | BLEU and ROUGE, from [tensorflow/nmt](https://github.com/tensorflow/nmt) as used by PETER |
| `prep/add_timestamps.py` | recovers review timestamps by joining `reviews.pickle` with the raw data |
| `prep/make_temporal_split.py` | global chronological split (train / validation / test) |
| `audit_full.py` | full (non-sampled) leakage audit of all splits |
| `run_all.sh` | runs datasets × protocols × 5 seeds |
| `aggregate_results.py` | mean ± standard deviation of `results.json` over seeds |
| `output_stats.py` | diversity of generated explanations and reproducible example selection |
| `target_overlap.py` | how often outputs reuse distinctive reference tokens (leaky vs. as-of), with an exact McNemar test |
| `fusion_probe.py` | contribution of the history vs. the ID embedding in a trained model |

## Requirements

Python 3 and the packages in `requirements.txt` (PyTorch ≥ 2.0; `ijson` for streaming the large TripAdvisor JSON file). A GPU is recommended for training.

## Data

The code uses the PETER-style `reviews.pickle` files and index files (Li et al., ACL 2021) for TripAdvisor, Yelp, Amazon Clothing and Amazon Movies & TV. Timestamps are recovered from the raw data:

| Dataset | Raw data | Join key | Time field |
|---|---|---|---|
| Amazon Clothing / Movies | Amazon 2014 (McAuley): `reviews_Clothing_Shoes_and_Jewelry_5.json.gz` or `reviews_Movies_and_TV_5.json.gz` (5-core or full), http://jmcauley.ucsd.edu/data/amazon/ | `(reviewerID, asin)` | `unixReviewTime` |
| Yelp | Yelp Open Dataset, `yelp_academic_dataset_review.json` (download from Yelp after accepting its license) | `(user_id, business_id)` | `date` |
| TripAdvisor | `OriginalReviews.json` (Li et al.) | `(userID, hotelID)` | `reviewDate` |

For TripAdvisor, most records have no `reviewDate`. Reviews are therefore ordered by the review id in `reviewURL`, which is assigned sequentially; this gives an approximate chronological order. Datasets are not redistributed here.

### 1. Recover timestamps and split

```bash
python prep/add_timestamps.py --dataset amazon \
  --pickle data/Clothing/reviews.pickle \
  --raw reviews_Clothing_Shoes_and_Jewelry_5.json.gz \
  --out data/Clothing/reviews_time.pickle

python prep/make_temporal_split.py \
  --pickle data/Clothing/reviews_time.pickle \
  --out_dir data/Clothing/temporal \
  --ratio 0.8 0.1 0.1 --drop_cold
```

Use `--dataset yelp` or `--dataset tripadvisor` for the other datasets. `--drop_cold` removes validation and test records whose user or item does not occur in the training split. Each run writes `reviews_time.pickle.stats.json` (match rate, time range) and `temporal/split_stats.json`.

### 2. Train and evaluate

```bash
python main.py --data_path data/Clothing/reviews_time.pickle \
               --index_dir data/Clothing/temporal \
               --aspect_mode pred --cuda --seed 1 --checkpoint runs/clothing_pred_s1/

DATA_ROOT=/path/to/datasets bash run_all.sh      # all datasets × {gold, pred} × 5 seeds
python aggregate_results.py "runs/clothing_pred_s*"
```

Each run writes `model.pt`, `generated.txt` and `results.json` to the `--checkpoint` folder.

Useful options:

- ablations: `--no_user_hist`, `--no_item_hist`, `--no_gate`, `--no_gru`, `--no_attn`, `--no_duster_mask`, `--aspect_mode none`;
- fusion of history and ID embedding: `--fuse concat|rezero|gate` (`rezero` is the zero-initialized residual fusion);
- aspect predictor in `pred` mode: `--aspect_source user|item|joint|freq` (`freq` = most frequent aspect in the user's and item's training reviews);
- vocabulary built from training data only: `--vocab_scope train`.

### 3. Audits and analyses

```bash
# full leakage audit (as-of contract, all splits)
python audit_full.py --data_path data/TripAdvisor/reviews_time.pickle \
                     --index_dir data/TripAdvisor/temporal --out runs/leakage_full_asof.json

# the same audit on the unconstrained (leaky) histories
python audit_full.py --data_path ... --index_dir ... --history_scope leaky \
                     --splits valid test --out runs/leakage_full_leaky.json

# target reuse: leaky vs. as-of outputs on the same test split
python target_overlap.py --asof runs/ta_gold_s1/generated.txt \
                         --leaky runs/ta_leaky_gold_s1/generated.txt \
                         --nohist runs/ta_nohist_gold_s1/generated.txt \
                         --data_path ... --index_dir ... --out runs/target_overlap.json

# output diversity, and history contribution in a trained model
python output_stats.py runs/ta_gold_s1/generated.txt runs/ta_nohist_gold_s1/generated.txt
python fusion_probe.py --data_path ... --index_dir ... --model runs/ta_gold_s1/model.pt --cuda
```

## Differences from the original DUSTER implementation

- Histories are built per target record under the as-of contract. Previously they were taken from the whole review file, and the target sentence could appear in its own history.
- The GRU output of the history encoder is used, histories are packed by their true lengths, and attention pooling masks padded reviews.
- User and item histories have separate encoders, the item ID uses its own embedding, and user and item fusion layers are separate.
- The "learnable mask" is implemented and named as what it is: a feature-wise gating vector shared by all reviews (`feature_gate`).
- Aspect logits are restricted to the feature-word vocabulary, and FMR is computed against the ground-truth feature word.
- The data file is loaded once, and histories are passed to the model rather than read inside `forward()`.

## Acknowledgements

The BLEU and ROUGE implementations come from [tensorflow/nmt](https://github.com/tensorflow/nmt), and the data format and evaluation follow [PETER](https://github.com/lileipisces/PETER) (Li et al., ACL 2021).
