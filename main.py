"""
DUSTER v2 - training / evaluation (one script for DUSTER and DUSTER_A).

  --aspect_mode gold : feature-given protocol (original DUSTER; ground-truth aspect fed at test time)
  --aspect_mode pred : decision-time protocol (DUSTER_A; aspect predicted from history only)
  --aspect_mode none : no aspect token

Example
  python main.py --data_path ../Amazon/ClothingShoesAndJewelry/reviews_time.pickle \
                 --index_dir ../Amazon/ClothingShoesAndJewelry/temporal \
                 --aspect_mode pred --cuda --seed 1 --checkpoint ./runs/clothing_pred_s1/
"""
import os
import json
import math
import argparse
import torch
import torch.nn as nn
from module import DUSTER
from utils import (rouge_score, bleu_score, DataLoader, Batchify, now_time, ids2tokens,
                   unique_sentence_percent, root_mean_square_error, mean_absolute_error,
                   feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity)

parser = argparse.ArgumentParser(description='DUSTER v2 (leakage-free)')
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--index_dir', type=str, required=True)
parser.add_argument('--emsize', type=int, default=512)
parser.add_argument('--nhead', type=int, default=2)
parser.add_argument('--nhid', type=int, default=2048)
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=1.0)
parser.add_argument('--clip', type=float, default=1.0)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--log_interval', type=int, default=200)
parser.add_argument('--checkpoint', type=str, default='./duster/')
parser.add_argument('--outf', type=str, default='generated.txt')
parser.add_argument('--vocab_size', type=int, default=20000)
parser.add_argument('--vocab_scope', choices=['all', 'train'], default='all')
parser.add_argument('--endure_times', type=int, default=5)
parser.add_argument('--rating_reg', type=float, default=0.1)
parser.add_argument('--context_reg', type=float, default=1.0)
parser.add_argument('--text_reg', type=float, default=0.8)
parser.add_argument('--aspect_reg', type=float, default=0.1)
parser.add_argument('--words', type=int, default=15)
parser.add_argument('--his_len', type=int, default=100, help='max number of history reviews')
parser.add_argument('--history_scope', choices=['auto', 'as_of', 'train', 'leaky'], default='auto',
                    help="'leaky' reproduces the original flawed history (leakage study only)")
parser.add_argument('--window_sizes', type=int, nargs='+', default=[3])
parser.add_argument('--num_filters', type=int, default=100)
parser.add_argument('--aspect_mode', choices=['gold', 'pred', 'none'], default='pred')
parser.add_argument('--aspect_tokens', choices=['feature', 'feature_adj'], default='feature')
parser.add_argument('--aspect_source', choices=['user', 'item', 'joint', 'freq'], default='user',
                    help='which predictor head chooses the aspect at inference (pred mode); '
                         'freq = no learned head: most frequent aspect in the user + item TRAINING '
                         'reviews (a training record never counts its own aspect)')
parser.add_argument('--train_aspect_input', choices=['gold', 'pred', 'mix'], default='gold',
                    help="pred mode only: aspect fed to the generator during TRAINING "
                         "(gold = teacher forcing as in the original code; pred = predicted aspect, "
                         "matches test conditions; mix = per-sample 50/50)")
parser.add_argument('--no_duster_mask', action='store_true', help='use plain left-to-right mask')
# ablations
parser.add_argument('--no_user_hist', action='store_true')
parser.add_argument('--no_item_hist', action='store_true')
parser.add_argument('--no_gate', action='store_true')
parser.add_argument('--no_gru', action='store_true')
parser.add_argument('--no_attn', action='store_true')
parser.add_argument('--fuse', choices=['concat', 'rezero', 'gate'], default='concat',
                    help='how history is combined with the ID embedding (concat = original DUSTER)')
parser.add_argument('--hist_dropout', type=float, default=0.0,
                    help='prob. of dropping a sample\'s whole user / item history during training')
parser.add_argument('--check_leakage', type=int, default=1000,
                    help='number of test/valid samples to verify before training (0 = skip)')
args = parser.parse_args()

print('-' * 40 + 'ARGUMENTS' + '-' * 40)
for arg in vars(args):
    print('{:40} {}'.format(arg, getattr(args, arg)))
print('-' * 40 + 'ARGUMENTS' + '-' * 40)

torch.manual_seed(args.seed)
import random as _random
_random.seed(args.seed)
if torch.cuda.is_available() and not args.cuda:
    print(now_time() + 'WARNING: You have a CUDA device, so you should probably run with --cuda')
device = torch.device('cuda' if args.cuda else 'cpu')
os.makedirs(args.checkpoint, exist_ok=True)
model_path = os.path.join(args.checkpoint, 'model.pt')
prediction_path = os.path.join(args.checkpoint, args.outf)

###############################################################################
# Load data
###############################################################################
print(now_time() + 'Loading data')
corpus = DataLoader(args.data_path, args.index_dir, args.vocab_size, seq_len=args.words,
                    his_len=args.his_len, history_scope=args.history_scope,
                    vocab_scope=args.vocab_scope)
print(now_time() + 'history scope = {} | timestamps = {}'.format(corpus.history_scope, corpus.has_time))
if args.check_leakage > 0:
    for split in ('valid', 'test'):
        ok, rep = corpus.check_leakage(split, args.check_leakage, seed=args.seed)
        print(now_time() + 'leakage check [{}]: {} | {}'.format(split, 'PASS' if ok else 'FAIL', rep))
        if not ok and args.history_scope != 'leaky':
            raise RuntimeError('History leakage detected - aborting.')

word2idx = corpus.word_dict.word2idx
idx2word = corpus.word_dict.idx2word
feature_set = corpus.feature_set
rule_aspect = None
if args.aspect_source == 'freq':
    from collections import Counter, defaultdict
    _g, _u, _i = Counter(), defaultdict(Counter), defaultdict(Counter)
    for x in corpus.train:
        _g[x['feature']] += 1
        _u[x['user']][x['feature']] += 1
        _i[x['item']][x['feature']] += 1
    _top = _g.most_common(1)[0][0]
    _train_idx = {x['idx'] for x in corpus.train}
    rule_aspect = torch.full((len(corpus.data),), _top, dtype=torch.long)
    for x in corpus.train + corpus.valid + corpus.test:
        c = _u[x['user']] + _i[x['item']]
        if x['idx'] in _train_idx:
            c[x['feature']] -= 2                  # remove the record's own aspect (user + item counter)
            c = +c
        if c:
            rule_aspect[x['idx']] = c.most_common(1)[0][0]
    rule_aspect = rule_aspect.to(device)
train_data = Batchify(corpus.train, word2idx, args.words, args.batch_size, shuffle=True)
val_data = Batchify(corpus.valid, word2idx, args.words, args.batch_size)
test_data = Batchify(corpus.test, word2idx, args.words, args.batch_size)

###############################################################################
# Build the model
###############################################################################
n_aspect = 0 if args.aspect_mode == 'none' else (1 if args.aspect_tokens == 'feature' else 2)
src_len = 2 + n_aspect                           # [u, i, (feature), (adjective)]
tgt_len = args.words + 1
ntokens = len(corpus.word_dict)
pad_idx = word2idx['<pad>']
model = DUSTER(src_len, tgt_len, pad_idx, len(corpus.user_dict), len(corpus.item_dict), ntokens,
               args.emsize, args.nhead, args.nhid, args.nlayers, args.window_sizes, args.num_filters,
               args.dropout, duster_mask=not args.no_duster_mask, aspect_ids=corpus.feature_ids,
               use_user_hist=not args.no_user_hist, use_item_hist=not args.no_item_hist,
               use_gate=not args.no_gate, use_gru=not args.no_gru, use_attn=not args.no_attn,
               fuse=args.fuse, hist_dropout=args.hist_dropout).to(device)
text_criterion = nn.NLLLoss(ignore_index=pad_idx)
rating_criterion = nn.MSELoss()
aspect_criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.25)
train_aspect_head = args.aspect_mode == 'pred'


###############################################################################
# Helpers
###############################################################################
def histories(idx):
    hu, lu = corpus.build_histories(idx, 'user')
    hi, li = corpus.build_histories(idx, 'item')
    return hu.to(device), lu.to(device), hi.to(device), li.to(device)


def combine(logit_u, logit_i):
    if args.aspect_source == 'user':
        return logit_u
    if args.aspect_source == 'item':
        return logit_i
    return torch.log_softmax(logit_u, -1) + torch.log_softmax(logit_i, -1)


def choose(logit_u, logit_i):
    return torch.argmax(combine(logit_u, logit_i), dim=1).unsqueeze(0)    # (1, B)


def aspect_input(feature, adjective, heads, use_gold, idx=None):
    """aspect rows of `text`: (n_aspect, B), or None when --aspect_mode none."""
    if args.aspect_mode == 'none':
        return None
    a_u, a_i, j_u, j_i = heads
    if use_gold is True:
        f, j = feature, adjective
    else:
        f, j = choose(a_u, a_i), choose(j_u, j_i)
        if rule_aspect is not None:
            f = rule_aspect[torch.as_tensor(idx, device=device)].unsqueeze(0)
        if use_gold is not False:                     # per-sample mask (1, B): True -> gold
            f = torch.where(use_gold, feature, f)
            j = torch.where(use_gold, adjective, j)
    if args.aspect_tokens == 'feature':
        return f
    return torch.cat([f, j], 0)


def aspect_loss(heads, feature, adjective):
    a_u, a_i, j_u, j_i = heads
    f, j = feature.squeeze(0), adjective.squeeze(0)
    loss = aspect_criterion(a_u, f) + aspect_criterion(a_i, f)
    if args.aspect_tokens == 'feature_adj':
        loss = loss + aspect_criterion(j_u, j) + aspect_criterion(j_i, j)
    return loss


def predict_context(log_context_dis, topk):
    word_prob = log_context_dis.exp()
    if topk == 1:
        return torch.argmax(word_prob, dim=1, keepdim=True)
    return torch.topk(word_prob, topk, 1)[1]


###############################################################################
# Training code
###############################################################################
def run_batch(data, train):
    user, item, rating, seq, feature, adjective, idx = data.next_batch()
    batch_size = user.size(0)
    user, item, rating = user.to(device), item.to(device), rating.to(device)
    seq = seq.t().to(device)                    # (tgt_len + 1, B)
    feature = feature.t().to(device)            # (1, B)
    adjective = adjective.t().to(device)
    hu, lu, hi, li = histories(idx)
    u_p, i_p, heads = model.encode_ui(user, item, hu, lu, hi, li)
    # training: teacher forcing with the gold aspect (as in the original code);
    # evaluation: gold aspect only in 'gold' mode, predicted aspect in 'pred' mode.
    if args.aspect_mode == 'gold' or (train and args.train_aspect_input == 'gold'):
        use_gold = True
    elif train and args.train_aspect_input == 'mix':
        use_gold = torch.rand(1, user.size(0), device=device) < 0.5
    else:
        use_gold = False
    asp = aspect_input(feature, adjective, heads, use_gold, idx)
    text = seq[:-1] if asp is None else torch.cat([asp, seq[:-1]], 0)
    log_word_prob, log_context_dis, rating_p, _ = model(u_p, i_p, text)
    context_dis = log_context_dis.unsqueeze(0).repeat((tgt_len - 1, 1, 1))
    c_loss = text_criterion(context_dis.view(-1, ntokens), seq[1:-1].reshape((-1,)))
    r_loss = rating_criterion(rating_p, rating)
    t_loss = text_criterion(log_word_prob.view(-1, ntokens), seq[1:].reshape((-1,)))
    a_loss = aspect_loss(heads, feature, adjective) if train_aspect_head else torch.zeros((), device=device)
    return batch_size, c_loss, t_loss, r_loss, a_loss


def train(data):
    model.train()
    sums = [0., 0., 0., 0.]
    total = 0
    while True:
        bs, c_loss, t_loss, r_loss, a_loss = run_batch(data, True)
        loss = (args.text_reg * t_loss + args.context_reg * c_loss + args.rating_reg * r_loss
                + args.aspect_reg * a_loss)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        for k, v in enumerate((c_loss, t_loss, r_loss, a_loss)):
            sums[k] += bs * v.item()
        total += bs
        if data.step % args.log_interval == 0 or data.step == data.total_step:
            c, t, r, a = (s / total for s in sums)
            print(now_time() + 'context ppl {:4.4f} | text ppl {:4.4f} | rating loss {:4.4f} | '
                  'aspect loss {:4.4f} | {:5d}/{:5d} batches'.format(
                      math.exp(c), math.exp(t), r, a, data.step, data.total_step))
            sums, total = [0., 0., 0., 0.], 0
        if data.step == data.total_step:
            break


def evaluate(data):
    model.eval()
    sums = [0., 0., 0., 0.]
    total = 0
    with torch.no_grad():
        while True:
            bs, c_loss, t_loss, r_loss, a_loss = run_batch(data, False)
            for k, v in enumerate((c_loss, t_loss, r_loss, a_loss)):
                sums[k] += bs * v.item()
            total += bs
            if data.step == data.total_step:
                break
    return [s / total for s in sums]


def generate(data):
    model.eval()
    idss_predict, context_predict, rating_predict, aspect_predict, aspect_top10 = [], [], [], [], []
    with torch.no_grad():
        while True:
            user, item, rating, seq, feature, adjective, idx = data.next_batch()
            user, item = user.to(device), item.to(device)
            bos = seq[:, 0].unsqueeze(0).to(device)
            feature = feature.t().to(device)
            adjective = adjective.t().to(device)
            hu, lu, hi, li = histories(idx)
            u_p, i_p, heads = model.encode_ui(user, item, hu, lu, hi, li)   # computed once
            asp = aspect_input(feature, adjective, heads, args.aspect_mode == 'gold', idx)
            if asp is not None:
                aspect_predict.extend(asp[0].tolist())
            if args.aspect_mode == 'pred' and rule_aspect is None:
                aspect_top10.extend(torch.topk(combine(heads[0], heads[1]), 10, dim=1)[1].tolist())
            text = bos if asp is None else torch.cat([asp, bos], 0)
            start_idx = text.size(0)
            for step in range(args.words):
                if step == 0:
                    log_word_prob, log_context_dis, rating_p, _ = model(u_p, i_p, text, False)
                    rating_predict.extend(rating_p.tolist())
                    context_predict.extend(predict_context(log_context_dis, args.words).tolist())
                else:
                    log_word_prob, _, _, _ = model(u_p, i_p, text, False, False, False)
                word_idx = torch.argmax(log_word_prob.exp(), dim=1)
                text = torch.cat([text, word_idx.unsqueeze(0)], 0)
            idss_predict.extend(text[start_idx:].t().tolist())
            if data.step == data.total_step:
                break

    results = {}
    predicted_rating = [(r, p) for (r, p) in zip(data.rating.tolist(), rating_predict)]
    results['RMSE'] = root_mean_square_error(predicted_rating, corpus.max_rating, corpus.min_rating)
    results['MAE'] = mean_absolute_error(predicted_rating, corpus.max_rating, corpus.min_rating)
    tokens_test = [ids2tokens(ids[1:], word2idx, idx2word) for ids in data.seq.tolist()]
    tokens_predict = [ids2tokens(ids, word2idx, idx2word) for ids in idss_predict]
    results['BLEU-1'] = bleu_score(tokens_test, tokens_predict, n_gram=1, smooth=False)
    results['BLEU-4'] = bleu_score(tokens_test, tokens_predict, n_gram=4, smooth=False)
    results['USR'], results['USN'] = unique_sentence_percent(tokens_predict)
    feature_batch = feature_detect(tokens_predict, feature_set)
    results['DIV'] = feature_diversity(feature_batch)
    results['FCR'] = feature_coverage_ratio(feature_batch, feature_set)
    feature_test = [idx2word[i] for i in data.feature.squeeze(1).tolist()]   # single GT feature word
    results['FMR'] = feature_matching_ratio(feature_batch, feature_test)
    if args.aspect_mode == 'pred':
        gold = data.feature.squeeze(1).tolist()
        results['AspectAcc@1'] = sum(int(p == g) for p, g in zip(aspect_predict, gold)) / len(gold)
        for k in ((5, 10) if aspect_top10 else ()):
            results['AspectAcc@{}'.format(k)] = sum(int(g in t[:k]) for t, g in zip(aspect_top10, gold)) / len(gold)
    text_test = [' '.join(t) for t in tokens_test]
    text_predict = [' '.join(t) for t in tokens_predict]
    for (k, v) in rouge_score(text_test, text_predict).items():
        results[k] = v
    for k, v in results.items():
        print(now_time() + '{} {:7.4f}'.format(k, v))
    tokens_context = [' '.join([idx2word[i] for i in ids]) for ids in context_predict]
    out = ''
    asp_words = [idx2word[i] for i in aspect_predict] if aspect_predict else [''] * len(text_test)
    for (real, ctx, fake, asp_w) in zip(text_test, tokens_context, text_predict, asp_words):
        out += 'GT: {}\nCTX: {}\nGEN: {}\nASPECT_IN: {}\n\n'.format(real, ctx, fake, asp_w)
    return out, results


###############################################################################
# Main loop
###############################################################################
best_val_loss = float('inf')
endure_count = 0
for epoch in range(1, args.epochs + 1):
    print(now_time() + 'epoch {}'.format(epoch))
    train(train_data)
    val_c, val_t, val_r, val_a = evaluate(val_data)
    val_loss = val_t + val_r if args.rating_reg != 0 else val_t
    if train_aspect_head:
        val_loss += args.aspect_reg * val_a
    print(now_time() + 'context ppl {:4.4f} | text ppl {:4.4f} | rating loss {:4.4f} | aspect loss {:4.4f} '
          '| valid loss {:4.4f} on validation'.format(math.exp(val_c), math.exp(val_t), val_r, val_a, val_loss))
    print(now_time() + 'fusion ' + ' '.join('{} {:.4f}'.format(k, v) for k, v in model.fusion_stats().items()))
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        with open(model_path, 'wb') as f:
            torch.save(model, f)
    else:
        endure_count += 1
        print(now_time() + 'Endured {} time(s)'.format(endure_count))
        if endure_count == args.endure_times:
            print(now_time() + 'Cannot endure it anymore | Exiting from early stop')
            break
        scheduler.step()
        print(now_time() + 'Learning rate set to {:2.8f}'.format(scheduler.get_last_lr()[0]))

with open(model_path, 'rb') as f:
    model = torch.load(f, weights_only=False).to(device)

test_c, test_t, test_r, test_a = evaluate(test_data)
print('=' * 89)
print(now_time() + 'context ppl {:4.4f} | text ppl {:4.4f} | rating loss {:4.4f} on test | End of training'.format(
    math.exp(test_c), math.exp(test_t), test_r))
print(now_time() + 'Generating text')
text_o, results = generate(test_data)
with open(prediction_path, 'w', encoding='utf-8') as f:
    f.write(text_o)
results.update({'seed': args.seed, 'aspect_mode': args.aspect_mode, 'history_scope': corpus.history_scope,
                'data_path': args.data_path, 'index_dir': args.index_dir,
                'fuse': args.fuse, 'hist_dropout': args.hist_dropout,
                'no_user_hist': args.no_user_hist, 'no_item_hist': args.no_item_hist,
                'text_reg': args.text_reg, **model.fusion_stats()})
with open(os.path.join(args.checkpoint, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(now_time() + 'Generated text saved to ({}) | metrics saved to results.json'.format(prediction_path))
