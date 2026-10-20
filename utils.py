"""
DUSTER v2 - data utilities (leakage-free history construction).

Changes w.r.t. the original DUSTER/DUSTER_A utils.py
----------------------------------------------------
1. reviews.pickle is loaded ONCE.
2. Every record carries its pickle index ('idx'), its split rank and (optionally) a timestamp.
3. User / item histories are built PER TARGET RECORD, never from the whole file:
     * the target record itself is always excluded;
     * only records whose split rank <= rank of the target are visible
       (train sample -> train records only; valid -> train+valid; test -> train+valid+test);
     * if timestamps exist, only records with time STRICTLY EARLIER than the target are visible
       (point-in-time / as-of history), ordered chronologically, the most recent `his_len` kept.
   This removes the leakage of the original `group_data()` + `[-max_length:]` design, in which the
   ground-truth sentence of every test pair was part of its own user/item history.
4. FMR is computed against the ground-truth FEATURE WORD (the original DUSTER_A compared a
   "feature adjective" string against single feature words, which always yields 0.00).
"""
import os
import math
import heapq
import random
import pickle
import bisect
import datetime
import torch
from rouge import rouge          # unchanged file from the original code base (PETER)
from bleu import compute_bleu    # unchanged file from the original code base (PETER)


# --------------------------------------------------------------------------------------
# Metrics (same definitions as the original code / PETER)
# --------------------------------------------------------------------------------------
def rouge_score(references, generated):
    """both are a list of strings"""
    score = rouge(generated, references)
    return {k: (v * 100) for (k, v) in score.items()}


def bleu_score(references, generated, n_gram=4, smooth=False):
    """a list of lists of tokens"""
    formatted_ref = [[ref] for ref in references]
    bleu_s, _, _, _, _, _ = compute_bleu(formatted_ref, generated, n_gram, smooth)
    return bleu_s * 100


def two_seq_same(sa, sb):
    if len(sa) != len(sb):
        return False
    for (wa, wb) in zip(sa, sb):
        if wa != wb:
            return False
    return True


def unique_sentence_percent(sequence_batch):
    unique_seq = []
    for seq in sequence_batch:
        if not any(two_seq_same(seq, uni) for uni in unique_seq):
            unique_seq.append(seq)
    return len(unique_seq) / len(sequence_batch), len(unique_seq)


def feature_detect(seq_batch, feature_set):
    return [set(i for i in ids if i in feature_set) for ids in seq_batch]


def feature_matching_ratio(feature_batch, test_feature):
    """test_feature: list of single ground-truth feature WORDS (not 'feature adjective' strings)"""
    count = sum(1 for (fea_set, fea) in zip(feature_batch, test_feature) if fea in fea_set)
    return count / len(feature_batch)


def feature_coverage_ratio(feature_batch, feature_set):
    features = set()
    for fb in feature_batch:
        features = features | fb
    return len(features) / len(feature_set)


def feature_diversity(feature_batch):
    list_len = len(feature_batch)
    total_count = 0
    for i, x in enumerate(feature_batch):
        for j in range(i + 1, list_len):
            total_count += len(x & feature_batch[j])
    denominator = list_len * (list_len - 1) / 2
    return total_count / denominator


def mean_absolute_error(predicted, max_r, min_r, mae=True):
    total = 0
    for (r, p) in predicted:
        p = min(max(p, min_r), max_r)
        sub = p - r
        total += abs(sub) if mae else sub ** 2
    return total / len(predicted)


def root_mean_square_error(predicted, max_r, min_r):
    return math.sqrt(mean_absolute_error(predicted, max_r, min_r, False))


# --------------------------------------------------------------------------------------
# Dictionaries
# --------------------------------------------------------------------------------------
class WordDictionary:
    def __init__(self):
        self.idx2word = ['<bos>', '<eos>', '<pad>', '<unk>']
        self.__predefine_num = len(self.idx2word)
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}
        self.__word2count = {}

    def add_sentence(self, sentence):
        for w in sentence.split():
            self.add_word(w)

    def add_word(self, w):
        if w not in self.word2idx:
            self.word2idx[w] = len(self.idx2word)
            self.idx2word.append(w)
            self.__word2count[w] = 1
        else:
            self.__word2count[w] = self.__word2count.get(w, 0) + 1

    def __len__(self):
        return len(self.idx2word)

    def keep_most_frequent(self, max_vocab_size=20000):
        if len(self.__word2count) > max_vocab_size:
            frequent_words = heapq.nlargest(max_vocab_size, self.__word2count, key=self.__word2count.get)
            self.idx2word = self.idx2word[:self.__predefine_num] + frequent_words
            self.word2idx = {w: i for i, w in enumerate(self.idx2word)}


class EntityDictionary:
    def __init__(self):
        self.idx2entity = []
        self.entity2idx = {}

    def add_entity(self, e):
        if e not in self.entity2idx:
            self.entity2idx[e] = len(self.idx2entity)
            self.idx2entity.append(e)

    def __len__(self):
        return len(self.idx2entity)


def sentence_format(sentence, max_len, pad, bos, eos):
    length = len(sentence)
    if length >= max_len:
        return [bos] + sentence[:max_len] + [eos]
    return [bos] + sentence + [eos] + [pad] * (max_len - length)


# --------------------------------------------------------------------------------------
# Data loader with point-in-time history construction
# --------------------------------------------------------------------------------------
TRAIN, VALID, TEST, UNUSED = 0, 1, 2, 9


class DataLoader:
    def __init__(self, data_path, index_dir, vocab_size, seq_len=15, his_len=100,
                 history_scope='auto', vocab_scope='all'):
        """
        history_scope:
            'as_of' : requires a 'time' field; history = earlier records (time < t) with split rank <= target rank
            'train' : no time needed; history = training records only, target excluded (file order)
            'auto'  : 'as_of' if every record has 'time', else 'train'
            'leaky' : REPRODUCES THE ORIGINAL (FLAWED) PIPELINE, for the leakage study only:
                      the last his_len records of the user / item over the whole log, any split,
                      including the target review itself and future reviews
        vocab_scope:
            'all'   : vocabulary from all reviews (PETER/EPER convention, keeps comparability)
            'train' : vocabulary from training reviews only (stricter)
        """
        assert os.path.exists(data_path), data_path
        self.seq_len = seq_len
        self.his_len = his_len
        reviews = pickle.load(open(data_path, 'rb'))          # loaded once
        self.has_time = all(('time' in r and r['time'] is not None) for r in reviews)
        if history_scope == 'auto':
            history_scope = 'as_of' if self.has_time else 'train'
        if history_scope == 'as_of' and not self.has_time:
            raise ValueError("history_scope='as_of' needs a 'time' field in every record "
                             "(run add_timestamps.py first)")
        self.history_scope = history_scope

        train_index, valid_index, test_index = self.load_index(index_dir)
        self.rank = [UNUSED] * len(reviews)
        for i in train_index:
            self.rank[i] = TRAIN
        for i in valid_index:
            self.rank[i] = VALID
        for i in test_index:
            self.rank[i] = TEST

        # ---------------- vocabulary / entities ----------------
        self.word_dict = WordDictionary()
        self.user_dict = EntityDictionary()
        self.item_dict = EntityDictionary()
        self.max_rating, self.min_rating = float('-inf'), float('inf')
        for i, review in enumerate(reviews):
            self.user_dict.add_entity(review['user'])
            self.item_dict.add_entity(review['item'])
            self.max_rating = max(self.max_rating, review['rating'])
            self.min_rating = min(self.min_rating, review['rating'])
            if vocab_scope == 'train' and self.rank[i] != TRAIN:
                continue
            (fea, adj, tem, sco) = review['template']
            self.word_dict.add_sentence(tem)
            self.word_dict.add_word(fea)
            self.word_dict.add_word(adj)
        self.word_dict.keep_most_frequent(vocab_size)
        self.__unk = self.word_dict.word2idx['<unk>']
        bos, eos, pad = (self.word_dict.word2idx[t] for t in ('<bos>', '<eos>', '<pad>'))
        self.pad = pad

        # ---------------- encode all records ----------------
        self.feature_set = set()
        self.data = []
        all_text = []
        self.time = []
        for i, review in enumerate(reviews):
            (fea, adj, tem, sco) = review['template']
            ids = self.seq2ids(tem)
            rec = {'idx': i,
                   'user': self.user_dict.entity2idx[review['user']],
                   'item': self.item_dict.entity2idx[review['item']],
                   'rating': review['rating'],
                   'text': ids,
                   'feature': self.word_dict.word2idx.get(fea, self.__unk),
                   'adjective': self.word_dict.word2idx.get(adj, self.__unk)}
            self.data.append(rec)
            all_text.append(sentence_format(ids, seq_len, pad, bos, eos))
            self.time.append(review['time'] if self.has_time else i)
            self.feature_set.add(fea if fea in self.word_dict.word2idx else '<unk>')
        # (N, seq_len + 2) token matrix used to materialise histories
        self.all_text = torch.tensor(all_text, dtype=torch.long)
        # ids of feature words (used to restrict the aspect predictor to real aspects)
        self.feature_ids = sorted({self.word_dict.word2idx[f] for f in self.feature_set
                                   if f in self.word_dict.word2idx})

        self.train = [self.data[i] for i in train_index]
        self.valid = [self.data[i] for i in valid_index]
        self.test = [self.data[i] for i in test_index]

        # ---------------- per-user / per-item chronological indices ----------------
        self.user_hist = self._group('user')
        self.item_hist = self._group('item')

    # ----------------------------------------------------------------------------------
    def _group(self, key):
        groups = {}
        for rec in self.data:
            i = rec['idx']
            if self.rank[i] == UNUSED:
                continue
            if self.history_scope == 'train' and self.rank[i] != TRAIN:
                continue
            groups.setdefault(rec[key], []).append(i)
        out = {}
        for k, idxs in groups.items():
            idxs.sort(key=lambda j: (self.time[j], j))       # chronological (file order if no time)
            out[k] = (idxs, [self.time[j] for j in idxs])
        return out

    def history_ids(self, target_idx, side):
        """record ids visible as history for one target record, oldest -> newest."""
        rec = self.data[target_idx]
        groups = self.user_hist if side == 'user' else self.item_hist
        entry = groups.get(rec[side])
        if entry is None:
            return []
        idxs, times = entry
        if self.history_scope == 'leaky':                            # original pipeline: [-his_len:]
            return list(idxs[-self.his_len:])
        t_rank = self.rank[target_idx]
        if self.history_scope == 'as_of':
            end = bisect.bisect_left(times, self.time[target_idx])   # strictly earlier only
        else:
            end = len(idxs)                                          # train-only pool, no time
        out = []
        for p in range(end - 1, -1, -1):
            j = idxs[p]
            if j == target_idx or self.rank[j] > t_rank:
                continue
            out.append(j)
            if len(out) == self.his_len:
                break
        out.reverse()
        return out

    def build_histories(self, idx_batch, side):
        """returns (B, his_len, seq_len+2) LongTensor of history sentences and (B,) lengths."""
        B = len(idx_batch)
        hist = torch.full((B, self.his_len, self.all_text.size(1)), self.pad, dtype=torch.long)
        lengths = torch.zeros(B, dtype=torch.long)
        for b, t in enumerate(idx_batch.tolist()):
            ids = self.history_ids(t, side)
            if ids:
                hist[b, :len(ids)] = self.all_text[ids]
                lengths[b] = len(ids)
        return hist, lengths

    def check_leakage(self, split='test', n=1000, seed=0):
        """sanity check: target never in its own history, no future / higher-rank evidence."""
        data = getattr(self, split)
        rnd = random.Random(seed)
        sample = rnd.sample(data, min(n, len(data)))
        report = {'checked': len(sample), 'self_in_history': 0, 'future_in_history': 0,
                  'higher_rank_in_history': 0, 'identical_text_in_history': 0,
                  'empty_user_history': 0, 'empty_item_history': 0}
        for rec in sample:
            t = rec['idx']
            for side in ('user', 'item'):
                ids = self.history_ids(t, side)
                if not ids:
                    report['empty_%s_history' % side] += 1
                if t in ids:
                    report['self_in_history'] += 1
                if self.history_scope == 'as_of' and any(self.time[j] >= self.time[t] for j in ids):
                    report['future_in_history'] += 1
                if any(self.rank[j] > self.rank[t] for j in ids):
                    report['higher_rank_in_history'] += 1
                if any(self.data[j]['text'] == rec['text'] for j in ids):
                    report['identical_text_in_history'] += 1   # informative only (duplicate phrases)
        ok = (report['self_in_history'] == 0 and report['future_in_history'] == 0
              and report['higher_rank_in_history'] == 0)
        return ok, report

    def seq2ids(self, seq):
        return [self.word_dict.word2idx.get(w, self.__unk) for w in seq.split()]

    @staticmethod
    def load_index(index_dir):
        assert os.path.exists(index_dir), index_dir
        read = lambda name: [int(x) for x in open(os.path.join(index_dir, name)).readline().split()]
        return read('train.index'), read('validation.index'), read('test.index')


class Batchify:
    def __init__(self, data, word2idx, seq_len=15, batch_size=128, shuffle=False):
        bos, eos, pad = word2idx['<bos>'], word2idx['<eos>'], word2idx['<pad>']
        self.idx = torch.tensor([x['idx'] for x in data], dtype=torch.int64)
        self.user = torch.tensor([x['user'] for x in data], dtype=torch.int64)
        self.item = torch.tensor([x['item'] for x in data], dtype=torch.int64)
        self.rating = torch.tensor([x['rating'] for x in data], dtype=torch.float)
        self.seq = torch.tensor([sentence_format(x['text'], seq_len, pad, bos, eos) for x in data],
                                dtype=torch.int64)
        self.feature = torch.tensor([[x['feature']] for x in data], dtype=torch.int64)
        self.adjective = torch.tensor([[x['adjective']] for x in data], dtype=torch.int64)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sample_num = len(data)
        self.index_list = list(range(self.sample_num))
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0

    def next_batch(self):
        if self.step == self.total_step:
            self.step = 0
            if self.shuffle:
                random.shuffle(self.index_list)
        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        index = self.index_list[start:offset]
        return (self.user[index], self.item[index], self.rating[index], self.seq[index],
                self.feature[index], self.adjective[index], self.idx[index])


def now_time():
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '


def ids2tokens(ids, word2idx, idx2word):
    eos = word2idx['<eos>']
    tokens = []
    for i in ids:
        if i == eos:
            break
        tokens.append(idx2word[i])
    return tokens
