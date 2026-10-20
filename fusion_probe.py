"""
Fusion probe for a trained DUSTER (concat fusion) checkpoint.

For test batches it measures, separately for the user token and the item token,
  * contribution ratio  ||W^h h|| / ||W^E E||   (how much of u_p / i_p comes from history)
  * mean pairwise cosine similarity, within a batch, of the history part W^h h
    and of the identifier part W^E E   (how similar the parts are across different users / items)

Usage (Colab, from the DUSTER_V2 folder):
  python fusion_probe.py --data_path .../reviews_time.pickle --index_dir .../temporal/ \
         --model runs/ta_none_tr1_s1/model.pt --cuda
"""
import argparse
import json
import torch
import torch.nn.functional as F
from utils import DataLoader, Batchify

ap = argparse.ArgumentParser()
ap.add_argument('--data_path', required=True)
ap.add_argument('--index_dir', required=True)
ap.add_argument('--model', required=True)
ap.add_argument('--batches', type=int, default=50)
ap.add_argument('--his_len', type=int, default=100)
ap.add_argument('--words', type=int, default=15)
ap.add_argument('--vocab_size', type=int, default=20000)
ap.add_argument('--history_scope', default='auto')
ap.add_argument('--cuda', action='store_true')
args = ap.parse_args()

device = torch.device('cuda' if args.cuda else 'cpu')
corpus = DataLoader(args.data_path, args.index_dir, args.vocab_size, seq_len=args.words,
                    his_len=args.his_len, history_scope=args.history_scope)
data = Batchify(corpus.test, corpus.word_dict.word2idx, args.words, 128)
model = torch.load(args.model, map_location=device, weights_only=False).to(device).eval()
d = model.emsize


def mean_pairwise_cos(x):
    x = F.normalize(x, dim=-1)
    s = x @ x.t()
    n = x.size(0)
    return ((s.sum() - s.diagonal().sum()) / (n * (n - 1))).item()


stats = {k: [] for k in ('user_ratio', 'item_ratio', 'user_cos_hist', 'user_cos_id',
                         'item_cos_hist', 'item_cos_id')}
with torch.no_grad():
    for _ in range(min(args.batches, data.total_step)):
        user, item, _, _, _, _, idx = data.next_batch()
        user, item = user.to(device), item.to(device)
        hu, lu = corpus.build_histories(idx, 'user')
        hi, li = corpus.build_histories(idx, 'item')
        h_u = model.user_encoder(hu.to(device), lu.to(device))
        h_i = model.item_encoder(hi.to(device), li.to(device))
        for side, h, e, W in (('user', h_u, model.user_embeddings(user), model.user_fuse.weight),
                              ('item', h_i, model.item_embeddings(item), model.item_fuse.weight)):
            ph = h @ W[:, :d].t()          # history part of the fused token
            pe = e @ W[:, d:].t()          # identifier part
            stats[side + '_ratio'].append((ph.norm(dim=-1) / pe.norm(dim=-1).clamp(min=1e-8)).mean().item())
            stats[side + '_cos_hist'].append(mean_pairwise_cos(ph))
            stats[side + '_cos_id'].append(mean_pairwise_cos(pe))

out = {k: sum(v) / len(v) for k, v in stats.items()}
wu, wi = model.user_fuse.weight, model.item_fuse.weight
out['Wu_hist/Wu_id'] = (wu[:, :d].norm() / wu[:, d:].norm()).item()
out['Wi_hist/Wi_id'] = (wi[:, :d].norm() / wi[:, d:].norm()).item()
print(json.dumps(out, indent=2))
