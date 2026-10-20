"""
DUSTER v2 - model.

Changes w.r.t. the original DUSTER / DUSTER_A module.py
-------------------------------------------------------
1. History encoder (HistoryEncoder, formerly DynamicReview):
   * the GRU output IS used (original DUSTER computed it and discarded it);
   * histories are packed with their true lengths, and attention pooling ignores padded reviews;
   * the "learnable mask" is kept but named what it is: a feature-wise gating vector g in R^d
     (element-wise scaling shared by all reviews), not a review-level gate.
2. The user history and the item history get their own encoder (own gate), as described in the paper.
   (DUSTER_A originally fed the USER history to the item encoder.)
3. The item ID embedding uses item_embeddings (originally user_embeddings(item)), and user / item
   fusion layers are separate.
4. The aspect predictor (DUSTER_A) is part of the same model: logits are restricted to the
   aspect (feature-word) vocabulary.
5. Histories are passed in already built by DataLoader.build_histories(), so the model never
   touches the corpus (no hidden data access inside forward()).
"""
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class HistoryEncoder(nn.Module):
    """TextCNN -> feature-wise gate -> GRU (+residual) -> masked attention pooling -> projection."""

    def __init__(self, ntoken, emsize, window_sizes=(3,), num_filters=100,
                 use_gate=True, use_gru=True, use_attn=True):
        super().__init__()
        self.review_embedding = nn.Embedding(ntoken, emsize)
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, (fs, emsize)) for fs in window_sizes])
        self.linear = nn.Linear(len(window_sizes) * num_filters, emsize)
        # feature-wise gating vector g (the "learnable mask" of the paper): x_l <- g * x_l
        self.feature_gate = nn.Parameter(torch.ones(emsize))
        self.gru = nn.GRU(emsize, emsize, batch_first=True)
        self.attention = nn.Sequential(nn.Linear(emsize, emsize), nn.Tanh(), nn.Linear(emsize, 1))
        self.out = nn.Linear(emsize, emsize)
        self.use_gate, self.use_gru, self.use_attn = use_gate, use_gru, use_attn

    def forward(self, hist, lengths):
        """hist: (B, L, T) token ids, oldest -> newest, padded at the end; lengths: (B,)"""
        B, L, T = hist.size()
        emb = self.review_embedding(hist.view(B * L, T)).unsqueeze(1)           # (B*L, 1, T, d)
        feats = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]
        feats = [F.max_pool1d(fm, fm.size(2)).squeeze(2) for fm in feats]
        x = self.linear(torch.cat(feats, dim=1)).view(B, L, -1)                 # (B, L, d)
        if self.use_gate:
            x = x * self.feature_gate

        valid = torch.arange(L, device=hist.device).unsqueeze(0) < lengths.unsqueeze(1)  # (B, L)
        x = x * valid.unsqueeze(-1)
        if self.use_gru:
            packed = pack_padded_sequence(x, lengths.clamp(min=1).cpu(), batch_first=True,
                                          enforce_sorted=False)
            h, _ = self.gru(packed)
            h, _ = pad_packed_sequence(h, batch_first=True, total_length=L)
            h = h + x                                                            # residual
        else:
            h = x
        if self.use_attn:
            scores = self.attention(h).squeeze(-1).masked_fill(~valid, -1e9)
            w = torch.softmax(scores, dim=1)
        else:                                                                    # mean pooling
            w = valid.float() / lengths.clamp(min=1).unsqueeze(1).float()
        pooled = (h * w.unsqueeze(-1)).sum(dim=1)
        pooled = pooled * (lengths > 0).float().unsqueeze(-1)                    # empty history -> 0
        return self.out(pooled)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        src2, attn = self.self_attn(src, src, src, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src, attn


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output, attns = src, []
        for mod in self.layers:
            output, attn = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attns.append(attn)
        if self.norm is not None:
            output = self.norm(output)
        return output, torch.stack(attns)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))

    def forward(self, x):
        return self.dropout(x + self.pe[:x.size(0), :])


class MLP(nn.Module):
    """rating head: r = W2 * sigmoid(W1 h + b1) + b2 (sigmoid is the hidden activation)."""

    def __init__(self, emsize=512):
        super().__init__()
        self.linear1 = nn.Linear(emsize, emsize)
        self.linear2 = nn.Linear(emsize, 1)
        self.sigmoid = nn.Sigmoid()
        for lin in (self.linear1, self.linear2):
            lin.weight.data.uniform_(-0.1, 0.1)
            lin.bias.data.zero_()

    def forward(self, hidden):
        return torch.squeeze(self.linear2(self.sigmoid(self.linear1(hidden))), -1)


def generate_square_subsequent_mask(total_len):
    return torch.tril(torch.ones(total_len, total_len)) == 0      # True = blocked


def generate_duster_mask(src_len, tgt_len):
    """left-to-right mask, but the src tokens [u, i, aspect(s)] attend to each other freely."""
    mask = generate_square_subsequent_mask(src_len + tgt_len)
    mask[:src_len, :src_len] = False
    return mask


class DUSTER(nn.Module):
    def __init__(self, src_len, tgt_len, pad_idx, nuser, nitem, ntoken, emsize, nhead, nhid, nlayers,
                 window_sizes=(3,), num_filters=100, dropout=0.2, duster_mask=True,
                 aspect_ids=None, use_user_hist=True, use_item_hist=True,
                 use_gate=True, use_gru=True, use_attn=True, fuse='concat', hist_dropout=0.0):
        super().__init__()
        self.fuse, self.hist_dropout = fuse, hist_dropout
        self.pos_encoder = PositionalEncoding(emsize, dropout)
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(emsize, nhead, nhid, dropout), nlayers)
        self.user_embeddings = nn.Embedding(nuser, emsize)
        self.item_embeddings = nn.Embedding(nitem, emsize)
        self.word_embeddings = nn.Embedding(ntoken, emsize)
        self.hidden2token = nn.Linear(emsize, ntoken)
        self.recommender = MLP(emsize)
        enc_kw = dict(window_sizes=window_sizes, num_filters=num_filters,
                      use_gate=use_gate, use_gru=use_gru, use_attn=use_attn)
        self.user_encoder = HistoryEncoder(ntoken, emsize, **enc_kw)
        self.item_encoder = HistoryEncoder(ntoken, emsize, **enc_kw)
        self.user_fuse = nn.Linear(2 * emsize, emsize)        # u_p = W[h_u ; E_u] + b
        self.item_fuse = nn.Linear(2 * emsize, emsize)        # i_p = W'[h_i ; E_i] + b'
        # residual fusions (v2.1): u_p = E_u + a_u * LN(h_u)   (rezero, a_u init 0)
        #                         u_p = E_u + sigmoid(W[E_u;h_u]+b) * W_h LN(h_u)  (gate, b init -2)
        self.hist_norm_u, self.hist_norm_i = nn.LayerNorm(emsize), nn.LayerNorm(emsize)
        self.alpha_u, self.alpha_i = nn.Parameter(torch.zeros(1)), nn.Parameter(torch.zeros(1))
        self.gate_u, self.gate_i = nn.Linear(2 * emsize, emsize), nn.Linear(2 * emsize, emsize)
        self.hproj_u, self.hproj_i = nn.Linear(emsize, emsize), nn.Linear(emsize, emsize)
        for g in (self.gate_u, self.gate_i):
            nn.init.zeros_(g.weight)
            nn.init.constant_(g.bias, -2.0)
        # aspect predictor heads (DUSTER_A); restricted to the aspect vocabulary
        self.aspect_head_u = nn.Linear(emsize, ntoken)
        self.aspect_head_i = nn.Linear(emsize, ntoken)
        # optional opinion-word (adjective) heads, used only with --aspect_tokens feature_adj
        self.adj_head_u = nn.Linear(emsize, ntoken)
        self.adj_head_i = nn.Linear(emsize, ntoken)
        allowed = torch.zeros(ntoken, dtype=torch.bool)
        if aspect_ids is not None:
            allowed[torch.tensor(aspect_ids, dtype=torch.long)] = True
        else:
            allowed[:] = True
        self.register_buffer('aspect_allowed', allowed)

        self.use_user_hist, self.use_item_hist = use_user_hist, use_item_hist
        self.ui_len = 2
        self.src_len = src_len
        self.pad_idx = pad_idx
        self.emsize = emsize
        self.attn_mask = (generate_duster_mask(src_len, tgt_len) if duster_mask
                          else generate_square_subsequent_mask(src_len + tgt_len))
        initrange = 0.1
        for emb in (self.user_embeddings, self.item_embeddings, self.word_embeddings):
            emb.weight.data.uniform_(-initrange, initrange)
        self.hidden2token.weight.data.uniform_(-initrange, initrange)
        self.hidden2token.bias.data.zero_()

    # ------------------------------------------------------------------ encoders
    def encode_ui(self, user, item, hist_u, len_u, hist_i, len_i):
        """returns u_p, i_p (B, d), masked aspect logits (a_u, a_i) and adjective logits (j_u, j_i)."""
        B = user.size(0)
        zeros = torch.zeros(B, self.emsize, device=user.device)
        h_u = self.user_encoder(hist_u, len_u) if self.use_user_hist else zeros
        h_i = self.item_encoder(hist_i, len_i) if self.use_item_hist else zeros
        fuse, hd = getattr(self, 'fuse', 'concat'), getattr(self, 'hist_dropout', 0.0)   # old checkpoints
        if self.training and hd > 0:            # drop the whole history per sample
            keep = (torch.rand(B, 1, device=user.device) >= hd).float()
            h_u, h_i = h_u * keep, h_i * (torch.rand(B, 1, device=user.device) >= hd).float()
        e_u, e_i = self.user_embeddings(user), self.item_embeddings(item)
        if fuse == 'concat':
            u_p = self.user_fuse(torch.cat((h_u, e_u), dim=-1))
            i_p = self.item_fuse(torch.cat((h_i, e_i), dim=-1))
        elif fuse == 'rezero':
            u_p = e_u + self.alpha_u * self.hist_norm_u(h_u)
            i_p = e_i + self.alpha_i * self.hist_norm_i(h_i)
        elif fuse == 'gate':
            g_u = torch.sigmoid(self.gate_u(torch.cat((e_u, h_u), dim=-1)))
            g_i = torch.sigmoid(self.gate_i(torch.cat((e_i, h_i), dim=-1)))
            u_p = e_u + g_u * self.hproj_u(self.hist_norm_u(h_u))
            i_p = e_i + g_i * self.hproj_i(self.hist_norm_i(h_i))
        else:
            raise ValueError(fuse)
        a_u = self.aspect_head_u(u_p).masked_fill(~self.aspect_allowed, -1e9)
        a_i = self.aspect_head_i(i_p).masked_fill(~self.aspect_allowed, -1e9)
        j_u, j_i = self.adj_head_u(u_p), self.adj_head_i(i_p)
        return u_p, i_p, (a_u, a_i, j_u, j_i)

    def fusion_stats(self):
        """how much the model relies on history (logged once per epoch)."""
        fuse = getattr(self, 'fuse', 'concat')
        if fuse == 'rezero':
            return {'alpha_u': self.alpha_u.item(), 'alpha_i': self.alpha_i.item()}
        if fuse == 'gate':
            return {'gate_bias_u': self.gate_u.bias.mean().item(), 'gate_bias_i': self.gate_i.bias.mean().item()}
        d = self.emsize
        wu, wi = self.user_fuse.weight, self.item_fuse.weight       # columns [:d] = history, [d:] = ID
        return {'Wu_hist/Wu_id': (wu[:, :d].norm() / wu[:, d:].norm()).item(),
                'Wi_hist/Wi_id': (wi[:, :d].norm() / wi[:, d:].norm()).item()}

    # ------------------------------------------------------------------ heads
    def predict_context(self, hidden):
        return F.log_softmax(self.hidden2token(hidden[1]), dim=-1)

    def predict_rating(self, hidden):
        return self.recommender(hidden[0])

    def predict_seq(self, hidden):
        return F.log_softmax(self.hidden2token(hidden[self.src_len:]), dim=-1)

    def generate_token(self, hidden):
        return F.log_softmax(self.hidden2token(hidden[-1]), dim=-1)

    def forward(self, u_p, i_p, text, seq_prediction=True, context_prediction=True,
                rating_prediction=True):
        """
        u_p, i_p : (B, d) from encode_ui()
        text     : (total_len - 2, B) = [aspect token(s); bos; e_1 ...]
        """
        device = u_p.device
        B = u_p.size(0)
        total_len = self.ui_len + text.size(0)
        attn_mask = self.attn_mask[:total_len, :total_len].to(device)
        left = torch.zeros(B, self.ui_len, dtype=torch.bool, device=device)
        right = text.t() == self.pad_idx
        key_padding_mask = torch.cat([left, right], 1)
        src = torch.cat([u_p.unsqueeze(0), i_p.unsqueeze(0), self.word_embeddings(text)], 0)
        src = self.pos_encoder(src * math.sqrt(self.emsize))
        hidden, attns = self.transformer_encoder(src, attn_mask, key_padding_mask)
        rating = self.predict_rating(hidden) if rating_prediction else None
        log_context_dis = self.predict_context(hidden) if context_prediction else None
        log_word_prob = self.predict_seq(hidden) if seq_prediction else self.generate_token(hidden)
        return log_word_prob, log_context_dis, rating, attns
