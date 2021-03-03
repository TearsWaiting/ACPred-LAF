# ---encoding:utf-8---
# @Time : 2020.11.09
# @Author : Waiting涙
# @Email : 1773432827@qq.com
# @IDE : PyCharm
# @File : ACPred-LAF-Basic.py


import torch
import torch.nn as nn
import numpy as np
from configuration import config
import pickle
from util import util_freeze


def get_attn_pad_mask(seq):
    batch_size, seq_len = seq.size()
    pad_attn_mask = seq.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
    pad_attn_mask_expand = pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]
    return pad_attn_mask_expand


class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding (look-up table)
        self.pos_embed = nn.Embedding(max_len, d_model)  # position embedding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        seq_len = x.size(1)  # x: [batch_size, seq_len]
        pos = torch.arange(seq_len, device=device, dtype=torch.long)  # [seq_len]
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]
        embedding = self.pos_embed(pos)
        embedding = embedding + self.tok_embed(x)
        embedding = self.norm(embedding)
        return embedding


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_head, seq_len, seq_len]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)  # [batch_size, n_head, seq_len, seq_len]
        context = torch.matmul(attn, V)  # [batch_size, n_head, seq_len, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_head)
        self.W_K = nn.Linear(d_model, d_k * n_head)
        self.W_V = nn.Linear(d_model, d_v * n_head)

        self.linear = nn.Linear(n_head * d_v, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_head, d_k).transpose(1, 2)  # q_s: [batch_size, n_head, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_head, d_k).transpose(1, 2)  # k_s: [batch_size, n_head, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_head, d_v).transpose(1, 2)  # v_s: [batch_size, n_head, seq_len, d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_head, 1, 1)
        context, attention_map = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            n_head * d_v)  # context: [batch_size, seq_len, n_head * d_v]
        output = self.linear(context)
        output = self.norm(output + residual)
        return output, attention_map


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(self.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
        self.attention_map = None

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attention_map = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                                        enc_self_attn_mask)  # enc_inputs to same Q,K,V
        self.attention_map = attention_map
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs


class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()

        global max_len, n_layers, n_head, d_model, d_ff, d_k, d_v, vocab_size, device
        max_len = config.max_len
        n_layers = config.num_layer
        n_head = config.num_head
        d_model = config.dim_embedding
        d_ff = config.dim_feedforward
        d_k = config.dim_k
        d_v = config.dim_v
        vocab_size = config.vocab_size
        device = torch.device("cuda" if config.cuda else "cpu")

        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc_task = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2),
        )
        self.classifier = nn.Linear(2, 2)

    def forward(self, input_ids):
        output = self.embedding(input_ids)  # [bach_size, seq_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(input_ids)  # [batch_size, maxlen, maxlen]
        for layer in self.layers:
            output = layer(output, enc_self_attn_mask)
            # output: [batch_size, max_len, d_model]

        # classification
        # only use [CLS]
        representation = output[:, 0, :]
        reduction_feature = self.fc_task(representation)
        reduction_feature = reduction_feature.view(reduction_feature.size(0), -1)
        logits_clsf = self.classifier(reduction_feature)
        representation = reduction_feature
        return logits_clsf, representation
