from unicodedata import bidirectional
from c2nl.models.seq2seq import Embedder, Encoder
from model import Attention
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, args, re_act_size, device):
        super().__init__()
        self.args = args
        self.embedder = Embedder(args).to(device)
        self.code_encoder = Encoder(args, self.embedder.enc_input_size)
        self.text_encoder = Encoder(args, self.embedder.enc_input_size)
        self.re_act_size = re_act_size
        self.attn = Attention(args.emsize, args.nhid)
        self.pred = nn.Sequential(nn.Linear(args.nhid * 2, args.nhid),
                                nn.Tanh(),
                                nn.Dropout(args.dropout),
                                nn.Linear(args.nhid, 1))
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, code, action_word, summarization, labels=None):
        '''
            Input:
            - code: (B, code_L)
            - action_word: (B, 5)
            - summarization: (B, sum_L)
            Ouput:
            if label:
                - logit: (B, 1)
            else:
                - loss

        '''
        code_emb = self.embedder(code, mode='encoder')
        code_hidden, code_memory_bank = self.code_encoder(code_emb, None)
        # code_memory_bank: (B, L, hidden)
        # code_hidden: (2* layer, B, hidden/2)
        code_hidden = code_hidden[0]
        code_hidden = torch.cat((code_hidden[-1,:,:], code_hidden[-2,:,:]), dim=1)
      #  print(summarization)
        summ_emb = self.embedder(summarization, mode='decoder')
        summ_hidden, summ_memory_bank = self.text_encoder(summ_emb, None)
        summ_hidden = summ_hidden[0]
        summ_hidden = torch.cat((summ_hidden[-1,:,:], summ_hidden[-2,:,:]), dim=1)

        action_word_emb = self.embedder(action_word, mode='decoder')
        _, res = self.attn([summ_hidden.unsqueeze(1).repeat(1, self.re_act_size, 1)] + [action_word_emb], action_word_emb)
        # res: (B, 1, hid)
        res = res.squeeze(1)

        logit = self.pred(torch.cat((res, code_hidden), dim=-1)).squeeze(1)
        with open('test.txt', 'w+') as f:
            print(logit, file=f)
        if labels is not None:
            loss = self.loss_func(logit, labels)
            return loss
        return logit