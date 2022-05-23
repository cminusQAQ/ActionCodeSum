from tkinter import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

from c2nl.models.transformer import Embedder, Transformer, SingleEncoder

from c2nl.utils.copy_utils import align, collapse_copy_scores, make_src_map, replace_unknown
from c2nl.utils.misc import tens2sen
    
class ActionWordGenerate(nn.Module):
    def __init__(self, args, action_word_map, device):
        super(ActionWordGenerate, self).__init__()
        self.args = args
        self.action_word_map = action_word_map
        self.device = device
        if len(args.max_relative_pos) != args.nlayers:
            assert len(args.max_relative_pos) == 1
            args.max_relative_pos = args.max_relative_pos * args.nlayers
        self.embedder = Embedder(args)
        self.encoder = SingleEncoder(args, self.embedder.enc_input_size)
        self.classification = nn.Linear(args.emsize, len(action_word_map) // 2)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, code_word_rep, code_len, tgt_action_word):
        code_rep = self.embedder(code_word_rep, None, None, mode='encoder')
        a, b = self.encoder(code_rep, code_len)
        # print(code_rep.shape)
        # print(a.shape, len(b))
        # assert 0
        a = self.classification(a[:,-1,:])
        loss = self.criterion(a, tgt_action_word)

        return a.detach(), loss

class ArgumentWordGenerate(nn.Module):
    def __init__(self, args, argument_word_map, device):
        super(ArgumentWordGenerate, self).__init__()
        self.args = args
        self.argument_word_map = argument_word_map
        self.device = device

        if len(args.max_relative_pos) != args.nlayers:
            assert len(args.max_relative_pos) == 1
            args.max_relative_pos = args.max_relative_pos * args.nlayers
        self.embedder = Embedder(args)
        self.encoder = SingleEncoder(args, self.embedder.enc_input_size)
        self.classification = nn.Linear(args.emsize, len(argument_word_map) // 2)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, code_word_rep, code_len, tgt_argument_word):
        code_rep = self.embedder(code_word_rep, None, None, mode='encoder')
        a, b = self.encoder(code_rep, code_len)
        a = self.classification(a[:,-1,:])
        loss = self.criterion(a, tgt_argument_word)
        # TODO:
        return a.detach(), loss

class SummarizationGenerator(nn.Module):
    def __init__(self, args, src_dict, tgt_dict, action_word_re, action_word_map, argument_word_re, argument_word_map, type, device, re_act_size) -> None:
        super(SummarizationGenerator, self).__init__()
        self.args = args
        self.type = type
        self.parallel = False
        self.use_cuda = device
        self.args.use_cuda = device
        self.device = device
        self.action_word_re = action_word_re
        self.argument_word_re = argument_word_re
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.args.tgt_vocab_size = len(tgt_dict)
        self.args.src_vocab_size = len(src_dict)
        self.action_word_map = action_word_map
        self.re_act_size = re_act_size
        if self.type != 'seq2seq':
            # self.word_pred = ActionWordGenerate(args, action_word_map, device)
            # self.argument_pred = ArgumentWordGenerate(args, argument_word_map, device)
            self.embedder = Embedder(args)
        self.network = Transformer(self.args, tgt_dict)

    def forward(self, ex, teacher_forcing_ratio=0.5, sampling=False, beam_search=False, epoch=-1):
        '''
            Output:
                if sampling:
                - output: (len, batch_size, vocab_size)
                else:
                - loss: scalar(int)
                - res: (batch_size, len)
                - h_act: (batch_size, 1)
                - loss_act: scalar(int)
                
        '''

        code_word_rep = ex['code_word_rep'].to(self.use_cuda)
        code_len = ex['code_len'].to(self.use_cuda)
        summ_word_rep = ex['summ_word_rep'].to(self.use_cuda)
        summ_len = ex['summ_len'].to(self.use_cuda)
        code_char_rep = None
        code_type_rep = None
        code_mask_rep = ex['code_mask_rep']
        summ_char_rep = ex['summ_char_rep']
        tgt_seq = ex['tgt_seq'].to(self.use_cuda)
        tgt_action_word = ex['tgt_action_word'].to(self.use_cuda)
        tgt_argument_word = ex['tgt_argument_word'].to(self.use_cuda)

        if any(l is None for l in ex['language']):
            ex_weights = None
        else:
            ex_weights = [self.args.dataset_weights[lang] for lang in ex['language']]
            ex_weights = torch.FloatTensor(ex_weights).to(self.use_cuda)

        batch_size = code_word_rep.shape[0]
        h_act = None
        emb_act = None
        loss_act = 0
        h_ag = None
        emb_ag = None
        loss_ag = 0
        # Run forward
        if self.type != 'seq2seq':
            # p, loss_act = self.word_pred(code_word_rep, code_len, tgt_action_word)
            h_act = torch.zeros(batch_size, self.re_act_size, dtype=torch.int64).to(self.device)
            # value, index = p.topk(self.re_act_size, dim=1, largest=True, sorted=True)
            # index = index.tolist()
            for i in range(batch_size):
                for j in range(self.re_act_size):
                    #h_act[i][j] = self.action_word_re[index[i][j]]
                    h_act[i][j] = self.action_word_re[tgt_action_word[i].item()]
            emb_act = self.embedder(h_act, None, None, mode='encoder')

            # p, loss_ag = self.argument_pred(code_word_rep, code_len, tgt_argument_word)
            h_ag = torch.zeros(batch_size, self.re_act_size, dtype=torch.int64).to(self.device)
            # value, index = p.topk(self.re_act_size, dim=1, largest=True, sorted=True)
            # index = index.tolist()
            for i in range(batch_size):
                for j in range(self.re_act_size):
                    #h_ag[i][j] = self.argument_word_re[index[i][j]]
                    h_ag[i][j] = self.argument_word_re[tgt_argument_word[i].item()]
            emb_ag = self.embedder(h_ag, None, None, mode='encoder')
        
        source_map, alignment = None, None
        blank, fill = None, None
        # To enable copy attn, collect source map and alignment info
        if self.args.copy_attn:
            assert 'src_map' in ex and 'alignment' in ex

            source_map = make_src_map(ex['src_map'])
            source_map = source_map.cuda(non_blocking=True) if self.use_cuda \
                else source_map

            alignment = align(ex['alignment'])
            alignment = alignment.cuda(non_blocking=True) if self.use_cuda \
                else alignment

            blank, fill = collapse_copy_scores(self.tgt_dict, ex['src_vocab'])

        # Run forward
        net_loss = self.network(code_word_rep=code_word_rep,
                                code_char_rep=code_char_rep,
                                code_type_rep=code_type_rep,
                                code_len=code_len,
                                summ_word_rep=summ_word_rep,
                                summ_char_rep=summ_char_rep,
                                summ_len=summ_len,
                                tgt_seq=tgt_seq,
                                src_map=source_map,
                                alignment=alignment,
                                src_dict=self.src_dict,
                                tgt_dict=self.tgt_dict,
                                max_len=self.args.max_tgt_len,
                                blank=blank,
                                fill=fill,
                                source_vocab=ex['src_vocab'],
                                code_mask_rep=code_mask_rep,
                                example_weights=ex_weights,
                                action_word_emb=emb_act,
                                argument_word_emb=emb_ag,
                                epoch=epoch,
                                )
        loss = net_loss['ml_loss']
        return loss, None, h_act, loss_act, h_ag, loss_ag

    def predict(self, ex, replace_unk=False):
        code_word_rep = ex['code_word_rep'].to(self.use_cuda)
        code_len = ex['code_len'].to(self.use_cuda)
        code_char_rep = ex['code_char_rep']
        code_type_rep = ex['code_type_rep']
        code_mask_rep = ex['code_mask_rep']
        tgt_action_word = ex['tgt_action_word'].to(self.use_cuda)
        tgt_argument_word = ex['tgt_argument_word'].to(self.use_cuda)

        batch_size = code_word_rep.shape[0]
        h_act = None
        emb_act = None
        loss_act = 0
        h_ag = None
        emb_ag = None
        loss_ag = 0
        # Run forward
        if self.type != 'seq2seq':
            # p, loss_act = self.word_pred(code_word_rep, code_len, tgt_action_word)
            h_act = torch.zeros(batch_size, self.re_act_size, dtype=torch.int64).to(self.device)
            # value, index = p.topk(self.re_act_size, dim=1, largest=True, sorted=True)
            # index = index.tolist()
            for i in range(batch_size):
                for j in range(self.re_act_size):
                    #h_act[i][j] = self.action_word_re[index[i][j]]
                    h_act[i][j] = self.action_word_re[tgt_action_word[i].item()]
            emb_act = self.embedder(h_act, None, None, mode='encoder')

            # p, loss_ag = self.argument_pred(code_word_rep, code_len, tgt_argument_word)
            h_ag = torch.zeros(batch_size, self.re_act_size, dtype=torch.int64).to(self.device)
            # value, index = p.topk(self.re_act_size, dim=1, largest=True, sorted=True)
            # index = index.tolist()
            for i in range(batch_size):
                for j in range(self.re_act_size):
                    #h_ag[i][j] = self.argument_word_re[index[i][j]]
                    h_ag[i][j] = self.argument_word_re[tgt_argument_word[i].item()]
            emb_ag = self.embedder(h_ag, None, None, mode='encoder')
                
        
        source_map, alignment = None, None
        blank, fill = None, None
        # To enable copy attn, collect source map and alignment info
        if self.args.copy_attn:
            assert 'src_map' in ex and 'alignment' in ex

            source_map = make_src_map(ex['src_map'])
            source_map = source_map.cuda(non_blocking=True) if self.use_cuda \
                else source_map

            blank, fill = collapse_copy_scores(self.tgt_dict, ex['src_vocab'])

        decoder_out = self.network(code_word_rep=code_word_rep,
                                   code_char_rep=code_char_rep,
                                   code_type_rep=code_type_rep,
                                   code_len=code_len,
                                   summ_word_rep=None,
                                   summ_char_rep=None,
                                   summ_len=None,
                                   tgt_seq=None,
                                   src_map=source_map,
                                   alignment=alignment,
                                   max_len=self.args.max_tgt_len,
                                   src_dict=self.src_dict,
                                   tgt_dict=self.tgt_dict,
                                   blank=blank, fill=fill,
                                   source_vocab=ex['src_vocab'],
                                   code_mask_rep=code_mask_rep,
                                   action_word_emb=emb_act,
                                   argument_word_emb=emb_ag,
                                   )

        predictions = tens2sen(decoder_out['predictions'],
                               self.tgt_dict,
                               ex['src_vocab'])
        if self.type != 'seq2seq':
            act = tens2sen(h_act, self.tgt_dict, ex['src_vocab'])
            ag = tens2sen(h_ag, self.tgt_dict, ex['src_vocab'])
        else:
            act = None
            ag = None
        
        if replace_unk:
            for i in range(len(predictions)):
                enc_dec_attn = decoder_out['attentions'][i]
                if self.args.model_type == 'transformer':
                    assert enc_dec_attn.dim() == 3
                    enc_dec_attn = enc_dec_attn.mean(1)
                predictions[i] = replace_unknown(predictions[i],
                                                 enc_dec_attn,
                                                 src_raw=ex['code_tokens'][i])
                if self.args.uncase:
                    predictions[i] = predictions[i].lower()

        return predictions, act, ag

    