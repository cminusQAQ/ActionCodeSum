from tkinter import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

import random
    
class ActionWordGenerate(nn.Module):
    def __init__(self, args, encoder, action_word_map, device):
        super(ActionWordGenerate, self).__init__()
        self.args = args
        self.action_word_map = action_word_map
        self.device = device

        self.embedder = nn.Embedding(self.args.src_vocab_size, self.args.emsize, max_norm=True)
        # self.encoder = Encoder(args, self.embedder.enc_input_size)
        self.rnn = encoder
        self.classification = nn.Linear(args.nhid, len(action_word_map) // 2)
        self.sig = nn.Sigmoid()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, ex):
        code_word_rep = ex['code_word_rep'].to(self.device)
        code_len = ex['code_len'].to(self.device)
        # for i in range(ex['tgt_action_word'].shape[0]):
        #     # print(ex['tgt_action_word'])
        #     print(self.action_word_map[ex['tgt_action_word'][i].item()], end=' ')
        #     print(ex['summ_tokens'][i])
        code_rep = self.embedder(code_word_rep)
        a, b = self.rnn(code_rep)
        p = self.classification(torch.cat((b[0], b[1]), dim=1))
        loss = self.criterion(p, ex['tgt_action_word'].to(self.device))
    # if not self.training:
        # with open('test.txt', 'a+') as f:
        #     for indx, s in enumerate(ex['summ_tokens']):
        #         print('tgt: ', end=' ', file=f)
        #         print(self.action_word_map[ex['tgt_action_word'][indx].item()], end=' ', file=f)
        #         print('\nsrc: ', end=' ', file=f)
        #         value, index = p[indx].topk(3, dim=0, largest=True, sorted=True)
        #         for i, j in zip(value, index):
        #             print(self.action_word_map[j.item()], ':', i.item(), end=' ', file=f)
        #         print('\n', file=f)

        return p, loss

class ArgumentWordGenerate(nn.Module):
    def __init__(self, args, encoder, argument_word_map, device):
        super(ArgumentWordGenerate, self).__init__()
        self.args = args
        self.argument_word_map = argument_word_map
        self.device = device

        self.embedder = nn.Embedding(self.args.src_vocab_size, self.args.emsize, max_norm=True)
        # self.encoder = Encoder(args, self.embedder.enc_input_size)
        self.rnn = encoder
        self.classification = nn.Linear(args.nhid, len(argument_word_map) // 2)
        self.sig = nn.Sigmoid()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, ex):
        code_word_rep = ex['code_word_rep'].to(self.device)
        code_len = ex['code_len'].to(self.device)
        # for i in range(ex['tgt_argument_word'].shape[0]):
        #     # print(ex['tgt_argument_word'])
        #     print(self.argument_word_map[ex['tgt_argument_word'][i].item()], end=' ')
        #     print(ex['summ_tokens'][i])
        code_rep = self.embedder(code_word_rep)
        a, b = self.rnn(code_rep)
        p = self.classification(torch.cat((b[0], b[1]), dim=1))
        loss = self.criterion(p, ex['tgt_argument_word'].to(self.device))
    # if not self.training:
        # with open('test.txt', 'a+') as f:
        #     for indx, s in enumerate(ex['summ_tokens']):
        #         print('tgt: ', end=' ', file=f)
        #         print(self.argument_word_map[ex['tgt_argument_word'][indx].item()], end=' ', file=f)
        #         print('\nsrc: ', end=' ', file=f)
        #         value, index = p[indx].topk(3, dim=0, largest=True, sorted=True)
        #         for i, j in zip(value, index):
        #             print(self.argument_word_map[j.item()], ':', i.item(), end=' ', file=f)
        #         print('\n', file=f)

        return p, loss




class Attention(nn.Module):
    def __init__(self, node_features, attn_size):
        super(Attention, self).__init__()
        self.fc = nn.Linear(node_features, attn_size, bias=False)
        self.fc2 = nn.Linear(attn_size, 1, bias=False)

    def forward(self, s, enc_output):
        '''
            Input:
            - s: list of tensor, satisfied (B, L, emb_size/hidden)
            - enc_output: (B, L, hidden)
        '''
        y = torch.tanh(torch.sum(self.fc(torch.stack(s)), dim=0))
        # y: [batch_size, src_len, attn_size]
        w = F.softmax(self.fc2(y).squeeze(2), dim=1)
        # w: [batch_size, src_len]
        h = torch.bmm(w.unsqueeze(1), enc_output)
        # h: [batch_size, 1, hidden]
        return w, h




class DoubleDecoder(nn.Module):
    def __init__(self, args, type):
        super(DoubleDecoder, self).__init__()
        self.enc_dec_attn = Attention(args.emsize, args.nhid)
        self.word_attn = Attention(args.emsize, args.nhid)
        self.argument_attn = Attention(args.emsize, args.nhid)
        self.args = args
        self.type = type

        if type == 'seq2seq':
            self.rnn = nn.GRU(args.emsize + args.nhid, args.nhid, 
                            batch_first=True, dropout=self.args.dropout_rnn, bidirectional=True)
            self.fc = nn.Linear(args.nhid * 3 + args.emsize, args.tgt_vocab_size)
        else:
            self.rnn = nn.GRU(args.emsize + args.nhid * 3, args.nhid, 
                            batch_first=True, dropout=self.args.dropout_rnn, bidirectional=True)
            self.fc = nn.Linear(args.nhid * 5 + args.emsize, args.tgt_vocab_size)
            
        

    def forward(self, decoder_input, s, enc_output, code_len, word_input = None, argument_input=None):
        '''
        Input:
            - decoder_input: (batch_size, 1, decoder_input_size[emb_size])
            - s: (2, batch_size, hidden_size)
            - enc_output: (batch_size, src_len, nhid)
            - word_input: (batch_size, re_act_size, emb_size)
        Ouput:
            - pred: (batch_size, output_size)
            - h_n: (2, batch_size, hidden_size)
        '''

        _, enc_dec_output = self.enc_dec_attn([s[-1,:,:].unsqueeze(1).repeat(1, code_len, 1)] + 
                                        [enc_output] + 
                                        [decoder_input.repeat(1, code_len, 1)], enc_output)

        if self.type == 'seq2seq':  
            output, h_n = self.rnn(torch.cat((enc_dec_output, decoder_input), dim=2), s)
            pred = self.fc(torch.cat((output, enc_dec_output, decoder_input), dim=2)).squeeze(1)
        else:                  
            re_act_size = word_input.shape[1]
            _, word_output = self.word_attn([word_input] + 
                                            [decoder_input.repeat(1, re_act_size, 1)] + 
                                            [enc_dec_output.repeat(1, re_act_size, 1)], word_input) 
            _, argument_output = self.argument_attn([argument_input] + 
                                                    [decoder_input.repeat(1, re_act_size, 1)] +
                                                    [enc_dec_output.repeat(1, re_act_size, 1)] + 
                                                    [word_output.repeat(1, re_act_size, 1)], argument_input)

            output, h_n = self.rnn(torch.cat((enc_dec_output, word_output, argument_output, decoder_input), dim=2), s)
            pred = self.fc(torch.cat((output, enc_dec_output, word_output, argument_output, decoder_input), dim=2)).squeeze(1)
            
        return pred, h_n
        



class SummarizationGenerator(nn.Module):
    def __init__(self, args, src_dict, tgt_dict, action_word_re, action_word_map, argument_word_re, argument_word_map, type, device, re_act_size) -> None:
        super(SummarizationGenerator, self).__init__()
        self.args = args
        self.type = type
        self.device = device
        self.action_word_re = action_word_re
        self.argument_word_re = argument_word_re
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.action_word_map = action_word_map
        self.re_act_size = re_act_size

        self.encoder = nn.GRU(self.args.emsize, args.nhid // 2, bidirectional=True, batch_first=True, num_layers=2)
        self.word_pred = ActionWordGenerate(args, self.encoder, action_word_map, device)
        self.argument_pred = ArgumentWordGenerate(args, self.encoder, argument_word_map, device)
        self.encoder_embedder = nn.Embedding(self.args.src_vocab_size, self.args.emsize, max_norm=True)
        self.decoder_embedder = nn.Embedding(self.args.tgt_vocab_size, self.args.emsize, max_norm=True)
        self.decoder = DoubleDecoder(args, type)

        self.fc = nn.Linear(args.nhid * 2, args.nhid)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, ex, teacher_forcing_ratio=1, sampling=False):
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

        code_word_rep = ex['code_word_rep'].to(self.device)
        code_len = ex['code_len'].to(self.device)
        summ_word_rep = ex['summ_word_rep'].to(self.device)
        summ_len = ex['summ_len'].to(self.device)

        batch_size = code_word_rep.shape[0]
        h_act = None
        loss_act = 0
        # Run forward
        if self.type != 'seq2seq':
            p, loss_act = self.word_pred(ex)
            h_act = torch.zeros(batch_size, self.re_act_size, dtype=torch.int64).to(self.device)
            value, index = p.topk(self.re_act_size, dim=1, largest=True, sorted=True)
            index = index.tolist()
            for i in range(batch_size): 
                for j in range(self.re_act_size):
                    h_act[i][j] = self.action_word_re[index[i][j]]
            emb_act = self.decoder_embedder(h_act)

            p, loss_ag = self.argument_pred(ex)
            h_ag = torch.zeros(batch_size, self.re_act_size, dtype=torch.int64).to(self.device)
            value, index = p.topk(self.re_act_size, dim=1, largest=True, sorted=True)
            index = index.tolist()
            for i in range(batch_size):
                for j in range(self.re_act_size):
                    h_ag[i][j] = self.argument_word_re[index[i][j]]
            emb_ag = self.decoder_embedder(h_ag)
            
                
        
        # B x action_vacab_size
        code_rep = self.encoder_embedder(code_word_rep)

        memory_bank, hidden = self.encoder(code_rep)
        # hidden: (d(2) * layer, batch_size,  hidden / 2)
        # output: (batch_size, src_len , hidden)

        s = torch.cat((hidden[:2,:,:], hidden[2:,:,:]), dim=2)

        len = torch.max(summ_len).item()
        src_len = torch.max(code_len).item()
        
        output = torch.zeros(len, batch_size, self.args.tgt_vocab_size).to(self.device)
        summ_word_rep = summ_word_rep.transpose(0, 1)
        decoder_input = summ_word_rep[0]
        loss = 0
        res = torch.zeros(len, batch_size, dtype=torch.int64).to(self.device)

        for i in range(1, len):
            decoder_input = self.decoder_embedder(decoder_input.unsqueeze(1))

            if self.type != 'seq2seq':
                decoder_output, s = self.decoder(decoder_input,
                                                s,
                                                memory_bank,
                                                src_len,
                                                emb_act,
                                                emb_ag)
            else:
                decoder_output, s = self.decoder(decoder_input,
                                                s,
                                                memory_bank,
                                                src_len)
            
            logprob = torch.log_softmax(decoder_output, dim=-1)
            output[i] = decoder_output
            if sampling:
                top1 = torch.multinomial(torch.exp(logprob), 1).squeeze(1)
            else:
                top1 = decoder_output.argmax(1).detach()
            teacher_force = random.random() < teacher_forcing_ratio
            res[i] = top1
            if teacher_forcing_ratio == 0:
                assert teacher_force == False
            if sampling:
                assert teacher_force == False
            decoder_input = summ_word_rep[i] if teacher_force else top1


        loss = self.criterion(output.reshape(batch_size * len, self.args.tgt_vocab_size),
                          summ_word_rep.reshape(batch_size * len))
        output = torch.log_softmax(output, dim=-1)
        if teacher_forcing_ratio == 0 or sampling:
            return output.transpose(0, 1), res.transpose(0, 1), h_act, loss_act, h_ag, loss_ag
        else:
            return loss, res.transpose(0, 1), h_act, loss_act, h_ag, loss_ag

    def predict(self, ex):
        code_word_rep = ex['code_word_rep'].to(self.device)
        code_len = ex['code_len'].to(self.device)
        summ_word_rep = ex['summ_word_rep'].to(self.device)
        summ_len = ex['summ_len'].to(self.device)

        batch_size = code_word_rep.shape[0]
        h_act = None
        loss_act = 0
        # Run forward
        if self.type != 'seq2seq':
            p, loss_act = self.word_pred(ex)
            h_act = torch.zeros(batch_size, self.re_act_size, dtype=torch.int64).to(self.device)
            value, index = p.topk(self.re_act_size, dim=1, largest=True, sorted=True)
            index = index.tolist()
            for i in range(batch_size): 
                for j in range(self.re_act_size):
                    h_act[i][j] = self.action_word_re[index[i][j]]
            emb_act = self.decoder_embedder(h_act)

            p, loss_ag = self.argument_pred(ex)
            h_ag = torch.zeros(batch_size, self.re_act_size, dtype=torch.int64).to(self.device)
            value, index = p.topk(self.re_act_size, dim=1, largest=True, sorted=True)
            index = index.tolist()
            for i in range(batch_size):
                for j in range(self.re_act_size):
                    h_ag[i][j] = self.argument_word_re[index[i][j]]
            emb_ag = self.decoder_embedder(h_ag)
                
        
        # B x action_vacab_size
        code_rep = self.encoder_embedder(code_word_rep)

        memory_bank, hidden = self.encoder(code_rep)
        # hidden: (d(2) * layer, batch_size,  hidden / 2)
        # output: (batch_size, src_len , hidden)

        s = torch.cat((hidden[:2,:,:], hidden[2:,:,:]), dim=2)

        len = torch.max(summ_len).item()
        src_len = torch.max(code_len).item()
        
        summ_word_rep = summ_word_rep.transpose(0, 1)
        decoder_input = summ_word_rep[0]
        res = torch.zeros(len, batch_size, dtype=torch.int64).to(self.device)
        output = torch.zeros(len, batch_size, self.args.tgt_vocab_size).to(self.device)

        for i in range(1, len):
            decoder_input = self.decoder_embedder(decoder_input.unsqueeze(1))

            if self.type != 'seq2seq':
                decoder_output, s = self.decoder(decoder_input,
                                                s,
                                                memory_bank,
                                                src_len,
                                                emb_act,
                                                emb_ag)
            else:
                decoder_output, s = self.decoder(decoder_input,
                                                s,
                                                memory_bank,
                                                src_len)
            output[i] = decoder_output
            top1 = decoder_output.argmax(1).detach()
            res[i] = top1
            decoder_input = top1

        res = res.transpose(0, 1)
        pred = []
        for i in range(batch_size):
            batch_output = ['<s>']
            for j in range(1, len):
                batch_output.append(self.tgt_dict[res[i][j].item()])
                if self.tgt_dict[res[i][j].item()] == '</s>':
                    break
            pred.append(batch_output)

        return pred
    