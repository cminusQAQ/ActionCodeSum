import tqdm
import sys
import logging
import pickle
import torch
import spacy

nlp = spacy.load("en_core_web_sm")
def get_action_word(s, action_word_map):
    tgt = torch.zeros(1)
    for i in range(1, 3):
        if s[i] in action_word_map:
            tgt[0] = action_word_map[s[i]]
    return tgt

def get_argument_word(s, argument_word_map):
    tgt = torch.zeros(1)
    for i in range(1, 3):
        if s[i] in argument_word_map:
            tgt[0] = argument_word_map[s[i]]
    return tgt

def get_action_word_list(vocab, tgt_dict):
    mp = {}
    for key in vocab:
        mp[vocab[key]] = tgt_dict[key]
    return mp

def get_argument_word_list(vocab, tgt_dict):
    mp = {}
    for key in vocab:
        mp[vocab[key]] = tgt_dict[key]
    return mp

def get_h_act(l, action_word_map, action_word_re, size, device):
    p = torch.zeros(len(l), size, dtype=torch.int64).to(device)
    for indx, s in enumerate(l):
        cnt = 0
        for indy, i in enumerate(s):
            if i in action_word_map:
                p[indx][cnt] = action_word_re[action_word_map[i]] 
                cnt += 1
                if cnt == size:
                    break
            if indy >= 3:
                break
        for indy in range(cnt, size):
            p[indx][indy] = 0
    return p

def get_h_argument(l, argument_word_map, argument_word_re, size, device):
    p = torch.zeros(len(l), size, dtype=torch.int64).to(device)
    for indx, s in enumerate(l):
        cnt = 0
        for indy, i in enumerate(s):
            if i in argument_word_map:
                p[indx][cnt] = argument_word_re[argument_word_map[i]] 
                cnt += 1
                if cnt == size:
                    break
            if indy >= 3:
                break
        for indy in range(cnt, size):
            p[indx][indy] = 0
    return p

if __name__ == '__main__':

    s =  ['return', 'returns', 'set', 'get', 'add', 'create', 'initialize', 'test', 'remove', 'check', 'is', 'call', 'retrieve', 'update', 'automate', 'write', 'determine', 'read', 'handle', 'to', 'if', 'insert', 'describe', 'use', 'load', 'delete', 'convert', 'start', 'clear', 'print', 'find', 'reset', 'save', 'send', 'generate', 'close', 'compare', 'indicate', 'perform', 'change', 'show']
    s = set(s)
    nlp = spacy.load("en_core_web_sm")
    with open('/home/hj/ActionWordCodeSum/data/java/train/javadoc.original') as f:
        for lines in tqdm.tqdm(f.readlines()):
            doc = nlp(lines)
            for ind, tokens in enumerate(doc):
                if tokens.tag_[:2] == 'VB' and tokens.text != 'null':
                    s.add(tokens.text)
                    break
                if ind >= 3:
                    break
    # with open('/home/hj/ActionWordCodeSum/data/python/test/javadoc.original') as f:
    #     for lines in tqdm.tqdm(f.readlines()):
    #         doc = nlp(lines)
    #         for tokens in doc:
    #             if tokens.tag_[:2] == 'VB':
    #                 s.add(tokens.text)
    #                 break
    # with open('/home/hj/ActionWordCodeSum/data/python/dev/javadoc.original') as f:
    #     for lines in tqdm.tqdm(f.readlines()):
    #         doc = nlp(lines)
    #         for tokens in doc:
    #             if tokens.tag_[:2] == 'VB':
    #                 s.add(tokens.text)
    #                 break
    print(len(s))
    s = list(s)
    s = ['null'] + s 
    with open('data/java/action_vocab.txt', 'wb') as f:
        pickle.dump(s, f)

    
