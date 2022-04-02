# src: https://github.com/facebookresearch/DrQA/blob/master/scripts/reader/train.py

import sys

from preprocession import get_action_word, get_argument_word

sys.path.append(".")
sys.path.append("..")

import os
import json
import time
import pickle
import torch
import logging
import subprocess
import argparse
import numpy as np
import torch.optim as optim
import torch.nn as nn

import c2nl.config as config
import c2nl.inputters.utils as util
from c2nl.inputters import constants

from tqdm import tqdm
from c2nl.inputters.timer import AverageMeter, Timer
import c2nl.inputters.vector as vector
import c2nl.inputters.dataset as data
from graph4nlp.pytorch.modules.evaluation import BLEU, ROUGE, METEOR

from preprocession import get_action_word_list, get_h_act, get_argument_word_list, get_h_argument
from main.model import SummarizationGenerator, ActionWordGenerate
from discriminator import Discriminator
from c2nl.eval.bleu import corpus_bleu
from c2nl.eval.rouge import Rouge
from c2nl.eval.meteor import Meteor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'),
                         ['', 'K', 'M', 'B', 'T'][magnitude])


def pg_loss(prob, gt, reward):
    batch_size = gt.shape[0]
    step = gt.shape[1]
    assert len(prob.shape) == 3
    assert len(gt.shape) == 2
    assert prob.shape[0:2] == gt.shape[0:2]
    mask = 1 - gt.data.eq(0).float()
    prob_select = torch.gather(prob.contiguous().view(batch_size*step, -1), 1, gt.contiguous().view(-1, 1))
    
    prob_select = prob_select.view_as(gt)
    prob_select.masked_fill_(mask=(1 - mask).bool(), value=0)
    loss = - torch.sum(prob_select*reward.unsqueeze(1)) / prob_select.shape[0]
    return loss


def add_train_args(parser):
    """Adds commandline arguments pertaining to training a model. These
    are different from the arguments dictating the model architecture.
    """
    parser.register('type', 'bool', str2bool)

    # Runtime environment;
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--data_workers', type=int, default=5,
                         help='Number of subprocesses for data loading')
    runtime.add_argument('--random_seed', type=int, default=1013,
                         help=('Random seed for all numpy/torch/cuda '
                               'operations (for reproducibility)'))
    runtime.add_argument('--num_epochs', type=int, default=40,
                         help='Train data iterations')
    runtime.add_argument('--batch_size', type=int, default=32,
                         help='Batch size for training')
    runtime.add_argument('--test_batch_size', type=int, default=128,
                         help='Batch size during validation/testing')
    runtime.add_argument('--generator_pretrain_epoch', type=int, default=40,
                         help='PreTrain data iterations')
    runtime.add_argument('--discriminator_pretrain_epoch', type=int, default=40,
                         help='PreTrain data iterations')

    # Files
    files = parser.add_argument_group('Filesystem')
    files.add_argument('--dataset_name', nargs='+', type=str, required=True,
                       help='Name of the experimental dataset')
    files.add_argument('--model_dir', type=str, default='/tmp/qa_models/',
                       help='Directory for saved models/checkpoints/logs')
    files.add_argument('--model_name', type=str, default='',
                       help='Unique model identifier (.mdl, .txt, .checkpoint)')
    files.add_argument('--data_dir', type=str, default='/data/',
                       help='Directory of training/validation data')
    files.add_argument('--train_src', nargs='+', type=str,
                       help='Preprocessed train source file')
    files.add_argument('--train_src_tag', nargs='+', type=str,
                       help='Preprocessed train source tag file')
    files.add_argument('--train_tgt', nargs='+', type=str,
                       help='Preprocessed train target file')
    files.add_argument('--dev_src', nargs='+', type=str, required=True,
                       help='Preprocessed dev source file')
    files.add_argument('--dev_src_tag', nargs='+', type=str,
                       help='Preprocessed dev source tag file')
    files.add_argument('--dev_tgt', nargs='+', type=str, required=True,
                       help='Preprocessed dev target file')

    # Saving + loading
    save_load = parser.add_argument_group('Saving/Loading')
    save_load.add_argument('--checkpoint', type='bool', default=False,
                           help='Save model + optimizer state after each epoch')
    save_load.add_argument('--pretrained', type=str, default=None,
                           help='Path to a pretrained model to warm-start with')

    # Data preprocessing
    preprocess = parser.add_argument_group('Preprocessing')
    preprocess.add_argument('--max_examples', type=int, default=-1,
                            help='Maximum number of examples for training')
    preprocess.add_argument('--uncase', type='bool', default=False,
                            help='Code and summary words will be lower-cased')
    preprocess.add_argument('--src_vocab_size', type=int, default=None,
                            help='Maximum allowed length for src dictionary')
    preprocess.add_argument('--tgt_vocab_size', type=int, default=None,
                            help='Maximum allowed length for tgt dictionary')
    preprocess.add_argument('--max_characters_per_token', type=int, default=30,
                            help='Maximum number of characters allowed per token')
    preprocess.add_argument('--ctype', type=str, default='seq2seq', dest='ctype',
                            help='The type which will be choose')

    # General
    general = parser.add_argument_group('General')
    general.add_argument('--valid_metric', type=str, default='bleu',
                         help='The evaluation metric used for model selection')
    general.add_argument('--display_iter', type=int, default=25,
                         help='Log state after every <display_iter> batches')
    general.add_argument('--sort_by_len', type='bool', default=True,
                         help='Sort batches by length for speed')
    general.add_argument('--only_test', type='bool', default=False,
                         help='Only do testing')

    # Log results Learning
    log = parser.add_argument_group('Log arguments')
    log.add_argument('--print_copy_info', type='bool', default=False,
                     help='Print copy information')
    log.add_argument('--print_one_target', type='bool', default=False,
                     help='Print only one target sequence')


def set_defaults(args):
    """Make sure the commandline arguments are initialized properly."""
    # Check critical files exist
    if not args.only_test:
        args.train_src_files = []
        args.train_tgt_files = []
        args.train_src_tag_files = []

        num_dataset = len(args.dataset_name)
        if num_dataset > 1:
            if len(args.train_src) == 1:
                args.train_src = args.train_src * num_dataset
            if len(args.train_tgt) == 1:
                args.train_tgt = args.train_tgt * num_dataset
            if len(args.train_src_tag) == 1:
                args.train_src_tag = args.train_src_tag * num_dataset

        for i in range(num_dataset):
            dataset_name = args.dataset_name[i]
            data_dir = os.path.join(args.data_dir, dataset_name)
            train_src = os.path.join(data_dir, args.train_src[i])
            train_tgt = os.path.join(data_dir, args.train_tgt[i])
            if not os.path.isfile(train_src):
                raise IOError('No such file: %s' % train_src)
            if not os.path.isfile(train_tgt):
                raise IOError('No such file: %s' % train_tgt)
            if args.use_code_type:
                train_src_tag = os.path.join(data_dir, args.train_src_tag[i])
                if not os.path.isfile(train_src_tag):
                    raise IOError('No such file: %s' % train_src_tag)
            else:
                train_src_tag = None

            args.train_src_files.append(train_src)
            args.train_tgt_files.append(train_tgt)
            args.train_src_tag_files.append(train_src_tag)

    args.dev_src_files = []
    args.dev_tgt_files = []
    args.dev_src_tag_files = []

    num_dataset = len(args.dataset_name)
    if num_dataset > 1:
        if len(args.dev_src) == 1:
            args.dev_src = args.dev_src * num_dataset
        if len(args.dev_tgt) == 1:
            args.dev_tgt = args.dev_tgt * num_dataset
        if len(args.dev_src_tag) == 1:
            args.dev_src_tag = args.dev_src_tag * num_dataset

    for i in range(num_dataset):
        dataset_name = args.dataset_name[i]
        data_dir = os.path.join(args.data_dir, dataset_name)
        dev_src = os.path.join(data_dir, args.dev_src[i])
        dev_tgt = os.path.join(data_dir, args.dev_tgt[i])
        if not os.path.isfile(dev_src):
            raise IOError('No such file: %s' % dev_src)
        if not os.path.isfile(dev_tgt):
            raise IOError('No such file: %s' % dev_tgt)
        if args.use_code_type:
            dev_src_tag = os.path.join(data_dir, args.dev_src_tag[i])
            if not os.path.isfile(dev_src_tag):
                raise IOError('No such file: %s' % dev_src_tag)
        else:
            dev_src_tag = None

        args.dev_src_files.append(dev_src)
        args.dev_tgt_files.append(dev_tgt)
        args.dev_src_tag_files.append(dev_src_tag)

    # Set model directory
    subprocess.call(['mkdir', '-p', args.model_dir])

    # Set model name
    if not args.model_name:
        import uuid
        import time
        args.model_name = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]

    # Set log + model file names
    suffix = '_test' if args.only_test else ''
    args.model_file = os.path.join(args.model_dir, args.model_name + '.mdl')
    args.log_file = os.path.join(args.model_dir, args.model_name + suffix + '.txt')
    args.pred_file = os.path.join(args.model_dir, args.model_name + suffix + '.json')
    if args.pretrained:
        args.pretrained = os.path.join(args.model_dir, args.pretrained + '.mdl')

    args.fix_embeddings = False

    return args

# ------------------------------------------------------------------------------
# Initalization from scratch.
# ------------------------------------------------------------------------------
class Trainer:
    def __init__(self, args) -> None:
        self.args = args
        self.logger = logging.getLogger()
        # Set logging
        self.logger.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                                '%m/%d/%Y %I:%M:%S %p')
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        self.logger.addHandler(console)
        if self.args.log_file:
            if self.args.checkpoint:
                logfile = logging.FileHandler(self.args.log_file, 'a')
            else:
                logfile = logging.FileHandler(self.args.log_file, 'w')
            logfile.setFormatter(fmt)
            self.logger.addHandler(logfile)
        self.logger.info('COMMAND: %s' % ' '.join(sys.argv))

    def convert(self, tokens):
        pred = []
        for i in range(tokens.shape[0]):
            batch_output = '<s>'
            for j in range(1, tokens.shape[1]):
                batch_output += self.tgt_dict[tokens[i][j].item()] + ' '
                if self.tgt_dict[tokens[i][j].item()] == '</s>':
                    break
            pred.append(batch_output)
        return pred

    def constring(self, tokens):
        pred = []
        for i in tokens:
            s = ''
            for j in i:
                #if j != '<s>' and j != '</s>':
                    s += j + ' '
            pred.append(s)
        return pred

    def init_from_scratch(self, train_exs, dev_exs):
        """New model, new data, new dictionary."""
        # Build a dictionary from the data questions + words (train/dev splits)
        self.logger.info('-' * 100)
        self.logger.info('Build word dictionary')
        self.src_dict = util.build_word_and_char_dict(self.args,
                                                examples=train_exs + dev_exs,
                                                fields=['code'],
                                                dict_size=self.args.src_vocab_size,
                                                no_special_token=True)
        self.tgt_dict = util.build_word_and_char_dict(self.args,
                                                examples=train_exs + dev_exs,
                                                fields=['summary'],
                                                dict_size=self.args.tgt_vocab_size,
                                                no_special_token=False)
        print(self.tgt_dict['</s>'])
        print('!!!!!!!!!!!!!!!!!!')
        self.logger.info('Num words in source = %d and target = %d' % (len(self.src_dict), len(self.tgt_dict)))
        # Initialize model
        with open(args.data_dir + args.dataset_name[0] + '/action_vocab.txt', 'rb+') as f:
            action_word_list = pickle.load(f)
        self.action_word_map = {}
        size = 0
        for ind, w in enumerate(action_word_list):
            if w in self.tgt_dict:
                self.action_word_map[w] = size
                self.action_word_map[size] = w
                size += 1
        self.action_word_re = get_action_word_list(self.action_word_map, self.tgt_dict)

        with open(args.data_dir + args.dataset_name[0] + '/argument_vocab.txt', 'rb+') as f:
            argument_word_list = pickle.load(f)
        self.argument_word_map = {}
        size = 0
        for ind, w in enumerate(argument_word_list):
            if w in self.tgt_dict:
                self.argument_word_map[w] = size
                self.argument_word_map[size] = w
                size += 1
        self.argument_word_re = get_argument_word_list(self.argument_word_map, self.tgt_dict)
        self.generator = SummarizationGenerator(config.get_model_args(self.args), self.src_dict, self.tgt_dict,
                                        self.action_word_re, 
                                        self.action_word_map, 
                                        self.argument_word_re,
                                        self.argument_word_map,
                                        self.args.ctype, device, 1).to(device)

        for name, param in self.generator.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal(param)
        
        self.discriminator = Discriminator(config.get_model_args(self.args), 1, device).to(device)
        for name, param in self.discriminator.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal(param)

        
    def save_generator(self, save_dir, epoch):
        self.generator.eval()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        model_name = save_dir + '/' + self.args.model_name + '_model_generator-{}.tar'.format(epoch)
        # torch.save(self.generator, model_name)
        torch.save({'model_state_dict': self.generator.state_dict(),
                    'optimizer_state_dict': self.generator_optimizer.state_dict(),
                    'action_word_re': self.action_word_re,
                    'action_word_map': self.action_word_map,
                    'argument_word_re': self.argument_word_re,
                    'argument_word_map': self.argument_word_map,
                    'src_dict': self.src_dict, 
                    'tgt_dict': self.tgt_dict
                   }, model_name)
        

    def load_pretrain_generator(self, epoch, save_dir, name):
        model_name = save_dir + '/' + name + '_model_generator-{}.tar'.format(epoch)
        checkpoint = torch.load(model_name)
        self.generator.load_state_dict(checkpoint['model_state_dict'])
        self.generator_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.generator.action_word_map = checkpoint['action_word_map']
        self.generator.action_word_re = checkpoint['action_word_re']
        self.generator.argument_word_map = checkpoint['argument_word_map']
        self.generator.argument_word_re = checkpoint['argument_word_re']
        self.generator.src_dict = self.src_dict = checkpoint['src_dict']
        self.generator.tgt_dict = self.tgt_dict = checkpoint['tgt_dict']
        self.generator.eval()
        
    # def pretrain_generator(self, dataloader, epoch, lr):
    #     start = time.time()
    #     num_batches = len(dataloader.dataset) // self.args.batch_size
    #     self.qwq.train()

    #     loss_generator_collect = []
    #     loss_lm_collect = []
    #     loss_adv_collect = []
    #     loss_vh_collect = []
    #     scheduler = optim.lr_scheduler.MultiStepLR(self.generator_optimizer,milestones=[20,80],gamma = 0.9)
    #     for step, data in enumerate(dataloader):
            
    #         p, loss = self.qwq(data)

    #         self.generator_optimizer.zero_grad()
    #         loss.backward()
    #         scheduler.step()
            
    #         loss_generator_collect.append(loss.item())
    #         loss_lm_collect.append(loss.item())

    #         if step % 100 == 0 and step != 0:
    #             end = time.time()
    #             self.logger.info(
    #                 "step {}/{} (epoch {}), Pre_training, generator_loss = {:.4f}, "
    #                 "lm_loss = {:.4f}, action_word_loss = {:.4f}, lr = {:.5f}, time/batch = {:.4f}"
    #                 .format(step, num_batches, epoch,
    #                 np.mean(loss_generator_collect), np.mean(loss_lm_collect),
    #                 np.mean(loss_vh_collect), float(lr), end - start))
                
    #             loss_generator_collect = []
    #             loss_lm_collect = []
    #             loss_vh_collect = []
    #             start = time.time()
    
    def pretrain_generator(self, dataloader, epoch, lr):
        self.generator.train()

        start = time.time()
        num_batches = len(dataloader.dataset) // self.args.batch_size

        loss_generator_collect = []
        loss_lm_collect = []
        loss_ag_collect = []
        loss_vh_collect = []

        for step, data in enumerate(dataloader):
        
            loss_lm, output, p_act, loss_vh, p_ag, loss_ag = self.generator(data)

            loss = loss_lm + loss_vh * 0.1 + loss_ag * 0.1
            self.generator_optimizer.zero_grad()
            loss.backward()

            self.generator_optimizer.step()
            
            loss_generator_collect.append(loss.item())
            loss_lm_collect.append(loss_lm.item())
            if self.args.ctype != 'seq2seq':
                loss_vh_collect.append(loss_vh.item())
                loss_ag_collect.append(loss_ag.item())

            if step % 100 == 0 and step != 0:
                end = time.time()
                self.logger.info(
                    "step {}/{} (epoch {}), Pre_training, generator_loss = {:.4f}, "
                    "lm_loss = {:.4f}, action_word_loss = {:.4f}, argument_word_loss = {:.4f}, lr = {:.5f}, time/batch = {:.4f}"
                    .format(step, num_batches, epoch,
                    np.mean(loss_generator_collect), np.mean(loss_lm_collect),
                    np.mean(loss_vh_collect), np.mean(loss_ag_collect), float(lr), end - start))
                
                loss_generator_collect = []
                loss_ag_collect = []
                loss_lm_collect = []
                loss_vh_collect = []
                start = time.time()


    @torch.no_grad()
    def evaluate_generator(self, data_loader, mode, epoch):
        self.generator.eval()
        eval_time = Timer()
        bl_total = []
        rg_total = []
        mt_total = []
        with torch.no_grad():
            pbar = tqdm(data_loader)
            for idx, ex in enumerate(pbar):
                pred = self.generator.predict(ex)
                pred = self.constring(pred)
                lab = self.constring(ex['summ_tokens'])
                for i, j in zip(pred, lab):
                    with open('text.txt', 'w+') as f:
                        print('test: ', i, file=f)
                        print('truth:', j, file=f)
                bleu = BLEU(n_grams=[1, 2, 3, 4])
                bl, _ = bleu.calculate_scores(ground_truth=lab, predict=pred)
                bl_total.append(bl[3])
                if idx >= 100:
                    break
                # rouge = ROUGE()
                # rg, _ = rouge.calculate_scores(ground_truth=lab, predict=pred)
                # rg_total.append(rg)
                # meteor = METEOR()
                # mt, _ = meteor.calculate_scores(ground_truth=lab, predict=pred)
                # mt_total.append(mt)


                
        self.logger.info('dev valid official: Epoch = %d | ' % (epoch) + 
                         'bleu = %.4f ' % (np.mean(bl_total) * 100) + 
                         'rouge = %.4f ' % (np.mean(rg_total) * 100) + 
                         'meteor = %.4f ' % (np.mean(mt_total) * 100) + 
                         'valid time = %.4f (s)' % eval_time.time())
        return np.mean(bl_total)
                
    def pretrain_discriminator(self, epoch, lr):
        self.discriminator.train()
        self.generator.train()
        start = time.time()
        loss_discriminator_collect = []
        loss_generator_collect = []
        loss_lm_collect = []
        loss_adv_collect = []
        for step, data in enumerate(self.train_loader):

            logit, output, p_act, loss_vh, p_ag, loss_ag = self.generator(data, teacher_forcing_ratio=0, sampling=True)
            
            # real
            labels = torch.ones(data['code_word_rep'].shape[0]).to(device)
            loss_real = self.discriminator(data['code_word_rep'].to(device), 
                                        p_act, output, labels)

            ## fake
            # sampled_results = logits.argmax(2)
            labels = torch.zeros(data['code_word_rep'].shape[0]).to(device)
            
            p_act = get_h_act(data['summ_tokens'], self.action_word_map, self.action_word_re, 1, device)
            loss_fake = self.discriminator(data['code_word_rep'].to(device),
                                            p_act, 
                                            data['summ_word_rep'].to(device),
                                            labels)
            
            loss_discriminator = (loss_real + loss_fake) / 2
            self.discriminator_optimizer.zero_grad()
            loss_discriminator.backward()
            self.discriminator_optimizer.step()
            loss_discriminator_collect.append(loss_discriminator.item())


            if step % 100 == 0 and step != 0:
                end = time.time()
                self.logger.info(
                    "[Adversarial Training][Discriminator] step {}/{} (epoch {}), discriminator_loss = {:.4f}, generator_loss = {:.4f}, "
                    "lm_loss = {:.4f}, adv_loss = {:.4f}, lr = {:.5f}, time/batch = {:.4f}"
                    .format(step, len(self.train_loader), epoch, np.mean(loss_discriminator_collect),
                    np.mean(loss_generator_collect), np.mean(loss_lm_collect),
                    np.mean(loss_adv_collect), float(lr), end - start))
                
                loss_discriminator_collect = []
                loss_generator_collect = []
                loss_lm_collect = []
                loss_adv_collect = []
                start = time.time()
        
        pass


    def train_generator(self, train_loader, dev_loader):
        best_bleu = 0
        best_epoch = 0
       #  self.load_pretrain_generator(epoch=1, save_dir=self.args.model_dir)
        for epoch in range(self.args.generator_pretrain_epoch):
            self.pretrain_generator(dataloader=train_loader, epoch=epoch,
                            lr=self.args.learning_rate)
            res = self.evaluate_generator(dev_loader, 'dev', epoch=epoch)
            if best_bleu < res:

                best_bleu = res
                best_epoch = epoch
                self.logger.info("BLEU metric enhanced !!!!")
                if self.args.checkpoint:
                    self.save_generator(save_dir=self.args.model_dir, epoch=epoch)
 
       
        if self.args.checkpoint:
            self.load_pretrain_generator(epoch=13, save_dir=self.args.model_dir, name="python-ag-aw")
        self.evaluate_generator(dev_loader, 'dev', epoch=-1)

    def test(self, train_loader, dev_loader):
        best_bleu = -1
        best_epoch = -1
        self.load_pretrain_generator(epoch=21, save_dir=self.args.model_dir)
        for epoch in range(1):
            # self.evaluate_generator(dev_loader, 'dev', epoch=epoch)
            self.pretrain_generator(dataloader=train_loader, epoch=epoch,
                            lr=self.args.learning_rate)
            res = self.evaluate_generator(dev_loader, 'dev', epoch=epoch)
            if best_bleu < res:

                best_bleu = res
                best_epoch = epoch
                self.logger.info("BLEU metric enhanced !!!!")
                self.save_generator(save_dir=self.args.model_dir, epoch=epoch)
                    
        self.load_pretrain_generator(epoch=53, save_dir=self.args.model_dir)
        self.evaluate_generator(dev_loader, 'dev', epoch=-1)

    def train_generator_pg(self, epoch, lr):
        self.discriminator.train()
        self.generator.train()
        start = time.time()
        loss_generator_collect = []
        loss_lm_rl_collect = []
        loss_vh_rl_collect = []
        for step, data in enumerate(self.train_loader):
            
            batch_size = data['code_len'].shape[0]

            loss_lm, output, p_act, loss_vh, p_ag, loss_ag = self.generator(data)
            
            loss_first = (loss_lm + loss_vh * 0.1) * 0.2
            self.generator_optimizer.zero_grad()
            #loss_first.backward()
            
            # baseline: 
            with torch.no_grad():
                _, res, p_act, loss_aw, p_ag, loss_ag = self.generator(data, teacher_forcing_ratio=0)
                sentence_baseline = self.convert(res)
            
                logit = self.discriminator(data['code_word_rep'].to(device),
                                            p_act, 
                                            res)
                reward_cons_baseline = torch.sigmoid(logit)
            
            # explore
            logits, res, p_act, loss_vh_rl, p_ag, loss_ag = self.generator(data, teacher_forcing_ratio=0, sampling=True)
            sentence_explore = self.convert(res)
            with torch.no_grad():
                logit = self.discriminator(data['code_word_rep'].to(device),
                                            p_act,
                                            res)
                reward_cons_explore = torch.sigmoid(logit)

            reward_cons = reward_cons_explore - reward_cons_baseline

            # calculate
            question_str = self.constring(data['summ_tokens'])
            bleu4_metric = []
            bleu = BLEU(n_grams=[1, 2, 3, 4])
            for i in range(batch_size):
                bleu4_baseline, _ = bleu.calculate_scores(ground_truth=[question_str[i]], predict=[sentence_baseline[i]])
                bleu4_explore, _ = bleu.calculate_scores(ground_truth=[question_str[i]], predict=[sentence_explore[i]])
                reward = bleu4_explore[3] - bleu4_baseline[3]
                bleu4_metric.append(reward)
                
            reward_bleu = torch.Tensor(bleu4_metric).to(device)
                

            reward_bleu_norm = reward_bleu
            reward_cons_norm = reward_cons
            # reward = reward_bleu_norm + reward_cons_norm.detach() * 0.5
            # print(reward)
            reward = reward_bleu_norm.detach()

            loss_lm_rl = pg_loss(prob=logits, gt=res.detach(), reward=reward.detach())

            loss_vh_rl = torch.mean(loss_vh_rl * reward.detach())
            #loss_gen = 0.8 * ( loss_lm_rl + loss_vh_rl*0.1*0.1) + 0.2 * loss_first

            #loss2 = 0.8 * ( loss_lm_rl + loss_vh_rl*0.1 * 0.1)
            loss2 = loss_lm_rl
            loss_gen = loss2
            loss2.backward()

            self.generator_optimizer.step()
            
            loss_generator_collect.append(loss_gen.item())
            loss_lm_rl_collect.append(loss_lm_rl.item())
            loss_vh_rl_collect.append(loss_vh_rl.item())

            if step % 100 == 0 and step != 0:
                end = time.time()
                self.logger.info(
                    "[Adversarial Training][Generator] step {}/{} (epoch {}), generator_loss = {:.4f}, "
                    "lm_loss = {:.4f}, visual_hint_loss = {:.4f}, lr = {:.5f}, time/batch = {:.4f}"
                    .format(step, len(self.train_loader), epoch,
                    np.mean(loss_generator_collect), np.mean(loss_lm_rl_collect),
                    np.mean(loss_vh_rl_collect), float(lr), end - start))
                
                loss_generator_collect = []
                loss_lm_rl_collect = []
                loss_vh_rl_collect = []
                start = time.time()
                break


    def evaluate_discriminator(self, dataloader, mode, epoch):
        self.discriminator.eval()
        self.generator.eval()
        correct_fake = 0
        all_cnt_fake = 0
        correct_gt = 0
        all_cnt_gt = 0
        start = time.time()
        with torch.no_grad():
            for step, data in enumerate(dataloader):
                batch_size = data['code_len'].shape[0]
                _, res, p_act, loss_vh = self.generator(data, teacher_forcing_ratio=0, sampling=True)
                logit = self.discriminator(data['code_word_rep'].to(device),
                                            p_act,
                                            res)
                pred = torch.sigmoid(logit) >= 0.5
                correct_fake += (pred == False).sum()
                all_cnt_fake += pred.shape[0]

                p_act = get_h_act(data['summ_tokens'], self.action_word_map, self.action_word_re, 1, device)
                logit = self.discriminator(data['code_word_rep'].to(device),
                                            p_act,
                                            data['code_word_rep'].to(device))

                pred = torch.sigmoid(logit) >= 0.5
                correct_gt += (pred == True).sum()
                all_cnt_gt += pred.shape[0]
            end = time.time()

            self.logger.info("*********** Evaluation, split ***********")
            self.logger.info("Time cost: {:.4f}".format(end - start))

            self.logger.info("Discriminator Accuracy Generator: {}".format(correct_fake / all_cnt_fake))
            self.logger.info("Discriminator Accuracy GT: {}".format(correct_gt / all_cnt_gt))
        return (correct_fake + correct_gt) / (all_cnt_fake + all_cnt_gt)

    def train_discriminator(self):
        best_acc = -1
        best_epoch = -1
        for epoch in range(self.args.discriminator_pretrain_epoch):
            self.pretrain_discriminator(epoch=epoch, lr=self.args.learning_rate)
            
            acc = self.evaluate_discriminator(self.dev_loader, mode='dev', epoch=epoch)
            if best_acc < acc:
                best_acc = acc
                best_epoch = epoch
            #     self.save_discriminator(save_dir=self.args.model_dir, epoch=epoch)
            #     with open(self.args.model_dir + self.args.model_name + "discriminator_best_epoch.pkl", "wb") as f:
            #         pickle.dump(best_epoch, f)

        # with open(self.args.model_dir + self.args.model_name + "discriminator_best_epoch.pkl", "rb") as f:
        #     best_epoch = pickle.load(f)
        # self.load_pretrain_discriminator(epoch=best_epoch, save_dir=self.args.model_dir)
        # acc = self.evaluate_discriminator(split="test", epoch=-1)
    # ------------------------------------------------------------------------------
    # Train loop.
    # ------------------------------------------------------------------------------


    def train(self, train_loader, dev_loader):
        # print(self.args.ctype)
        self.train_generator(train_loader, dev_loader)
        if self.args.ctype != 'discriminator':
            return
        self.train_discriminator()
        best_epoch = -1
        best_bleu4 = -1
        for epoch in range(100):
            self.train_generator_pg(epoch, lr=self.args.learning_rate)
            score = self.evaluate_generator(train_loader, 'train', epoch)
            # self.evaluate_discriminator(split="val", epoch=epoch)
            # self.save_generator(save_dir=self.opt.generator_checkpoint_path, epoch=epoch)
            # self.save_discriminator(save_dir=self.opt.discriminator_checkpoint_path, epoch=epoch)
            if score > best_bleu4:
                best_bleu4 = score
                best_epoch = epoch
                self.logger.info("Updated in val !!!!")
                self.evaluate_generator(dev_loader, 'dev', epoch)
                    # self.evaluate_discriminator(split="test", epoch=best_epoch)
                
            self.pretrain_discriminator(epoch, lr=self.args.learning_rate)
            # self.evaluate_discriminator(split="val", epoch=epoch)


    # ------------------------------------------------------------------------------
    # Main.
    # ------------------------------------------------------------------------------


    def main(self):
        # --------------------------------------------------------------------------
        # DATA
        self.logger.info('-' * 100)
        self.logger.info('Load and process data files')

        train_exs = []
        if not self.args.only_test:
            self.args.dataset_weights = dict()
            for train_src, train_src_tag, train_tgt, dataset_name in \
                    zip(self.args.train_src_files, self.args.train_src_tag_files,
                        self.args.train_tgt_files, self.args.dataset_name):
                train_files = dict()
                train_files['src'] = train_src
                train_files['src_tag'] = train_src_tag
                train_files['tgt'] = train_tgt
                exs = util.load_data(self.args,
                                    train_files,
                                    max_examples=self.args.max_examples,
                                    dataset_name=dataset_name)
                lang_name = constants.DATA_LANG_MAP[dataset_name]
                self.args.dataset_weights[constants.LANG_ID_MAP[lang_name]] = len(exs)
                train_exs.extend(exs)

            self.logger.info('Num train examples = %d' % len(train_exs))
            self.args.num_train_examples = len(train_exs)
            for lang_id in self.args.dataset_weights.keys():
                weight = (1.0 * self.args.dataset_weights[lang_id]) / len(train_exs)
                self.args.dataset_weights[lang_id] = round(weight, 2)
            self.logger.info('Dataset weights = %s' % str(self.args.dataset_weights))

        dev_exs = []
        for dev_src, dev_src_tag, dev_tgt, dataset_name in \
                zip(self.args.dev_src_files, self.args.dev_src_tag_files,
                    self.args.dev_tgt_files, self.args.dataset_name):
            dev_files = dict()
            dev_files['src'] = dev_src
            dev_files['src_tag'] = dev_src_tag
            dev_files['tgt'] = dev_tgt
            exs = util.load_data(self.args,
                                dev_files,
                                max_examples=self.args.max_examples,
                                dataset_name=dataset_name,
                                test_split=True)
            dev_exs.extend(exs)
        self.logger.info('Num dev examples = %d' % len(dev_exs))

        # --------------------------------------------------------------------------
        # MODEL
        self.logger.info('-' * 100)
        start_epoch = 1
        self.init_from_scratch(train_exs, dev_exs)

        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=2e-5)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=1e-3)

        if os.path.exists(self.args.data_dir + self.args.dataset_name[0] + '/train/train_action_word.txt'):
            with open(self.args.data_dir + self.args.dataset_name[0] + '/train/train_action_word.txt', 'rb') as f:
                l = pickle.load(f)
            for ind, j in enumerate(tqdm(train_exs)):
                j['tgt_action_word'] = l[ind]
        else:
            l = []
            for ind, j in enumerate(tqdm(train_exs)):
                j['tgt_action_word'] = get_action_word(j['summary'].tokens, self.action_word_map)
                l.append(j['tgt_action_word'])
            with open(self.args.data_dir + self.args.dataset_name[0] + '/train/train_action_word.txt', 'wb') as f:
                pickle.dump(l, file=f)
        
        if os.path.exists(self.args.data_dir + self.args.dataset_name[0] + '/dev/dev_action_word.txt'):
            with open(self.args.data_dir + self.args.dataset_name[0] + '/dev/dev_action_word.txt', 'rb') as f:
                l = pickle.load(f)
            for ind, j in enumerate(tqdm(dev_exs)):
                j['tgt_action_word'] = l[ind]
        else:
            l = []
            for ind, j in enumerate(tqdm(dev_exs)):
                j['tgt_action_word'] = get_action_word(j['summary'].tokens, self.action_word_map)
                l.append(j['tgt_action_word'])
            with open(self.args.data_dir + self.args.dataset_name[0] + '/dev/dev_action_word.txt', 'wb') as f:
                pickle.dump(l, file=f)

        if os.path.exists(self.args.data_dir + self.args.dataset_name[0] + '/train/train_argument.txt'):
            with open(self.args.data_dir + self.args.dataset_name[0] + '/train/train_argument.txt', 'rb') as f:
                l = pickle.load(f)
            for ind, j in enumerate(tqdm(train_exs)):
                j['tgt_argument_word'] = l[ind]
        else:
            l = []
            for ind, j in enumerate(tqdm(train_exs)):
                j['tgt_argument_word'] = get_argument_word(j['summary'].tokens, self.argument_word_map)
                l.append(j['tgt_argument_word'])
            with open(self.args.data_dir + self.args.dataset_name[0] + '/train/train_argument.txt', 'wb') as f:
                pickle.dump(l, file=f)

        if os.path.exists(self.args.data_dir + self.args.dataset_name[0] + '/dev/dev_argument_word.txt'):
            with open(self.args.data_dir + self.args.dataset_name[0] + '/dev/dev_argument_word.txt', 'rb') as f:
                l = pickle.load(f)
            for ind, j in enumerate(tqdm(dev_exs)):
                j['tgt_argument_word'] = l[ind]
        else:
            l = []
            for ind, j in enumerate(tqdm(dev_exs)):
                j['tgt_argument_word'] = get_argument_word(j['summary'].tokens, self.argument_word_map)
                l.append(j['tgt_argument_word'])
            with open(self.args.data_dir + self.args.dataset_name[0] + '/dev/dev_argument_word.txt', 'wb') as f:
                pickle.dump(l, file=f)

        # --------------------------------------------------------------------------
        # DATA ITERATORS
        # Two datasets: train and dev. If we sort by length it's faster.
        self.logger.info('-' * 100)
        self.logger.info('Make data loaders')

        if not self.args.only_test:
            train_dataset = data.CommentDataset(train_exs, self.generator)
            if self.args.sort_by_len:
                train_sampler = data.SortedBatchSampler(train_dataset.lengths(),
                                                        self.args.batch_size,
                                                        shuffle=True)
            else:
                train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)

            self.train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                sampler=train_sampler,
                num_workers=self.args.data_workers,
                collate_fn=vector.batchify,
                pin_memory=self.args.cuda,
                drop_last=self.args.parallel
            )

        dev_dataset = data.CommentDataset(dev_exs, self.generator)
        dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)

        self.dev_loader = torch.utils.data.DataLoader(
            dev_dataset,
            batch_size=self.args.test_batch_size,
            sampler=dev_sampler,
            num_workers=self.args.data_workers,
            collate_fn=vector.batchify,
            pin_memory=self.args.cuda,
            drop_last=self.args.parallel
        )

        # debug(train_loader, dev_loader, generator)
        # return 
        # -------------------------------------------------------------------------
        # PRINT CONFIG
        self.logger.info('-' * 100)
        self.logger.info('CONFIG:\n%s' %
                    json.dumps(vars(self.args), indent=4, sort_keys=True))


        # --------------------------------------------------------------------------
        # TRAIN/VALID LOOP
        self.logger.info('-' * 100)
        self.logger.info('Starting training...')
        stats = {'timer': Timer(), 'epoch': start_epoch, 'best_valid': 0, 'no_improvement': 0}
        self.train(self.train_loader, self.dev_loader)


if __name__ == '__main__':
    # Parse cmdline self.args and setup environment
    parser = argparse.ArgumentParser(
        'Code to Natural Language Generation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_train_args(parser)
    config.add_model_args(parser)
    args = parser.parse_args()
    set_defaults(args)

    # Set cuda
    args.cuda = torch.cuda.is_available()
    args.parallel = torch.cuda.device_count() > 1

    trainer = Trainer(args)

    trainer.main()


