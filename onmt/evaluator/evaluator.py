#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import argparse
import codecs
import os
import math

import torch
from itertools import count
from collections import defaultdict

import onmt.model_builder
import onmt.inputters as inputters
import onmt.opts as opts


def build_evalutor(opt, report_score=True, logger=None, out_file=None):
    if out_file is None:
        out_file = codecs.open(opt.output, 'w', 'utf-8')

    if opt.gpu > -1:
        torch.cuda.set_device(opt.gpu)

    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    fields, model, model_opt = \
        onmt.model_builder.load_test_model(opt, dummy_opt.__dict__)

    scorer = onmt.translate.GNMTGlobalScorer(opt.alpha,
                                             opt.beta,
                                             opt.coverage_penalty,
                                             opt.length_penalty)

    kwargs = {k: getattr(opt, k)
              for k in ["beam_size", "n_best", "max_length", "min_length",
                        "stepwise_penalty", "block_ngram_repeat",
                        "ignore_when_blocking", "dump_beam",
                        "data_type", "replace_unk", "gpu", "verbose"]}

    translator = Evaluator(model, fields, global_scorer=scorer,
                           out_file=out_file, report_score=report_score,
                           copy_attn=model_opt.copy_attn, logger=logger,
                           **kwargs)
    return translator


class Evaluator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
       logger(logging.Logger): logger.
    """

    def __init__(self,
                 model,
                 fields,
                 beam_size,
                 n_best=1,
                 max_length=100,
                 global_scorer=None,
                 copy_attn=False,
                 logger=None,
                 gpu=False,
                 dump_beam="",
                 min_length=0,
                 stepwise_penalty=False,
                 block_ngram_repeat=0,
                 ignore_when_blocking=[],
                 sample_rate='16000',
                 window_size=.02,
                 window_stride=.01,
                 window='hamming',
                 use_filter_pred=False,
                 data_type="text",
                 replace_unk=False,
                 report_score=True,
                 report_bleu=False,
                 report_rouge=False,
                 verbose=False,
                 out_file=None):
        self.logger = logger
        self.gpu = gpu
        self.cuda = gpu > -1

        self.model = model
        self.fields = fields
        self.n_best = n_best
        self.max_length = max_length
        self.global_scorer = global_scorer
        self.copy_attn = copy_attn
        self.beam_size = beam_size
        self.min_length = min_length
        self.stepwise_penalty = stepwise_penalty
        self.dump_beam = dump_beam
        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = set(ignore_when_blocking)
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_stride = window_stride
        self.window = window
        self.use_filter_pred = use_filter_pred
        self.replace_unk = replace_unk
        self.data_type = data_type
        self.verbose = verbose
        self.out_file = out_file
        self.report_score = report_score
        self.report_bleu = report_bleu
        self.report_rouge = report_rouge

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None
        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    @staticmethod
    def merge_dict(father, son):
        for key in son:
            father[key] += son[key]

    def get_special(self, correct_dict, total_dict, prefix):
        correct, total = 0, 0
        for key in total_dict:
            _key = self.fields["tgt"].vocab.itos[key]
            if _key.startswith(prefix):
                correct += correct_dict[key]
                total += total_dict[key]

        return correct, total, correct / total

    def translate(self,
                  src_path=None,
                  src_data_iter=None,
                  tgt_path=None,
                  tgt_data_iter=None,
                  src_dir=None,
                  batch_size=None,
                  attn_debug=False):
        """
        Note: batch_size must not be None
        Note: one of ('src_path', 'src_data_iter') must not be None
        """
        if batch_size is None:
            raise ValueError("batch_size must be set")
        data = inputters.build_dataset(self.fields,
                                       self.data_type,
                                       src_path=src_path,
                                       src_data_iter=src_data_iter,
                                       tgt_path=tgt_path,
                                       tgt_data_iter=tgt_data_iter,
                                       src_dir=src_dir,
                                       sample_rate=self.sample_rate,
                                       window_size=self.window_size,
                                       window_stride=self.window_stride,
                                       window=self.window,
                                       use_filter_pred=self.use_filter_pred)

        data_iter = inputters.OrderedIterator(
            dataset=data, device=self.gpu,
            batch_size=batch_size, train=False, sort=False,
            sort_within_batch=True, shuffle=False)

        all_scores = []
        all_correct, all_total = defaultdict(int), defaultdict(int)

        done_job_count = 0

        for batch in data_iter:
            correct, total = self._run_target(batch, data)
            self.merge_dict(all_correct, correct)
            self.merge_dict(all_total, total)
            done_job_count += 1
            if done_job_count % 10 == 0:
                print("%s job has done... " % (done_job_count * batch_size))

        for _key in all_total:
            key = self.fields["tgt"].vocab.itos[_key]
            total = all_total[_key]
            correct = all_correct[_key]
            print("key: %s, total: %d, correct: %d, rate %f" % (key, total, correct, correct / total))

        print("Pitch: correct %d, total %d, acc %f" % self.get_special(all_correct, all_total, "P:"))
        print("Beat: correct %d, total %d, acc %f" % self.get_special(all_correct, all_total, "B:"))

        return all_scores

    def _run_target(self, batch, data):
        data_type = data.data_type
        if data_type == 'text':
            _, src_lengths = batch.src
        else:
            src_lengths = None
        src = inputters.make_features(batch, 'src', data_type)
        tgt_in = inputters.make_features(batch, 'tgt')[:-1]

        #  (1) run the encoder on the src
        enc_states, memory_bank = self.model.encoder(src, src_lengths)
        dec_states = \
            self.model.decoder.init_decoder_state(src, memory_bank, enc_states)

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        correct = defaultdict(int)
        total = defaultdict(int)

        dec_out, _, _ = self.model.decoder(
            tgt_in, memory_bank, dec_states, memory_lengths=src_lengths)

        for dec, tgt in zip(dec_out, batch.tgt[1:].data):
            # Log prob of each word.
            out = self.model.generator.forward(dec)
            for _x, _y in zip(out.max(1)[1], tgt):
                x = _x.item()
                y = _y.item()
                total[y] += 1
                if x == y:
                    correct[x] += 1

        return correct, total

    def _report_score(self, name, score_total, words_total):
        msg = ("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
            name, score_total / words_total,
            name, math.exp(-score_total / words_total)))
        return msg

    def _report_bleu(self, tgt_path):
        import subprocess
        path = os.path.split(os.path.realpath(__file__))[0]

        res = subprocess.check_output("perl %s/tools/multi-bleu.perl %s %s"
                                      % (path, tgt_path, self.output),
                                      stdin=self.out_file,
                                      shell=True).decode("utf-8")

        msg = ">> " + res.strip()
        return msg

    def _report_rouge(self, tgt_path):
        import subprocess
        path = os.path.split(os.path.realpath(__file__))[0]
        res = subprocess.check_output(
            "python %s/tools/test_rouge.py -r %s -c STDIN"
            % (path, tgt_path),
            shell=True,
            stdin=self.out_file).decode("utf-8")
        msg = res.strip()
        return msg
