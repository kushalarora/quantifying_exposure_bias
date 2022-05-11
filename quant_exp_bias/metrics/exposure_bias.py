from overrides import overrides
from typing import List

from allennlp.training.metrics.metric import Metric
from quant_exp_bias.oracles.oracle_base import Oracle

import logging
import torch
import numpy as np
import random
import math
from functools import reduce, partial

from nltk import ngrams
from collections import defaultdict, Counter


import torch.nn.functional as F

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

EPSILON = 1e-200 

REPEAT_CONTEXT_LENGTHS = [16, 32, 128, 512]

def ngram_metrics(token_list, pad=1):
    stats = defaultdict(float)
    for n in range(1, 5):
        ngs = [ng for ng in ngrams(token_list, n)]
        counter = Counter([ng for ng in ngrams(token_list, n)])
        stats[n] = 1.0 - len(counter)/len(ngs)
    return stats


def repeat_at_1(predictions, targets, context_length):
    with torch.no_grad():
        targets = targets.unsqueeze(0)
        T = targets.size(1)

        # T x T where prev_targets[t, :] = [y_1,...,y_t-1, -1, -1,..., -1]
        prev_targets = targets.expand(T, T).tril().masked_fill_(torch.ones_like(targets.expand(T, T)).byte().triu().bool(), -1)

        # each row t is [-1, ..., -1, y_{t-k-1}, ..., y_{t-1}, -1, ..., -1] where k is context length
        prev_targets = prev_targets.masked_fill_(torch.ones_like(targets.expand(T, T)).byte().tril(-(context_length+1)).bool(), -1)

        repeat_at_1 = (predictions[:, None] == prev_targets)
        has_repeat_at_1 = repeat_at_1.sum(1).gt(0)
        total_repeat_at_1 = has_repeat_at_1.sum().item()

        is_incorrect = (predictions != targets.view(-1)).view(-1, 1)
        total_wrong_repeat_at_1 = ((repeat_at_1 * is_incorrect).sum(1).gt(0)).sum().item()

        total_human_repeat_at_1 = (prev_targets == targets.view(T, 1)).sum(1).gt(0).sum().item()

    return total_repeat_at_1, total_wrong_repeat_at_1, total_human_repeat_at_1


@Metric.register("exp_bias")
class ExposureBias(Metric):
    """
    This :class:`Metric` breaks with the typical ``Metric`` API and just stores values that were
    computed in some fashion outside of a ``Metric``.  If you have some external code that computes
    the metric for you, for instance, you can use this to report the average result using our
    ``Metric`` API.
    """

    def __init__(self,
                 oracle: Oracle,
                 context_length: int, 
                 max_generated_len: int = 100,
                 temperature: int = 1.0,
                ) -> None:
        self.max_generated_len = max_generated_len
        self.temp = temperature
        self._oracle = oracle
        self._context_length = context_length

        self.kl = 0.0
        self.cross_ent = 0.0
        self.Hmm = 0.
        self.Hmo = 0.
        self._total_count = 0
        self._total_seq = 0
        self.count_till_len = [0] * max_generated_len
        self.kl_till_len = [0.] * max_generated_len
        self.kl_till_len_norm = [0.] * self.max_generated_len

        self.cross_ent_till_len = [0.] * max_generated_len
        self.Hmm_till_len = [0.] * max_generated_len
        self.Hmo_till_len = [0] * max_generated_len
        self.model_xent_till_len = [0] * max_generated_len
        self.oracle_xent_till_len = [0] * max_generated_len
        self.model_xent_till_len_accum = [0] * max_generated_len
        self.oracle_xent_till_len_accum = [0] * max_generated_len
        self.disagreements_till_len = {
            1: [0] * max_generated_len,
            2: [0] * max_generated_len,
            3: [0] * max_generated_len,
            5: [0] * max_generated_len,
            10: [0] * max_generated_len,
            100: [0] * max_generated_len,
        }

        self.count_at_len = [0] * max_generated_len
        self.kl_at_len = [0] * max_generated_len
        self.cross_ent_at_len = [0] * max_generated_len
        self.Hmm_at_len = [0.] * max_generated_len
        self.Hmo_at_len = [0] * max_generated_len
        self.model_xent_at_len = [0] * max_generated_len
        self.oracle_xent_at_len = [0] * max_generated_len
        self.disagreements_at_len = {
            1: [0] * max_generated_len,
            2: [0] * max_generated_len,
            3: [0] * max_generated_len,
            5: [0] * max_generated_len,
            10: [0] * max_generated_len,
            100: [0] * max_generated_len,
        }
        
        self.seq_lens = [0] * self.max_generated_len

        self.cuml_repeat = {
            16: 0,
            32: 0,
            128: 0,
            512: 0,
        }

        self.cuml_wrong_repeat = {
            16: 0,
            32: 0,
            128: 0,
            512: 0,
        }

        self.cuml_human_repeat = {
            16: 0,
            32: 0,
            128: 0,
            512: 0,
        }
        self.seq_rep_1_cuml = 0.
        self.seq_rep_4_cuml = 0.

        self.oracle_nll_cuml = 0.
        self.uniq_seqs = set([])
        self.human_uniq_seqs = set([])

        self.uniqs = set([])
        self.human_uniqs = set([])

    @overrides
    def __call__(self,
                 predictions: List[str],
                 seq_lengths: List[int],
                 tokens: torch.LongTensor,
                 target_tokens: torch.LongTensor,
                 step_logits: torch.FloatTensor,
                ):
        """
        Parameters
        ----------
        value : ``float``
            The value to average.
        """
        # TODO (Kushal) Add comments to explain what is going on.
        # Compute DL(P||M)
        batch_size = len(predictions)
        kl_total = 0.
        cross_ent_total = 0.
        Hmo_total = 0.
        Hmm_total = 0.
        total_count = 0

        kls = []
        cross_ents = []
        Hmos = []
        Hmms = []

        step_log_probs = F.log_softmax(step_logits, dim=-1)

        model_seq_probs = (-1 * torch.gather(step_log_probs, -1,
                                        tokens[:,1:].unsqueeze(2)) \
                                    .squeeze(2)) \
                                    .tolist()

        oracle_probs = []
        oracle_seq_probs = []

        oracle_probs_and_seq_probs = self._oracle.compute_sent_probs(
                                                    predictions, tokens)
        for i in range(batch_size):
            if len(predictions[i]) == 0:
                continue

            seq_len = min(seq_lengths[i], self.max_generated_len)

            kl_till_lens = []
            kl_at_lens = []
            disagreements_at_lens1 = []
            cross_ent_till_lens = []
            cross_ent_at_lens = []

            Hmo_at_lens = []
            Hmm_at_lens = []


            kl_seq = 0.
            cross_ent_seq = 0.
            
            model_xent_seq = 0.
            oracle_xent_seq = 0.

            Hmo_seq = 0.
            Hmm_seq = 0.
            disagreements = {
                1: 0,
                2: 0,
                3: 0,
                5: 0,
                10: 0,
                100: 0,
            }

            oracle_step_log_probs = oracle_probs_and_seq_probs[i][-1]
            oracle_seq_prob = (-1 * torch.gather(oracle_step_log_probs, -1,
                                            tokens[i,1:].unsqueeze(1)) \
                                        .squeeze(1)) \
                                        .tolist()
    
            oracle_seq_probs.append(oracle_seq_prob)

            # self._context_length = 1
            # for l,j in enumerate(range(self._context_length, seq_len-1)):
            for l,j in enumerate(range(1, seq_len-1)):
                oracle_step_log_prob = oracle_probs_and_seq_probs[i][-1][j]
                step_log_prob = step_log_probs[i, j]

                if self.temp != 1:
                    oracle_step_log_prob = \
                        F.log_softmax(oracle_step_log_prob/self.temp, dim=-1)
                    step_log_prob = \
                         F.log_softmax(step_log_prob/self.temp, dim=-1)

                oracle_step_prob = torch.exp(oracle_step_log_prob)
                step_prob = torch.exp(step_log_prob)
                kl = F.kl_div(step_log_prob, 
                                oracle_step_prob,
                                reduction='sum')
                
                cross_ent =  (-1 * oracle_step_prob * step_log_prob).sum()
                Hoo = (-1 * oracle_step_prob * oracle_step_log_prob).sum()
                Hmm = (-1 * step_prob * step_log_prob).sum()
                Hmo = (-1 * step_prob * oracle_step_log_prob).sum()

                pred_at_j = tokens[i][j+1].item()
                oracle_top_k = torch.topk(oracle_step_prob, k=100)[1].tolist()
                for k in [1, 2, 3, 5, 10, 100]:
                    disagreement = int(pred_at_j not in set(oracle_top_k[:k]))
                    disagreements[k] += disagreement
                    self.disagreements_at_len[k][j] += disagreement
                    self.disagreements_till_len[k][j] += disagreements[k]

                disagreements_at_lens1.append(int(pred_at_j not in set(oracle_top_k[:10])))
                kl = kl.item()
                
                model_xent = model_seq_probs[i][j]
                oracle_xent = oracle_seq_prob[j]

                cross_ent = cross_ent.item()
                Hmo = Hmo.item()
                Hmm = Hmm.item()
                
                kl_at_lens.append(kl)
                cross_ent_at_lens.append(cross_ent)
                Hmo_at_lens.append(Hmo)
                Hmm_at_lens.append(Hmm)


                kl_seq += kl
                model_xent_seq += model_xent
                oracle_xent_seq += oracle_xent

                cross_ent_seq += cross_ent
                Hmo_seq += Hmo
                Hmm_seq += Hmm

                kl_till_lens.append(kl_seq)
                cross_ent_till_lens.append(cross_ent_seq)

                self._total_count += 1

                self.kl_till_len[j] += kl_seq
                self.kl_till_len_norm[j] += kl_seq/(l+1)

                self.model_xent_till_len[j] += model_xent_seq/(l+1)
                self.oracle_xent_till_len[j] += oracle_xent_seq/(l+1)
                self.model_xent_till_len_accum[j] += model_xent_seq
                self.oracle_xent_till_len_accum[j] += oracle_xent_seq

                self.cross_ent_till_len[j] += cross_ent_seq#/j
                self.Hmo_till_len[j] += Hmo_seq/(l+1)
                self.Hmm_till_len[j] += Hmm_seq/(l+1)
                self.count_till_len[j] += 1
        
                self.kl_at_len[j] += kl
                self.cross_ent_at_len[j] += cross_ent
                self.Hmo_at_len[j] += Hmo
                self.Hmm_at_len[j] += Hmm
                self.count_at_len[j] += 1
                self.model_xent_at_len[j] += model_xent
                self.oracle_xent_at_len[j] += oracle_xent

            self.kl += kl_seq#/seq_len
            self.cross_ent += cross_ent_seq#/seq_len
            self.Hmo += Hmo_seq/seq_len
            self.Hmm += Hmm_seq/seq_len
            self._total_seq += 1

            kls.append(kl_seq/seq_len)
            cross_ents.append(cross_ent_seq/seq_len)
            Hmos.append(Hmo_seq/seq_len)
            Hmms.append(Hmm_seq/seq_len)

            self.seq_lens[seq_len - 1] += 1
            oracle_nll = -1 * np.log(oracle_probs_and_seq_probs[i][0])
            self.oracle_nll_cuml += oracle_nll

            for till in REPEAT_CONTEXT_LENGTHS:
                rep, wrep, hrep = repeat_at_1(tokens[i], target_tokens[i], till)
                self.cuml_repeat[till] += rep/len(tokens[i])
                self.cuml_wrong_repeat[till] += wrep/len(tokens[i])
                self.cuml_human_repeat[till] += hrep/len(tokens[i])

            prediction_completions = tokens[i, self._context_length:].tolist()
            prediction_completions_str = " ".join([str(x) for x in  prediction_completions])
            target_completions = target_tokens[i, self._context_length:].tolist()
            target_completions_str = " ".join([str(x) for x in  target_completions])

            self.uniqs=self.uniqs.union(set(prediction_completions))
            self.human_uniqs=self.human_uniqs.union(set(target_completions))

            self.uniq_seqs.add(prediction_completions_str)
            self.human_uniq_seqs.add(target_completions_str)

            ngm_metric = ngram_metrics(prediction_completions)

            self.seq_rep_1_cuml += ngm_metric[1]
            self.seq_rep_4_cuml += ngm_metric[4]
            oracle_probs.append(oracle_nll)

        return {
            "oracle_probs": oracle_probs, 
            "oracle_seq_probs": oracle_seq_probs, 
            "model_seq_probs": model_seq_probs
        }

    @overrides
    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The average of all values that were passed to ``__call__``.
        """
        output =  {
            "kl": self.kl/self._total_count,
            "cross_ent": self.cross_ent/self._total_count,
            "Hmo": self.Hmo/self._total_count,
            "Hmm": self.Hmm/self._total_count,
            "model_xent_till_len": self.model_xent_till_len,
            "oracle_xent_till_len": self.oracle_xent_till_len,
            "model_xent_till_len_accum": self.model_xent_till_len_accum,
            "oracle_xent_till_len_accum": self.oracle_xent_till_len_accum,

            "count_till_len": self.count_till_len,
            "kl_till_len": self.kl_till_len,
            "kl_till_len_norm": self.kl_till_len_norm,

            "cross_ent_till_len": self.cross_ent_till_len,
            "Hmo_till_len": self.Hmo_till_len,
            "Hmm_till_len": self.Hmm_till_len,
            "disagreements_till_len": self.disagreements_till_len,

            "count_at_len": self.count_at_len,
            "kl_at_len": self.kl_at_len,
            "cross_ent_at_len": self.cross_ent_at_len,
            "model_xent_at_len": self.model_xent_at_len,
            "oracle_xent_at_len": self.oracle_xent_at_len,
            "Hmm_at_len": self.Hmm_at_len,
            "Hmo_at_len": self.Hmo_at_len,
            "disagreements_at_len": self.disagreements_at_len,

            "seq_lens": self.seq_lens,

            "oracle_nll": self.oracle_nll_cuml/self._total_seq,
            "seq-rep-1": self.seq_rep_1_cuml/self._total_seq,
            "seq-rep-4": self.seq_rep_4_cuml/self._total_seq,
            "uniq-seq": len(self.uniq_seqs),
            "human-uniq-seq": len(self.human_uniq_seqs),

            "rep/16": self.cuml_repeat[16]/self._total_seq,
            "rep/32": self.cuml_repeat[32]/self._total_seq,
            "rep/128": self.cuml_repeat[128]/self._total_seq,
            "rep/512": self.cuml_repeat[512]/self._total_seq,
            "wrep/16": self.cuml_wrong_repeat[16]/self._total_seq,
            "wrep/32": self.cuml_wrong_repeat[32]/self._total_seq,
            "wrep/128": self.cuml_wrong_repeat[128]/self._total_seq,
            "wrep/512": self.cuml_wrong_repeat[512]/self._total_seq,
            "hrep/16": self.cuml_human_repeat[16]/self._total_seq,
            "hrep/32": self.cuml_human_repeat[32]/self._total_seq,
            "hrep/128": self.cuml_human_repeat[128]/self._total_seq,
            "hrep/512": self.cuml_human_repeat[512]/self._total_seq,
            "uniq": len(self.uniqs),
            "human-uniq": len(self.human_uniqs),
        }

        if reset:
            self.reset()

        return output
    @overrides
    def reset(self):
        self.kl = 0.0
        self.cross_ent = 0.0
        self.Hmm = 0.
        self.Hmo = 0.
        self._total_count = 0
        self._total_seq = 0

        self.count_till_len = [0] * self.max_generated_len
        self.kl_till_len = [0.] * self.max_generated_len
        self.kl_till_len_norm = [0.] * self.max_generated_len

        self.cross_ent_till_len = [0.] * self.max_generated_len
        self.Hmm_till_len = [0.] * self.max_generated_len
        self.Hmo_till_len = [0] * self.max_generated_len
        self.model_xent_till_len = [0] * self.max_generated_len
        self.oracle_xent_till_len = [0] * self.max_generated_len
        self.model_xent_till_len_accum = [0] * self.max_generated_len
        self.oracle_xent_till_len_accum = [0] * self.max_generated_len
        self.disagreements_till_len = {
            1: [0] * self.max_generated_len,
            2: [0] * self.max_generated_len,
            3: [0] * self.max_generated_len,
            5: [0] * self.max_generated_len,
            10: [0] * self.max_generated_len,
            100: [0] * self.max_generated_len,
        }

        self.count_at_len = [0] * self.max_generated_len
        self.kl_at_len = [0] * self.max_generated_len
        self.cross_ent_at_len = [0] * self.max_generated_len
        self.Hmm_at_len = [0.] * self.max_generated_len
        self.Hmo_at_len = [0] * self.max_generated_len
        self.model_xent_at_len = [0] * self.max_generated_len
        self.oracle_xent_at_len = [0] * self.max_generated_len
        self.disagreements_at_len = {
            1: [0] * self.max_generated_len,
            2: [0] * self.max_generated_len,
            3: [0] * self.max_generated_len,
            5: [0] * self.max_generated_len,
            10: [0] * self.max_generated_len,
            100: [0] * self.max_generated_len,
        }

        self.seq_lens = [0] * self.max_generated_len

        self.oracle_nll_cuml = 0.
        self.seq_rep_1_cuml = 0.
        self.seq_rep_4_cuml = 0.
        self.uniq_seqs = set([])
        self.human_uniq_seqs = set([])


        self.uniqs = set([])
        self.human_uniqs = set([])

        self.acc = 0.
        self.rep = 0.
        self.wrep = 0.
        self.uniq = 0.