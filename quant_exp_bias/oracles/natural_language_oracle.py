from typing import Dict, Tuple
from allennlp.data.vocabulary import Vocabulary

import logging
import math
import torch
import torch.nn.functional as F
import time 

from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, RepetitionPenaltyLogitsProcessor

from lmpl.oracles.oracle_base import Oracle
from multiprocessing import Pool

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Oracle.register('gpt2_oracle')
class NaturalLanguageOracle(Oracle):
    def __init__(self, 
                 model_name="gpt2",
                 parallelize=True,
                 num_threads=128,
                 cuda_device=-1,
                 batch_size=None,
                 start_token='@@@@',
                 end_token='####',
                ):
        super(Oracle, self).__init__()
        # self._parallelize = parallelize
        
        self._num_threads = num_threads
        # self._pool = Pool(self._num_threads)

        self.device = "cpu"
        if cuda_device > -1:
            self.device = f"cuda:{cuda_device}"
        elif cuda_device == -2:
            self.device = torch.cuda.current_device()
        
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load pre-trained model (weights)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        
        self.batch_size = batch_size
        self.model.eval()
        self._start_token = start_token
        self._start_token_id = self.tokenizer.convert_tokens_to_ids(self._start_token)

        self._end_token = end_token
        self._end_token_id = self.tokenizer.convert_tokens_to_ids(self._end_token)
        self._pad_token_id = self._end_token_id

        self._vocab_idxs = None

        # self._processor = RepetitionPenaltyLogitsProcessor(1.0)

    def sample_training_set(self, num_samples: int):
        """
        TODO: sample subset of sentences from the data used for training GPT-2
        """
        pass

    def compute_sent_probs(self, sequences: List[str], tokens: torch.LongTensor=None):
        # TODO (Kushal): Try to figure out how to do this efficiently
        # by batching the inputs.
        seq_batch_size = len(sequences)
        output = []
        batch_size = self.batch_size or seq_batch_size

        if False and tokens is not None:
            start_token_idxs = tokens == self._start_token_id
            end_token_idxs = tokens == self._end_token_id

            tokens[start_token_idxs] = self.tokenizer.bos_token_id
            tokens[end_token_idxs] = self.tokenizer.eos_token_id
        else:
            sequences = [f"{self.tokenizer.bos_token}{x.strip()}{self.tokenizer.eos_token}" \
                                                                        for x in sequences]
        
        for i in range(0, seq_batch_size, batch_size):
            if tokens is None:
                batch = sequences[i:i + batch_size] \
                            if i + batch_size < seq_batch_size \
                                else sequences[i:seq_batch_size]

                encoded_batch = self.tokenizer(batch, return_tensors='pt', padding=True)

                tensor_input = encoded_batch['input_ids'].to(self.device)
                attention_mask = encoded_batch['attention_mask'].to(self.device)
            else:
                tensor_input = tokens[i:i + batch_size] \
                                if i + batch_size < seq_batch_size \
                                    else tokens[i:seq_batch_size]
                attention_mask = tensor_input != self.tokenizer.pad_token_id

            bsize = tensor_input.shape[0]

            with torch.no_grad():
                # outputs = self.model.generate(input_ids=tensor_input[:, :5], do_sample=True)

                results =  self.model(input_ids=tensor_input, 
                                        labels=tensor_input, 
                                        attention_mask=attention_mask)
                labels = tensor_input[:, 1:]
                logits = results[1]
                
                # for k in range(logits.size(1)):
                #     logits[:, k, :] = self._processor(tensor_input, logits[:, k, :])
                
                step_log_probs = F.log_softmax(logits[:, :-1], dim=-1)
                loss_batch_seq = (torch.gather(step_log_probs, -1, 
                                                labels.unsqueeze(2)) \
                                        .squeeze(2)) * \
                                    attention_mask[:, 1:]

                seq_sizes = attention_mask[:, 1:].sum(dim=-1)
                loss_batch = loss_batch_seq.sum(dim=-1)/(seq_sizes)
                
                probs = torch.exp(loss_batch)
                seq_probs = torch.exp(loss_batch_seq)

                for j in range(bsize):
                    output.append((probs[j].item(), seq_probs[j].tolist(), seq_sizes[j].item(), step_log_probs[j]))
        return output

    # TODO (Figure out how to support mixed rollout with this.)
    def reference_rollout(self, 
                            prefixes: torch.LongTensor, 
                            rollout_steps: int,
                            token_to_idx: Dict[str, int],
                            idx_to_token: Dict[int, str]
                            ):
        batch_size, _ = prefixes.shape
        model_vocab_size = len(token_to_idx.keys())

        if self._vocab_mask is None:
            vocab_idxs = self.tokenizer.convert_tokens_to_ids(token_to_idx.keys())
            self._vocab_mask = torch.zeros(self.tokenizer.vocab_size)
            self._vocab_mask.scatter_(0, torch.tensor(vocab_idxs), 1)
            self._vocab_mask.to(self.device)
        
        prefix_tokens = []
        for seq in prefixes.tolist():
            prefix_tokens.append([])
            for idx in seq:
                prefix_tokens[-1].append(idx_to_token[idx])
                                               
        prefixes = torch.LongTensor([self.tokenizer.convert_tokens_to_ids(seq) 
                                        for seq in prefix_tokens], device=self.device)
        past = None
        for step in range(rollout_steps):
            logits, past =  self.model(prefixes[:, 1:], past=past)
             
            mask = (self._vocab_mask.expand(logits.shape) + 1e-45).log()
            logits = logits + mask

            _, predictions = torch.topk(logits[:, -1, :], k=5)

            # If EOS appears in top-5 and we have generated atleast 50% of rollout steps, 
            # we assume we can meaningfully end the sentence and we do.
            next_tokens = torch.where((int(step > 0.5 * rollout_steps) * 
                                        ((predictions == self.tokenizer.eos_token_id).sum(-1) > 0) \
                                            ) \
                                      .bool(), 
                                torch.zeros_like(predictions[:, 0]).fill_(self.tokenizer.eos_token_id), 
                                predictions[:, 0]) \
                                    .unsqueeze(1)

            prefixes = torch.cat([prefixes, next_tokens], dim=1)
        
        prediction_tokens = [self.tokenizer.convert_ids_to_tokens(ids) \
                                for ids in prefixes.tolist()]

        prediction_idxs = []
        for seq in prediction_tokens:
            prediction_idxs.append([])
            for token in seq:
                if token in set([self.tokenizer.convert_ids_to_tokens(198), 
                                    self.tokenizer.eos_token]):
                    prediction_idxs[-1].append(self._end_token)
                    break
                
        prediction_idxs[-1].append(token_to_idx[token])

        return prefixes.new(prediction_idxs)

    def reference_step_rollout(self, 
                                step: int,
                                last_predictions: torch.LongTensor, 
                                state: Dict[str, torch.Tensor],
                                token_to_idx: Dict[str, int],
                                idx_to_token: Dict[int, str],
                             ) -> Tuple[torch.LongTensor, Dict[str, torch.Tensor]]:
        batch_size = last_predictions.shape[0]
        model_vocab_size = len(token_to_idx.keys())

        if self._vocab_idxs is None:
            self._vocab_idxs = torch.tensor(
                    self.tokenizer.convert_tokens_to_ids(token_to_idx.keys())).to(self.device)

        past = state['rollout_params'].get('past', None)
        rollout_prefixes = state['rollout_params'].get('rollout_prefixes', None)
        batch_size, prefix_len = rollout_prefixes.shape

        if not state['rollout_params'].get('rollout_prefixes_in_oracle_vocab', False):
            rollout_prefixes = torch.gather(self._vocab_idxs.unsqueeze(0).expand(batch_size, -1), 
                                                -1, rollout_prefixes)

            state['rollout_params']['rollout_prefixes_in_oracle_vocab'] = True
        
        last_prediction_oracle = torch.gather(self._vocab_idxs.unsqueeze(0).expand(batch_size, -1), 
                                                -1, last_predictions.unsqueeze(1))
        
        oracle_prefixes = torch.cat([rollout_prefixes[:, 1:], last_prediction_oracle], dim=1) \
                            if past is None else last_prediction_oracle

        # state['rollout_params']['rollout_prefixes'] = oracle_prefixes

        logits, past =  self.model(oracle_prefixes, past=past)
        state['rollout_params']['past'] = past
        target_logits = torch.index_select(logits[:, -1], -1, self._vocab_idxs)
        return target_logits, state