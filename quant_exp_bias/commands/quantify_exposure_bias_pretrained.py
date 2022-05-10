"""
The ``evaluate`` subcommand can be used to
evaluate a trained model against a dataset
and report any metrics calculated by the model.

.. code-block:: bash

    $ allennlp evaluate --help
    usage: allennlp evaluate [-h] [--output-file OUTPUT_FILE]
                             [--weights-file WEIGHTS_FILE]
                             [--cuda-device CUDA_DEVICE] [-o OVERRIDES]
                             [--batch-weight-key BATCH_WEIGHT_KEY]
                             [--extend-vocab]
                             [--embedding-sources-mapping EMBEDDING_SOURCES_MAPPING]
                             [--include-package INCLUDE_PACKAGE]
                             archive_file input_file

    Evaluate the specified model + dataset

    positional arguments:
      archive_file          path to an archived trained model
      input_file            path to the file containing the evaluation data

    optional arguments:
      -h, --help            show this help message and exit
      --output-file OUTPUT_FILE
                            path to output file
      --weights-file WEIGHTS_FILE
                            a path that overrides which weights file to use
      --cuda-device CUDA_DEVICE
                            id of GPU to use (if any)
      -o OVERRIDES, --overrides OVERRIDES
                            a JSON structure used to override the experiment
                            configuration
      --batch-weight-key BATCH_WEIGHT_KEY
                            If non-empty, name of metric used to weight the loss
                            on a per-batch basis.
      --extend-vocab        if specified, we will use the instances in your new
                            dataset to extend your vocabulary. If pretrained-file
                            was used to initialize embedding layers, you may also
                            need to pass --embedding-sources-mapping.
      --embedding-sources-mapping EMBEDDING_SOURCES_MAPPING
                            a JSON dict defining mapping from embedding module
                            path to embeddingpretrained-file used during training.
                            If not passed, and embedding needs to be extended, we
                            will try to use the original file paths used during
                            training. If they are not available we will use random
                            vectors for embedding extension.
      --include-package INCLUDE_PACKAGE
                            additional packages to include
"""
from typing import Dict, Any, List, Iterable
import argparse
import logging
import json
import os
import pprint
import torch
import numpy as np
import math
import random
import torch.nn.functional as F
import wandb 

from allennlp.commands.subcommand import Subcommand
from allennlp.common.util import prepare_environment
from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.common.tqdm import Tqdm
from allennlp.common import Params

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance

from allennlp.data.data_loaders.simple_data_loader import SimpleDataLoader
from allennlp.data.data_loaders import DataLoader

from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.training.util import evaluate
from allennlp.nn import util as nn_util
from allennlp.nn.beam_search import MultinomialSampler, TopKSampler, TopPSampler


from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from copy import deepcopy
from scipy.stats.stats import pearsonr

from quant_exp_bias.metrics.exposure_bias import ExposureBias
from quant_exp_bias.oracles.natural_language_oracle import NaturalLanguageOracle

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def generate_list(lst, counts, clen):
    return [(i, x/y) for i, (x,y) in enumerate(zip(lst, counts)) if y > 0]

@Subcommand.register("quantify-exposure-bias-pretrained")
class QuantifyExposureBiasPretrained(Subcommand):
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Evaluate the specified model + dataset'''
        subparser = parser.add_parser(
                self.name, description=description, help='Evaluate the specified model + dataset.')

        subparser.add_argument('--eval-model', 
                                type=str,
                                default='gpt2',
                                help='Model to be evaluated')

        subparser.add_argument('--oracle-model', 
                                type=str,
                                default='gpt2-xl',
                                help='config file for oracle.')

        subparser.add_argument('--output-dir', 
                               required=True,
                               type=str, 
                               help='path to output directory')

        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument('--cuda-device',
                                 type=int,
                                 default=-1,
                                 help='id of GPU to use (if any)')

        subparser.add_argument('--num-samples',
                                 type=int,
                                 default=15,
                                 help='Number of samples to draw from $n~\mathcal{N}$" + \
                                        "for approximating expectation over sequence lengths.')

        subparser.add_argument('--context-file-or-filename',
                               type=str,
                               help='Path to the context file.',
                               default='wikitext-103')

        subparser.add_argument('--context-len',
                                 type=int,
                                 default=50,
                                 help='Number of tokens to use for context.')

        subparser.add_argument('--sample-outputs',
                                 action="store_true",
                                 help='Sample output instead of greedy decoding.')

        subparser.add_argument('--sampling-temperature',
                                 type=float,
                                 default=None,
                                 help='Sampling Temperature.')

        subparser.add_argument('--exp-temperature',
                                 type=float,
                                 default=1.0,
                                 help='Exposure Bias Temperature.')

        subparser.add_argument('--top-k',
                                 type=int,
                                 default=None,
                                 help='Top k sampling coeff.')

        subparser.add_argument('--top-p',
                                 type=float,
                                 default=None,
                                 help='Top p sampling coeff.')

        subparser.add_argument('--beam',
                                 type=int,
                                 default=None,
                                 help='Beam size to us.')

        subparser.set_defaults(func=quantify_exposure_bias_pretrained_from_args)

        return subparser

def quantify_exposure_bias_pretrained_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return quantify_exposure_bias_pretrained(
                                 output_dir=args.output_dir,
                                 eval_model=args.eval_model,
                                 oracle_model=args.oracle_model,
                                 context_file_or_filename=args.context_file_or_filename,
                                 context_len=args.context_len,
                                 num_samples=args.num_samples,
                                 cuda_device=args.cuda_device,
                                 sample_outputs=args.sample_outputs,
                                 sampling_temperature=args.sampling_temperature,
                                 exp_temperature=args.exp_temperature,
                                 top_k=args.top_k,
                                 top_p=args.top_p, 
                                 beam=args.beam)

def quantify_exposure_bias_pretrained(output_dir: str,
                           eval_model:str,
                           oracle_model:str,
                           context_file_or_filename: str,
                           split='test',
                           context_len: int = 25,
                           num_samples: int = 100,
                           cuda_device: int = -1,
                           top_k: int = None,
                           top_p: float = None,
                           repeat_penalty: float = None,
                           beam: int = None,
                           sample_outputs: bool = True,
                           sampling_temperature: float = None,
                           exp_temperature: float = 1.0,
                           generation_size: int = 128,
                           batch_size: int = 16,
                          ):

    print(f'Num Samples: {num_samples}')
    print(f'Sample Outputs: {sample_outputs}')
    print(f'Sampling Temperature: {sampling_temperature}')
    print(f'Context Len: {context_len}')
    print(f'Exposure Bias Temperature: {exp_temperature}')
    print(f"Top-p Sampling Coeff: {top_p}")
    print(f"Top-k Sampling Coeff: {top_k}")
    print(f"Repeat Penalty: {repeat_penalty}")
    print(f"Beam Size: {beam}")
    print(f"Generation Size: {generation_size}")
    print(f"Output Dir: {output_dir}")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load from archive
    model = AutoModelForCausalLM.from_pretrained(eval_model)

    if cuda_device > -1:
        model.cuda()
    model.eval()

    oracle = NaturalLanguageOracle(oracle_model, 
                        cuda_device=cuda_device)

    exposure_bias_target = ExposureBiasV2(oracle, 
                        max_generated_len=generation_size,
                        temperature=exp_temperature, 
                        context_length=context_len)

    exposure_bias = ExposureBiasV2(oracle, 
                        max_generated_len=generation_size,
                        temperature=exp_temperature, 
                        context_length=context_len)


    if context_file_or_filename == 'wikitext-103':
        dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')
    elif context_file_or_filename == 'wikitext-2':
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

    tokenizer = AutoTokenizer.from_pretrained(eval_model)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    encoded_dataset = dataset.map(lambda x: tokenizer(x['text']), 
                                    batched=True, 
                                    num_proc=10, 
                                    remove_columns=dataset[split].column_names)

    block_size = min(generation_size, tokenizer.model_max_length)
    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = encoded_dataset.map(
        group_texts,
        batched=True,
        num_proc=10,
        desc=f"Grouping texts in chunks of {block_size}",
    )
    lm_datasets.set_format(type='torch')
    
    dataloader = torch.utils.data.DataLoader(
                                lm_datasets[split], 
                                batch_size=8, 
                                shuffle=False)

    print("Metrics:")
    H_o_m_cuml = 0.
    H_o_m_count = 0

    for sample_num, input_dict in enumerate(dataloader):
        for key, val in input_dict.items():
            if cuda_device > -1:
                input_dict[key] = val.cuda()

        if sample_num >= num_samples:
            break

        with torch.no_grad():
            output_dict = model(**input_dict)
        
        H_o_m_cuml += float(output_dict['loss'])
        H_o_m_count += 1

        detokenized_targets: List[str] = tokenizer.batch_decode(input_dict['input_ids'],
                                                            skip_special_tokens=True)

        target_tokens = input_dict['input_ids']
        target_seq_lengths: List[List[str]] = (target_tokens != pad_token_id).sum(-1).tolist()

        rollin_step_logits = output_dict['logits']

        target_outputs = exposure_bias_target(
                            predictions=detokenized_targets,
                            seq_lengths=target_seq_lengths,
                            tokens=target_tokens,
                            target_tokens=target_tokens,
                            step_logits=rollin_step_logits,)

        target_metrics = exposure_bias_target.get_metric(reset=False)

        context_tokens = input_dict['input_ids'][:, :context_len]
        predictions = model.generate(
                        input_ids=context_tokens, 
                        max_length=generation_size,
                        do_sample=sample_outputs,
                        top_k=top_k, 
                        top_p=top_p,
                        num_beams=beam,
                        repetition_penalty=repeat_penalty,
                        temperature=sampling_temperature)

        detokenized_predictions: List[str] = tokenizer.batch_decode(predictions,
                                                            skip_special_tokens=True)

        predicted_seq_lengths = (predictions != pad_token_id).sum(-1).tolist()

        output_dict = model(input_ids=predictions)
        step_logits = output_dict['logits']

        prediction_outputs = exposure_bias(
                                predictions=detokenized_predictions,
                                seq_lengths=predicted_seq_lengths,
                                tokens=predictions,
                                target_tokens=target_tokens,
                                step_logits=step_logits)

        metrics = exposure_bias.get_metric(reset=False)

        if output_dir:
            detokenized_contexts = tokenizer.batch_decode(context_tokens,
                                                    skip_special_tokens=True)
            with open(os.path.join(output_dir, 'model_sampled_generated.txt'), "a+") as file:
                for context, prediction in zip(detokenized_contexts, 
                                                detokenized_predictions):
                    context = context.replace("\n", " ")
                    predictions = prediction.replace("\n", " ")
                    print(f'Context::    {context}\n'+
                          f'Prediction:: {predictions}', 
                          file=file)

            print("Trial: %3d :: %s: (%-5.4f/%-5.4f=%-5.4f)" 
                        % (sample_num, "Exp Bias.(KL)", metrics['kl'], target_metrics['kl'], metrics['kl']/target_metrics['kl']))

    kl_till_len = generate_list(metrics['kl_till_len'], 
                                metrics['count_till_len'], context_len)
    target_kl_till_len = generate_list(target_metrics['kl_till_len'], 
                                        target_metrics['count_till_len'], context_len)
    kl_till_len_norm = generate_list(metrics['kl_till_len_norm'], 
                                metrics['count_till_len'], 
                                context_len)
    target_kl_till_len_norm = generate_list(target_metrics['kl_till_len_norm'], 
                                        target_metrics['count_till_len'], context_len)

    ratio_kl_till_len = []
    kl_till_len_dict = dict(kl_till_len)
    target_kl_till_len_dict = dict(target_kl_till_len)
    till_len_idxs = set(kl_till_len_dict.keys()).intersection(target_kl_till_len_dict.keys()) 
    till_len_idxs_sorted = sorted(till_len_idxs)
    for idx in till_len_idxs_sorted:
        if target_kl_till_len_dict[idx] == 0:
            continue

        ratio = kl_till_len_dict[idx]/target_kl_till_len_dict[idx]
        ratio_kl_till_len.append((idx, ratio))
    
    acc_err_till_len = []
    for idx, ratio in ratio_kl_till_len:
        acc_err_till_len.append((idx, ratio * idx))

    excess_acc_err_till_len = []
    for idx, ratio in ratio_kl_till_len:
        excess_acc_err_till_len.append((idx, (ratio - 1)*100))

    cross_ent_till_len = generate_list(metrics['cross_ent_till_len'], 
                                        metrics['count_till_len'], context_len)
    target_cross_ent_till_len = generate_list(target_metrics['cross_ent_till_len'], 
                                                target_metrics['count_till_len'], context_len)

    ratio_cross_ent_till_len = []
    cross_ent_till_len_dict = dict(cross_ent_till_len)
    target_cross_ent_till_len_dict = dict(target_cross_ent_till_len)
    till_len_idxs = set(cross_ent_till_len_dict.keys()).intersection(target_cross_ent_till_len_dict.keys()) 
    till_len_idxs_sorted = sorted(till_len_idxs)
    for idx in till_len_idxs_sorted:
        if target_cross_ent_till_len_dict[idx] == 0:
            continue

        ratio = cross_ent_till_len_dict[idx]/target_cross_ent_till_len_dict[idx]
        ratio_cross_ent_till_len.append((idx, ratio))

    model_xent_till_len = generate_list(metrics['model_xent_till_len'], 
                                    metrics['count_till_len'], context_len)
    oracle_xent_till_len = generate_list(metrics['oracle_xent_till_len'], 
                                    metrics['count_till_len'], context_len)
    target_model_xent_till_len = generate_list(target_metrics['model_xent_till_len'], 
                                    target_metrics['count_till_len'], context_len)
    target_oracle_xent_till_len = generate_list(target_metrics['oracle_xent_till_len'], 
                                    target_metrics['count_till_len'], context_len)

    model_xent_till_len_accum = generate_list(metrics['model_xent_till_len_accum'], 
                                    metrics['count_till_len'], context_len)
    oracle_xent_till_len_accum = generate_list(metrics['oracle_xent_till_len_accum'], 
                                    metrics['count_till_len'], context_len)
    target_model_xent_till_len_accum = generate_list(target_metrics['model_xent_till_len_accum'], 
                                    target_metrics['count_till_len'], context_len)
    target_oracle_xent_till_len_accum = generate_list(target_metrics['oracle_xent_till_len_accum'], 
                                    target_metrics['count_till_len'], context_len)

    Hmo_till_len = generate_list(metrics['Hmo_till_len'], 
                                    metrics['count_till_len'], context_len)
    target_Hmo_till_len = generate_list(target_metrics['Hmo_till_len'], 
                                        target_metrics['count_till_len'], context_len)

    Hmm_till_len = generate_list(metrics['Hmm_till_len'], 
                                    metrics['count_till_len'], context_len)
    target_Hmm_till_len = generate_list(target_metrics['Hmm_till_len'], 
                                        target_metrics['count_till_len'], context_len)

    disagreements_till_len1 = generate_list(metrics['disagreements_till_len'][1], 
                                            metrics['count_till_len'], context_len)
    disagreements_till_len2 = generate_list(metrics['disagreements_till_len'][2], 
                                            metrics['count_till_len'], context_len)
    disagreements_till_len5 = generate_list(metrics['disagreements_till_len'][5], 
                                            metrics['count_till_len'], context_len)
    disagreements_till_len10 = generate_list(metrics['disagreements_till_len'][10], 
                                            metrics['count_till_len'], context_len)

    target_disagreements_till_len1 = generate_list(target_metrics['disagreements_till_len'][1], 
                                                    target_metrics['count_till_len'], context_len)
    target_disagreements_till_len2 = generate_list(target_metrics['disagreements_till_len'][2], 
                                                    target_metrics['count_till_len'], context_len)
    target_disagreements_till_len5 = generate_list(target_metrics['disagreements_till_len'][5], 
                                                    target_metrics['count_till_len'], context_len)
    target_disagreements_till_len10 = generate_list(target_metrics['disagreements_till_len'][10], 
                                                    target_metrics['count_till_len'], context_len)

    kl_at_len = generate_list(metrics['kl_at_len'], 
                                metrics['count_at_len'], context_len)
    target_kl_at_len = generate_list(target_metrics['kl_at_len'], 
                                        target_metrics['count_at_len'], context_len)
    model_xent_at_len = generate_list(metrics['model_xent_at_len'], 
                                    metrics['count_at_len'], context_len)
    oracle_xent_at_len = generate_list(metrics['oracle_xent_at_len'], 
                                    metrics['count_at_len'], context_len)
    target_model_xent_at_len = generate_list(target_metrics['model_xent_at_len'], 
                                    target_metrics['count_at_len'], context_len)
    target_oracle_xent_at_len = generate_list(target_metrics['oracle_xent_at_len'], 
                                    target_metrics['count_at_len'], context_len)

    ratio_kl_at_len = []
    kl_at_len_dict = dict(kl_at_len)
    target_kl_at_len_dict = dict(target_kl_at_len)
    till_len_idxs = set(kl_at_len_dict.keys()).intersection(target_kl_at_len_dict.keys()) 
    till_len_idxs_sorted = sorted(till_len_idxs)
    for idx in till_len_idxs_sorted:
        if target_kl_at_len_dict[idx] == 0:
            continue

        ratio = kl_at_len_dict[idx]/target_kl_at_len_dict[idx]
        ratio_kl_at_len.append((idx, ratio))
    

    cross_ent_at_len = generate_list(metrics['cross_ent_at_len'], 
                                        metrics['count_at_len'], context_len)
    target_cross_ent_at_len = generate_list(target_metrics['cross_ent_at_len'], 
                                            target_metrics['count_at_len'], context_len)

    ratio_cross_ent_at_len = []
    cross_ent_at_len_dict = dict(cross_ent_at_len)
    target_cross_ent_at_len_dict = dict(target_cross_ent_at_len)
    till_len_idxs = set(cross_ent_at_len_dict.keys()).intersection(target_cross_ent_at_len_dict.keys()) 
    till_len_idxs_sorted = sorted(till_len_idxs)
    for idx in till_len_idxs_sorted:
        if target_cross_ent_at_len_dict[idx] == 0:
            continue

        ratio = cross_ent_at_len_dict[idx]/target_cross_ent_at_len_dict[idx]
        ratio_cross_ent_at_len.append((idx, ratio))

    Hmm_at_len = generate_list(metrics['Hmm_at_len'], 
                                    metrics['count_at_len'], context_len)
    target_Hmm_at_len = generate_list(target_metrics['Hmm_at_len'], 
                                        target_metrics['count_at_len'], context_len)

    Hmo_at_len = generate_list(metrics['Hmo_at_len'], 
                                    metrics['count_at_len'], context_len)
    target_Hmo_at_len = generate_list(target_metrics['Hmo_at_len'], 
                                        target_metrics['count_at_len'], context_len)

    disagreements_at_len1 = generate_list(metrics['disagreements_at_len'][1], 
                                            metrics['count_at_len'], context_len)
    disagreements_at_len2 = generate_list(metrics['disagreements_at_len'][2], 
                                            metrics['count_at_len'], context_len)
    disagreements_at_len5 = generate_list(metrics['disagreements_at_len'][5], 
                                            metrics['count_at_len'], context_len)
    disagreements_at_len10 = generate_list(metrics['disagreements_at_len'][10], 
                                            metrics['count_at_len'], context_len)

    target_disagreements_at_len1 = generate_list(target_metrics['disagreements_at_len'][1], 
                                                    target_metrics['count_at_len'], context_len)
    target_disagreements_at_len2 = generate_list(target_metrics['disagreements_at_len'][2], 
                                                    target_metrics['count_at_len'], context_len)
    target_disagreements_at_len5 = generate_list(target_metrics['disagreements_at_len'][5], 
                                                    target_metrics['count_at_len'], context_len)
    target_disagreements_at_len10 = generate_list(target_metrics['disagreements_at_len'][10], 
                                                    target_metrics['count_at_len'], context_len)
 
    model_xent_ratio = [(a, y/x) for (a,x),(_, y) in zip(target_model_xent_till_len, kl_till_len)]
    target_xent_ratio_norm = [(a, a * y/x) for (a,x),(_, y) in zip(target_model_xent_till_len, target_kl_till_len_norm)]

    corr_target_err_xent = pearsonr([x[1] for x in target_model_xent_till_len[context_len:]], [x[1] for x in target_kl_till_len_norm[context_len:]])[0]
    corr_model_err_xent = pearsonr([x[1] for x in target_model_xent_till_len[context_len:]], [x[1] for x in kl_till_len_norm[context_len:]])[0]
    

    metrics = {
        "kl": metrics['kl'],
        "target_kl": target_metrics['kl'],
        "Hmm": metrics['Hmm'],
        "Hmo": metrics['Hmo'],
        "corr_target_err_xent": corr_target_err_xent,
        "corr_model_err_xent": corr_model_err_xent,
        "excess_err": excess_acc_err_till_len[-1][1],
        
        "oracle_nll": metrics["oracle_nll"],
        "human_oracle_nll": target_metrics['oracle_nll'],
        "seq-rep-1": metrics["seq-rep-1"],
        "seq-rep-4": metrics["seq-rep-4"],
        "human-seq-rep-1": target_metrics["seq-rep-1"],
        "human-seq-rep-4": target_metrics["seq-rep-4"],
        "uniq-seq": metrics["uniq-seq"],
        "human-uniq-seq": metrics["human-uniq-seq"],

        "rep/16": metrics["rep/16"],
        "rep/32": metrics["rep/32"],
        "rep/128": metrics["rep/128"],
        "rep/512": metrics["rep/512"],
        "wrep/16": metrics["wrep/16"],
        "wrep/32": metrics["wrep/32"],
        "wrep/128": metrics["wrep/128"],
        "wrep/512": metrics["wrep/512"],
        "hrep/16": metrics["hrep/16"],
        "hrep/32": metrics["hrep/32"],
        "hrep/128": metrics["hrep/128"],
        "hrep/512": metrics["hrep/512"],
        "uniq": metrics["uniq"],
        "human-uniq": metrics["human-uniq"],


        "target_Hmm": target_metrics['Hmm'],
        "target_Hmo": target_metrics['Hmo'],
        "kl_till_len": kl_till_len,
        "target_kl_till_len": target_kl_till_len,
        "kl_till_len_norm": kl_till_len_norm,
        "target_kl_till_len_norm": target_kl_till_len_norm,
        "model_xent_ratio": model_xent_ratio,
        "target_xent_ratio_norm": target_xent_ratio_norm,

        "model_xent_till_len": model_xent_till_len,
        "target_model_xent_till_len": target_model_xent_till_len,
        "oracle_xent_till_len": oracle_xent_till_len,
        "target_oracle_xent_till_len": target_oracle_xent_till_len,
    
        "model_xent_till_len_accum": model_xent_till_len_accum,
        "target_model_xent_till_len_accum": target_model_xent_till_len_accum,
        "oracle_xent_till_len_accum": oracle_xent_till_len_accum,
        "target_oracle_xent_till_len_accum": target_oracle_xent_till_len_accum,

        "ratio_kl_till_len": ratio_kl_till_len,
        "acc_err_till_len": acc_err_till_len,
        "excess_acc_err_till_len": excess_acc_err_till_len,

        "cross_ent_till_len": cross_ent_till_len,
        "target_cross_ent_till_len": target_cross_ent_till_len,
        "ratio_cross_ent_till_len": ratio_cross_ent_till_len,
        "Hmo_till_len": Hmo_till_len,
        "Hmm_till_len": Hmm_till_len,
        "target_Hmo_till_len": target_Hmo_till_len,
        "target_Hmm_till_len": target_Hmm_till_len,
        "disagreements_till_len": {
            1: disagreements_till_len1,
            2: disagreements_till_len2,
            5: disagreements_till_len5,
            10: disagreements_till_len10,
        },
        "target_disagreements_till_len": {
            1: target_disagreements_till_len1,
            2: target_disagreements_till_len2,
            5: target_disagreements_till_len5,
            10: target_disagreements_till_len10,
        },
        "kl_at_len": kl_at_len,
        "target_kl_at_len": target_kl_at_len,
        "ratio_kl_at_len": ratio_kl_at_len,
        "model_xent_at_len": model_xent_at_len,
        "target_model_xent_at_len": target_model_xent_at_len,
        "oracle_xent_at_len": oracle_xent_at_len,
        "target_oracle_xent_at_len": target_oracle_xent_at_len,
        "cross_ent_at_len": cross_ent_at_len,
        "target_cross_ent_at_len": target_cross_ent_at_len,
        "ratio_cross_ent_at_len": ratio_cross_ent_at_len,
        "Hmm_at_len": Hmm_at_len,
        "Hmo_at_len": Hmo_at_len,
        "target_Hmm_at_len": target_Hmm_at_len,
        "target_Hmo_at_len": target_Hmo_at_len,
        "disagreements_at_len": {
            1: disagreements_at_len1,
            2: disagreements_at_len2,
            5: disagreements_at_len5,
            10: disagreements_at_len10,
        },
        "target_disagreements_at_len": {
            1: target_disagreements_at_len1,
            2: target_disagreements_at_len2,
            5: target_disagreements_at_len5,
            10: target_disagreements_at_len10,
        }
    }

    with open(os.path.join(output_dir, 'metrics.json'), "w") as file:
        json.dump(metrics, file, indent=4)

    print("Exposure Bias mean: %5.3f", metrics['kl']/target_metrics['kl'])

    print("H(M,O) mean: %5.3f", metrics['Hmo'])

    print("H(M,M) mean: %5.3f", metrics['Hmm'])

    print("Done!!")

    return metrics