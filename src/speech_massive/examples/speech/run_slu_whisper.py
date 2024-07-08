#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence speech recognition.
"""
# You can also adapt this script on your own sequence to sequence speech
# recognition task. Pointers for this are left as comments.

import logging
import os
import sys
import warnings
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import datasets
import evaluate
import torch
from datasets import DatasetDict, load_dataset, concatenate_datasets, interleave_datasets

import transformers
from transformers import (
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
    HfArgumentParser,
    # Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

import json
import copy
import random
import numpy as np

from src.speech_massive.examples.scripts.massive_eval import MassiveEval
from src.speech_massive.examples.scripts import s3prl_slot_eval
from src.speech_massive.examples.speech.trainer_seq2seq_whisper_slu import CustomSeq2SeqTrainer

from pathlib import Path

os.environ["WANDB_DISABLED"] = "true"
_V1_LANGS = ['ar-SA', 'de-DE', 'es-ES', 'fr-FR', 'hu-HU', 'ko-KR', 'nl-NL', 'pl-PL', 'pt-PT', 'ru-RU', 'tr-TR', 'vi-VN']

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.37.0.dev0")

require_version("datasets>=1.18.0", "To fix: pip install -r examples/pytorch/speech-recognition/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "feature extractor name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    freeze_feature_encoder: bool = field(
        default=False, metadata={"help": "Whether to freeze the feature encoder layers of the model."}
    )
    freeze_encoder: bool = field(
        default=False, metadata={"help": "Whether to freeze the entire encoder of the seq2seq model."}
    )
    forced_decoder_ids: List[List[int]] = field(
        default=None,
        metadata={
            "help": (
                "A list of pairs of integers which indicates a mapping from generation indices to token indices "
                "that will be forced before sampling. For example, [[0, 123]] means the first generated token "
                "will always be a token of index 123."
            )
        },
    )
    suppress_tokens: List[int] = field(
        default=None, metadata={"help": "A list of tokens that will be suppressed at generation."}
    )
    apply_spec_augment: bool = field(
        default=False,
        metadata={
            "help": "Whether to apply *SpecAugment* data augmentation to the input features. This is currently only relevant for Wav2Vec2, HuBERT, WavLM and Whisper models."
        },
    )

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        # filter out fields that are defined as field(init=False)
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}

        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task: str = field(
        metadata={"help": "Task, either `transcribe` for speech recognition or `translate` for speech translation."},
    )
    dataset_name: str = field(
        default=None, metadata={"help": "The name of the dataset to use for train(via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    is_few_shot: bool = field(
        default=False, metadata={"help": "Whether to train few-shot or not. Default is False."}
    )
    eval_dataset_name: str = field(
        default=None, metadata={"help": "The name of the dataset to use for evaluation (via the datasets library)."}
    )
    eval_dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the EVAL dataset to use (via the datasets library)."}
    )
    test_dataset_name: str = field(
        default=None, metadata={"help": "The name of the dataset to use for evaluation (via the datasets library)."}
    )
    test_dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the TEST dataset to use (via the datasets library)."}
    )
    slurp_dataset_path: str = field(
        default=None, metadata={"help": "Give slurp dataset path to use slurp dataset."} 
    )
    slurp_dataset_config_name: Optional[str] = field(
        default="slurp_real", metadata={"help": "The configuration name of the slurp dataset."}
    )
    train_slurp: bool = field(
        default=False, metadata={"help": "Whether or not to train on SLURP train set"}
    )
    eval_slurp: bool = field(
        default=False, metadata={"help": "Whether or not to eval on SLURP dev set"}
    )
    italic_dataset_path: str = field(
        default=None, metadata={"help": "Give italic dataset path to use italic dataset."} 
    )
    italic_dataset_config_name: Optional[str] = field(
        default="italic-train", metadata={"help": "The configuration name of the italic dataset."}
    )
    train_italic: bool = field(
        default=False, metadata={"help": "Whether or not to train on italic train set"}
    )
    eval_italic: bool = field(
        default=False, metadata={"help": "Whether or not to eval on italic dev set"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    text_column_name: str = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'"},
    )
    max_duration_in_seconds: float = field(
        default=30.0,
        metadata={
            "help": (
                "Truncate audio files that are longer than `max_duration_in_seconds` seconds to"
                " 'max_duration_in_seconds`"
            )
        },
    )
    min_duration_in_seconds: float = field(
        default=0.0, metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"}
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only do data preprocessing and skip training. This is especially useful when data"
                " preprocessing errors out in distributed training due to timeout. In this case, one should run the"
                " preprocessing in a non-distributed setup with `preprocessing_only=True` so that the cached datasets"
                " can consequently be loaded in distributed training"
            )
        },
    )
    train_split_name: str = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: str = field(
        default="validation",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'validation'"
        },
    )
    test_split_name: str = field(
        default="test",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'test'"
        },
    )
    do_lower_case: bool = field(
        default=True,
        metadata={"help": "Whether the target text should be lower cased."},
    )
    replace_task_token: bool = field(
        default=None,
        metadata={"help": "Whether to replace <transcribe> or <translate> tag with <startoflm> tag."},
        # replacing with different tag should be implemented and added later.
    )
    add_slu_tag: bool = field(
        default=False,
        metadata={"help": "Whether to use <startoflm> right after <transcribe> or <translate> tag to indicate the SLU task"},
    ) # this argument is moved to training args intentionally to keep used in trainer_seq2seq2_whisper_slu
    add_separator: str = field(
        default=None,
        metadata={"help": "If no separator is given, it will be without the separator. Make sure add tokenizer split separator(Ġ)"},
    )
    tokens_to_remove_from_suppress: Optional[List[str]] = field(
        default=None, metadata={"help": "If true, remove separator from the suppressed token if it exists"}
    )
    add_space_case: bool = field(
        default=False,
        metadata={"help": "Whether to use <startoflm> right after <transcribe> or <translate> tag to indicate the SLU task"},
    )
    target_format_content: str = field(
        default="transcript_slots_intent",
        metadata={"help": "Options: transcript_slots_intent, slots_intent"},
    )
    target_format_structure: str = field(
        default=None,
        metadata={"help": "Options: none, natural, json."},
    )

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        # filter out fields that are defined as field(init=False)
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}

        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d

@dataclass
class CustomArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    do_early_stopping: bool = field(
        default=True,
        metadata={"help": "Whether to do early stop or not. Default True"},
    )
    early_stopping_patience: Optional[int] = field(
        default=10,
        metadata={"help": "Number of evaluation calls with no improvement after which training will be stopped."},
    )
    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        # filter out fields that are defined as field(init=False)
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}

        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`WhisperProcessor`])
            The processor used for processing the data.
        decoder_start_token_id (`int`)
            The begin-of-sentence of the decoder.
        forward_attention_mask (`bool`)
            Whether to return attention_mask.
    """

    processor: Any
    decoder_start_token_id: int
    forward_attention_mask: bool

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        model_input_name = self.processor.model_input_names[0]
        input_features = [{model_input_name: feature[model_input_name]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        if self.forward_attention_mask:
            batch["attention_mask"] = torch.LongTensor([feature["attention_mask"] for feature in features])

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, CustomArguments))

    if sys.argv[1].endswith(".yaml"):
        model_args, data_args, training_args, custom_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_speech_recognition_seq2seq", model_args, data_args)

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # 3. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    if not data_args.dataset_name and not data_args.slurp_dataset_path and not data_args.italic_dataset_path and not data_args.test_dataset_name:
        raise ValueError(
                "At least one dataset has to be provided. "
                "Either specify dataset_name or use use_slurp_dataset."
                "Or test_data_name should be given."
            )


    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 4. Load dataset
    raw_datasets = DatasetDict()

    if training_args.do_train:
        slurp_train_dataset = None
        italic_train_dataset = None
        massive_train_dataset_list = None

        if data_args.slurp_dataset_path and data_args.train_slurp:
            logger.info(f"Slurp data set is being loaded for Training.")
            slurp_train_dataset = load_dataset(
                data_args.slurp_dataset_path,
                data_args.slurp_dataset_config_name,
                split=data_args.train_split_name,
                cache_dir=model_args.cache_dir,
            )

        if data_args.italic_dataset_path and data_args.train_italic:
            logger.info(f"ITALIC data set is being loaded for Training.")
            italic_train_dataset = load_dataset(
                data_args.italic_dataset_path,
                data_args.italic_dataset_config_name,
                split=data_args.train_split_name,
                cache_dir=model_args.cache_dir,
            )

        if data_args.dataset_name:
            logger.info(f"Speech-MASSIVE data set is being loaded for Training.")
            # TODO : Support multiple language cases
            massive_train_dataset_list = []
            dataset_config_list = data_args.dataset_config_name.split(",")
            
            few_shot_lang_list = copy.deepcopy(_V1_LANGS)

            for dataset_config in dataset_config_list:
                few_shot_lang_list.remove(dataset_config) # avoid duplicated few-shot dataset for the main language
                logger.info(f"Speech-MASSIVE {dataset_config} is being loaded for training.")
                massive_train_dataset_list.append(load_dataset(
                data_args.dataset_name,
                dataset_config,
                split=data_args.train_split_name,
                cache_dir=model_args.cache_dir,
            ))
            if data_args.is_few_shot:
                for few_shot_lang in few_shot_lang_list:
                    logger.info(f"Speech-MASSIVE {few_shot_lang} few-shot split is being loaded for training.")
                    massive_train_dataset_list.append(load_dataset(
                        data_args.dataset_name,
                        few_shot_lang,
                        split='train_115',
                        cache_dir=model_args.cache_dir,
                    ))

        raw_datasets_list = []

        if slurp_train_dataset:
            raw_datasets_list.append(slurp_train_dataset)
        if italic_train_dataset:
            raw_datasets_list.append(italic_train_dataset)
        if massive_train_dataset_list is not None and len(massive_train_dataset_list):
            raw_datasets_list.extend(massive_train_dataset_list)

        if len(raw_datasets_list) == 0:
            raise ValueError(
                "Something wrong with the loaded datasets."
            )
            
        logger.info(f'Train batch size : {training_args.train_batch_size}')

        min_single_dataset_len = min([len(x) for x in raw_datasets_list])

        # alawys make the size becomes xN of eval_batch_size

        raw_datasets['train'] = concatenate_datasets(raw_datasets_list)

        all_hparams = {}
        all_hparams.update(model_args.to_dict())
        all_hparams.update(training_args.to_dict())
        all_hparams.update(data_args.to_dict())
        all_hparams.update(custom_args.to_dict())

    if training_args.do_eval:
        slurp_eval_dataset = None
        massive_eval_dataset = None
        italic_eval_dataset = None

        num_eval_langs = 1
        selected_dataset_list = []
        random.seed(1)

        if data_args.slurp_dataset_path and data_args.eval_slurp:
            logger.info(f"Slurp data set is being loaded for Evaluation.")
            slurp_eval_dataset = load_dataset(
                data_args.slurp_dataset_path,
                data_args.slurp_dataset_config_name,
                split=data_args.eval_split_name,
                cache_dir=model_args.cache_dir,
            )
            slurp_eval_dataset = slurp_eval_dataset.shuffle(seed=1)

        if data_args.italic_dataset_path and data_args.eval_italic:
            logger.info(f"ITALIC data set is being loaded for Evaluation.")
            italic_eval_dataset = load_dataset(
                data_args.italic_dataset_path,
                data_args.italic_dataset_config_name,
                split=data_args.eval_split_name,
                cache_dir=model_args.cache_dir,
            )
            italic_eval_dataset = italic_eval_dataset.shuffle(seed=1)

        if data_args.eval_dataset_name:
            massive_eval_dataset = DatasetDict()
            eval_dataset_config_name = data_args.eval_dataset_config_name
            if eval_dataset_config_name == 'all':
                eval_langs = ['ar-SA', 'de-DE', 'es-ES', 'fr-FR', 'hu-HU', 'ko-KR', 'nl-NL', 'pl-PL', 'pt-PT', 'ru-RU', 'tr-TR', 'vi-VN']
                num_eval_langs = len(eval_langs)
            else:
                eval_langs = [eval_dataset_config_name]

            logger.info(f"Speech-MASSIVE data set is being loaded for Evaluation.")
            for lang in eval_langs:
                massive_eval_dataset[lang] = load_dataset(
                    data_args.eval_dataset_name,
                    lang,
                    split=data_args.eval_split_name,
                    cache_dir=model_args.cache_dir,
                )

        raw_datasets['eval'] = DatasetDict()

        #TODO: Remove redundancy
        #TODO: support for multi-gpu cases
        if slurp_eval_dataset and massive_eval_dataset and italic_eval_dataset:
            logger.info(f"Interleaving SLURP, ITALIC and Speech-MASSIVE dataset for evaluation.")
            massive_eval_dataset = massive_eval_dataset.remove_columns(['worker_id' ,'slot_method', 'judgments', 'bad_transcript_reported', 'recorder_id', 'validator_id'])
            num_eval_langs = len(massive_eval_dataset.keys()) + 2
            min_single_dataset_len = len(slurp_eval_dataset)
            logger.info(f'Num eval langs : {num_eval_langs}')
            logger.info(f'Eval batch size : {training_args.eval_batch_size}')

            for eval_dataset_split in massive_eval_dataset.keys():
                min_single_dataset_len = min(min_single_dataset_len, len(eval_dataset_split))

            # alawys make the size becomes xN of eval_batch_size
            if data_args.max_eval_samples:
                logger.info(f'Num max eval samples : {data_args.max_eval_samples}')
                num_eval_samples_per_lang = data_args.max_eval_samples // num_eval_langs // training_args.eval_batch_size
            else:
                num_eval_samples_per_lang = min_single_dataset_len // num_eval_langs // training_args.eval_batch_size

            if num_eval_samples_per_lang == 0:
                raise ValueError(f'num_eval_samples_per_lang cannot be 0.')
            # reverting it back, because now it is always xN of eval_batch_size
            num_eval_samples_per_lang = num_eval_samples_per_lang * training_args.eval_batch_size
            logger.info(f'Num samples of evaluation per language : {num_eval_samples_per_lang}')

            for eval_dataset_split in massive_eval_dataset.keys():                
                rand_eval_idx_list = random.choices(range(0, min_single_dataset_len), k=num_eval_samples_per_lang)
                selected_dataset_list.append(massive_eval_dataset[eval_dataset_split].select(rand_eval_idx_list))

            rand_eval_idx_list = random.choices(range(0, len(slurp_eval_dataset)), k=num_eval_samples_per_lang)
            selected_dataset_list.append(slurp_eval_dataset.select(rand_eval_idx_list))

            rand_eval_idx_list = random.choices(range(0, len(italic_eval_dataset)), k=num_eval_samples_per_lang)
            selected_dataset_list.append(italic_eval_dataset.select(rand_eval_idx_list))

            eval_dataset = concatenate_datasets(selected_dataset_list)
        elif massive_eval_dataset:
            massive_eval_dataset_keys = list(massive_eval_dataset.keys())
            num_eval_langs = len(massive_eval_dataset_keys)
            dataset_len = len(massive_eval_dataset[massive_eval_dataset_keys[0]]) * num_eval_langs
            logger.info(f"Interleaving Speech-MASSIVE dataset's different languages for evaluation.")
            logger.info(f'Num eval langs : {num_eval_langs}')
            logger.info(f'Eval batch size : {training_args.eval_batch_size}')

            # alawys make the size becomes xN of eval_batch_size
            if data_args.max_eval_samples:
                logger.info(f'Num max eval samples : {data_args.max_eval_samples}')
                num_eval_samples_per_lang = data_args.max_eval_samples // num_eval_langs // training_args.eval_batch_size
            else:
                # few samples will be discarded as it will be the remainders
                num_eval_samples_per_lang = dataset_len // num_eval_langs // training_args.eval_batch_size

            if num_eval_samples_per_lang == 0:
                raise ValueError(f'num_eval_samples_per_lang cannot be 0.')
            # reverting it back, because now it is always xN of eval_batch_size
            num_eval_samples_per_lang = num_eval_samples_per_lang * training_args.eval_batch_size
            logger.info(f'Num samples of evaluation per language : {num_eval_samples_per_lang}')

            for eval_dataset_split in massive_eval_dataset.keys():                
                dataset_len = len(massive_eval_dataset[eval_dataset_split])
                rand_eval_idx_list = random.choices(range(0,dataset_len), k=num_eval_samples_per_lang)
                selected_dataset_list.append(massive_eval_dataset[eval_dataset_split].select(rand_eval_idx_list))

            eval_dataset = concatenate_datasets(selected_dataset_list)
        elif slurp_eval_dataset:
            logger.info(f"Using only SLURP dataset for evaluation.")
            logger.info(f'Num eval langs : {num_eval_langs}')
            logger.info(f'Eval batch size : {training_args.eval_batch_size}')

            if data_args.max_eval_samples:
                logger.info(f'Num max eval samples : {data_args.max_eval_samples}')
                dataset_len = len(slurp_eval_dataset)
                num_eval_samples_per_lang = min(dataset_len, data_args.max_eval_samples)
                rand_eval_idx_list = random.choices(range(0,dataset_len), k=num_eval_samples_per_lang)
                eval_dataset = slurp_eval_dataset.select(rand_eval_idx_list)
            else:
                eval_dataset = slurp_eval_dataset
        elif italic_eval_dataset:
            logger.info(f"Using only ITALIC dataset for evaluation.")
            logger.info(f'Num eval langs : {num_eval_langs}')
            logger.info(f'Eval batch size : {training_args.eval_batch_size}')

            if data_args.max_eval_samples:
                logger.info(f'Num max eval samples : {data_args.max_eval_samples}')
                dataset_len = len(italic_eval_dataset)
                num_eval_samples_per_lang = min(dataset_len, data_args.max_eval_samples)
                rand_eval_idx_list = random.choices(range(0,dataset_len), k=num_eval_samples_per_lang)
                eval_dataset = italic_eval_dataset.select(rand_eval_idx_list)
            else:
                eval_dataset = italic_eval_dataset
        else:
            raise ValueError(
                "Something wrong with the loaded datasets."
            )

        raw_datasets["eval"] = eval_dataset
        
    if training_args.do_predict:
        if data_args.test_dataset_name:
            logger.info(f"Speech-MASSIVE data set is being loaded for Prediciton.")
            raw_datasets["test"] = load_dataset(
                data_args.test_dataset_name,
                data_args.test_dataset_config_name,
                split=data_args.test_split_name,
                cache_dir=model_args.cache_dir,
            )
        else:
            raise ValueError("--do_predict requires a test dataset")
        

    if data_args.audio_column_name not in next(iter(raw_datasets.values())).column_names:
        if training_args.do_predict and data_args.audio_column_name not in next(iter(raw_datasets['test'].values())).column_names:
            raise ValueError(
                f"--audio_column_name '{data_args.audio_column_name}' not found in dataset '{data_args.dataset_name}'. "
                "Make sure to set `--audio_column_name` to the correct audio column - one of "
                f"{', '.join(next(iter(raw_datasets.values())).column_names)}."
            )


    if data_args.text_column_name not in next(iter(raw_datasets.values())).column_names:
        if training_args.do_predict and data_args.text_column_name not in next(iter(raw_datasets['test'].values())).column_names:
            raise ValueError(
                f"--audio_column_name '{data_args.audio_column_name}' not found in dataset '{data_args.dataset_name}'. "
                "Make sure to set `--audio_column_name` to the correct audio column - one of "
                f"{', '.join(next(iter(raw_datasets.values())).column_names)}."
            )

    # 5. Load pretrained model, tokenizer, and feature extractor
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    config = WhisperConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    # SpecAugment for whisper models
    if getattr(config, "model_type", None) == "whisper":
        config.update({"apply_spec_augment": model_args.apply_spec_augment})

    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        model_args.feature_extractor_name if model_args.feature_extractor_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        # local_files_only=True,
    )
    tokenizer = WhisperTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        # language="en",
        task=data_args.task,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code
    )

    logger.warning(f'Setting model.config.forced_decoder_ids = None')
    logger.warning(f'This is to disable backward comaptibility in modeling_whisper.py')
    model.config.forced_decoder_ids = None

    if data_args.tokens_to_remove_from_suppress:
        # add additional tokens to be released in case of being generated with/without space
        token_list_to_release = []
        # token_list_to_release = [f'Ġ{x}' for x in data_args.tokens_to_remove_from_suppress]
        token_list_to_release.extend(data_args.tokens_to_remove_from_suppress)
        logger.warning(f'All tokens to be relased from suppressed : {token_list_to_release}')

        for token_to_release in token_list_to_release:
            token_idx = tokenizer._convert_token_to_id(token_to_release)
            if token_idx in model.generation_config.suppress_tokens:
                logger.warning(f'Releasing [{token_to_release}] - [idx:{token_idx}] from suppressed tokens.')
                model.generation_config.suppress_tokens.remove(token_idx)
            else:
                logger.warning(f'[{token_to_release}] - [idx:{token_idx}] is not in the suppressed tokens.')

    logger.warning(f'Updated models config - suppress_tokens : {model.generation_config.suppress_tokens}')

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if model_args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    if model_args.freeze_encoder:
        model.freeze_encoder()
        model.model.encoder.gradient_checkpointing = False

    # 7. Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    max_input_length = data_args.max_duration_in_seconds * feature_extractor.sampling_rate
    min_input_length = data_args.min_duration_in_seconds * feature_extractor.sampling_rate
    audio_column_name = data_args.audio_column_name
    num_workers = data_args.preprocessing_num_workers
    text_column_name = data_args.text_column_name
    model_input_name = feature_extractor.model_input_names[0]
    do_lower_case = data_args.do_lower_case

    # if SpecAugment is used for whisper models, return attention_mask to guide the mask along time axis
    forward_attention_mask = (
        getattr(config, "model_type", None) == "whisper"
        and getattr(config, "apply_spec_augment", False)
        and getattr(config, "mask_time_prob", 0) > 0
    )

    if data_args.max_train_samples is not None:
        raw_datasets["train"] = raw_datasets["train"].select(range(data_args.max_train_samples))

    def prepare_dataset(batch, add_slu_tag, replace_task_token, target_format_content, target_format_structure, separator):
        # process audio
        sample = batch[audio_column_name]
        # Whisper always match to 30 seconds
        inputs = feature_extractor(
            sample["array"], sampling_rate=sample["sampling_rate"], return_attention_mask=forward_attention_mask
        )
        # process audio length
        batch[model_input_name] = inputs.get(model_input_name)[0]
        # batch["input_length"] = len(sample["array"])
        batch["input_length"] = batch[model_input_name].shape[1] #(128, 3000)
        if forward_attention_mask:
            batch["attention_mask"] = inputs.get("attention_mask")[0]

        # Not to make the first word to be splitted, we need to add leading space
        # without additional space we get
        # ... '<|notimestamps|>', 'O', 'ther', 'ĠOther', 'ĠOther', 'Ġmusic', '_', 'qu', 'ery', '<|endoftext|>']
        # slots = ' ' + ' '.join(batch["labels"]) 
        transcript = batch["utt"]
        slots = ' '.join(batch["labels"]) 
        intent = batch["intent_str"]

        if target_format_content == 'transcript_slots_intent':
            if separator:
                if target_format_structure is None:
                    formatted_labels = f'{transcript} {separator} {slots} {separator} {intent}'
                if target_format_structure == 'natural':
                    formatted_labels = f'Transcript: {transcript} {separator} Slots: {slots} {separator} Intent: {intent}'
            else:
                if target_format_structure is None:
                    formatted_labels = f'{transcript} {slots} {intent}'
                if target_format_structure == 'natural':
                    formatted_labels = f'Transcript: {transcript} Slots: {slots} Intent: {intent}'
        elif target_format_content == 'slots_intent':
            if separator:
                if target_format_structure is None:
                    formatted_labels = f'{slots} {separator} {intent}'
                if target_format_structure == 'natural':
                    formatted_labels = f'Slots: {slots} {separator} Intent: {intent}'
            else:
                if target_format_structure is None:
                    formatted_labels = f'{slots} {intent}'
                if target_format_structure == 'natural':
                    formatted_labels = f'Slots: {slots} Intent: {intent}'
        elif target_format_content == 'transcript_intent':
            if separator:
                if target_format_structure is None:
                    formatted_labels = f'{transcript} {separator} {intent}'
                if target_format_structure == 'natural':
                    formatted_labels = f'Transcript: {transcript} {separator} Intent: {intent}'
            else:
                if target_format_structure is None:
                    formatted_labels = f'{transcript} {intent}'
                if target_format_structure == 'natural':
                    formatted_labels = f'Transcript: {transcript} Intent: {intent}'
        else:
            raise ValueError(f'Unsupported {target_format_content}.')


        # below adding tags should be done before covnerting to ids
        batch["labels"] = tokenizer(formatted_labels).input_ids

        if replace_task_token:
            index_to_replace = None
            try:
                index_to_replace = batch["labels"].index(tokenizer.convert_tokens_to_ids('<|transcribe|>'))
            except:
                index_to_replace = batch["labels"].index(tokenizer.convert_tokens_to_ids('<|translate|>'))
            if not index_to_replace:
                raise ValueError(f'Could not locate the task token in the labels {batch["labels"]}.')
            batch["labels"][index_to_replace] = tokenizer.convert_tokens_to_ids("<|startoflm|>")

        if add_slu_tag:
            if replace_task_token:
                raise ValueError(
                    'add_slu_tag cannot be done when replace_task_token is True.'
                    'make sure use only one of them'
                    )
            try:
                index_to_insert_tag = batch["labels"].index(tokenizer.convert_tokens_to_ids('<|transcribe|>')) + 1
            except:
                index_to_insert_tag = batch["labels"].index(tokenizer.convert_tokens_to_ids('<|translate|>')) + 1
            batch["labels"].insert(index_to_insert_tag, tokenizer.convert_tokens_to_ids("<|startoflm|>"))
        
        # insert language code
        lang_code = f'<|{batch["locale"][:2]}|>'
        batch["labels"].insert(1, tokenizer.convert_tokens_to_ids(lang_code))
        return batch

    with training_args.main_process_first(desc="dataset map pre-processing"):
        if training_args.do_predict:
            predict_dataset_dict = {}
            predict_dataset = raw_datasets["test"]
            columns_to_remove = next(iter(predict_dataset.values())).column_names
            columns_to_remove.remove('id')
            columns_to_remove.remove('annot_utt')
            columns_to_remove.remove('labels')
            columns_to_remove.remove('intent_str')

            if "eval" in data_args.test_dataset_config_name:
                split_prefix = 'validation'
            elif "test" in data_args.test_dataset_config_name:
                split_prefix = 'test'
            else:
                raise ValueError(f'not supported format {data_args.test_dataset_config_name}')

            lang_list = ['_'.join(test_set_name.split('_')[1:]) for test_set_name in list(predict_dataset.keys())]

            for lang in lang_list:
                split_name = f'{split_prefix}_{lang}'
                predict_dataset_dict[split_name] = predict_dataset[split_name].map(
                    prepare_dataset,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=columns_to_remove,
                    desc="preprocess test datasets ...",
                    load_from_cache_file=not data_args.overwrite_cache,
                    fn_kwargs={
                        'add_slu_tag': data_args.add_slu_tag,
                        'replace_task_token': data_args.replace_task_token,
                        'target_format_content': data_args.target_format_content,
                        'target_format_structure': data_args.target_format_structure,
                        'separator': data_args.add_separator,
                    }
                )
        else:
            if training_args.do_train:
                train_columns_to_remove  = list(raw_datasets['train'].features.keys())
                train_columns_to_remove.remove('annot_utt')
                train_columns_to_remove.remove('labels')
                train_columns_to_remove.remove('intent_str')
                train_dataset = raw_datasets['train'].map(
                    prepare_dataset,
                    remove_columns=train_columns_to_remove,
                    num_proc=data_args.preprocessing_num_workers,
                    desc="preprocess train datasets ...",
                    load_from_cache_file=not data_args.overwrite_cache,
                    fn_kwargs={
                        'add_slu_tag': data_args.add_slu_tag,
                        'replace_task_token': data_args.replace_task_token,
                        'target_format_content': data_args.target_format_content,
                        'target_format_structure': data_args.target_format_structure,
                        'separator': data_args.add_separator,
                        }
                )
            if training_args.do_eval:
                eval_columns_to_remove  = list(raw_datasets['eval'].features.keys())
                eval_columns_to_remove.remove('annot_utt')
                eval_columns_to_remove.remove('labels')
                eval_columns_to_remove.remove('intent_str')

                eval_dataset = raw_datasets['eval'].map(
                    prepare_dataset,
                    remove_columns=eval_columns_to_remove,
                    num_proc=data_args.preprocessing_num_workers,
                    desc="preprocess eval datasets ...",
                    load_from_cache_file=not data_args.overwrite_cache,
                    fn_kwargs={
                        'add_slu_tag': data_args.add_slu_tag,
                        'replace_task_token': data_args.replace_task_token,
                        'target_format_content': data_args.target_format_content,
                        'target_format_structure': data_args.target_format_structure,
                        'separator': data_args.add_separator,
                        }
                )

    # filter data that is shorter than min_input_length or longer than
    # max_input_length
    def is_audio_in_length_range(length):
        return length > min_input_length and length < max_input_length

    if not training_args.do_predict:
        if training_args.do_train:
            train_dataset = train_dataset.filter(
                is_audio_in_length_range,
                num_proc=num_workers,
                input_columns=["input_length"],
            )
        if training_args.do_eval:
            eval_dataset = eval_dataset.filter(
                is_audio_in_length_range,
                num_proc=num_workers,
                input_columns=["input_length"],
            )

    # for large datasets it is advised to run the preprocessing on a
    # single machine first with `args.preprocessing_only` since there will mostly likely
    # be a timeout when running the script in distributed mode.
    # In a second step `args.preprocessing_only` can then be set to `False` to load the
    # cached dataset
    if data_args.preprocessing_only:
        cache_train = {k: v.cache_files for k, v in train_dataset.items()}
        logger.info(f"Train data preprocessing finished. Files cached at {cache_train}.")
        cache_eval = {k: v.cache_files for k, v in eval_dataset.items()}
        logger.info(f"Eval data preprocessing finished. Files cached at {cache_eval}.")
        return

    # 8. Load Metric
    metric = evaluate.load("wer")
    audio_sample_idx_list = []
    #TODO: to be able to play audio in evaluation, we need audio_path
    def compute_metrics(pred):
        global_step = trainer.state.global_step
        epoch = trainer.state.epoch

        pred_ids = pred.predictions
        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)

        # evalaute on S3PRL way (slot-type f1, slot-value CER)
        def _parse_prediction(target_format_content, separator, pred_list):
            output = {
                'massive_nlu_eval_format' : [],
                'transcript' : [],
                'slots' : [],
                'intent' : []
            }

            for pred in pred_list:
                if target_format_content == 'transcript_slots_intent':
                    splitted = pred.split(separator)
                    transcript_pred = ''
                    slots_pred = ''
                    intent_pred = ''

                    if len(splitted) >= 3:
                        transcript_pred = splitted[0].strip()
                        slots_pred = splitted[1].strip()
                        intent_pred = splitted[2].strip()
                    elif len(splitted) == 2:
                        transcript_pred = splitted[0].strip()
                        slots_pred = splitted[1].strip()
                    elif len(splitted) == 1:
                        transcript_pred = splitted[0].strip()
                    else:
                        pass
                        # cannot evaluate
                    # handle the case where not all the components are produced
                    massive_nlu_eval_format = f'{slots_pred} {intent_pred}' # this part will be used to evaluate the tool of NLU
                    if not slots_pred and not intent_pred:
                        massive_nlu_eval_format = ''
                elif target_format_content == 'transcript_intent':
                    splitted = pred.split(separator)
                    transcript_pred = ''
                    slots_pred = ''
                    intent_pred = ''

                    if len(splitted) >= 2:
                        transcript_pred = splitted[0].strip()
                        intent_pred = splitted[1].strip()
                    elif len(splitted) == 1:
                        transcript_pred = splitted[0].strip()
                    else:
                        pass
                        # cannot evaluate
                    # handle the case where not all the components are produced
                    massive_nlu_eval_format = f'{slots_pred} {intent_pred}' # this part will be used to evaluate the tool of NLU
                    if not slots_pred and not intent_pred:
                        massive_nlu_eval_format = ''
                else:
                    print('unsupported format yet')

                output['massive_nlu_eval_format'].append(massive_nlu_eval_format)
                output['transcript'].append(transcript_pred),
                output['slots'].append(slots_pred),
                output['intent'].append(intent_pred)
            
            return output

        pred_parse_result_list = _parse_prediction(data_args.target_format_content, data_args.add_separator, pred_str)
        label_parse_result_list = _parse_prediction(data_args.target_format_content, data_args.add_separator, label_str)

        massive_nlu_eval_format_pred = pred_parse_result_list['massive_nlu_eval_format']
        transcript_pred_list = pred_parse_result_list['transcript']
        slots_pred_list= pred_parse_result_list['slots']
        intent_pred = pred_parse_result_list['intent']

        massive_nlu_eval_format_label = label_parse_result_list['massive_nlu_eval_format']
        transcript_label_list = label_parse_result_list['transcript']
        slots_label_list = label_parse_result_list['slots']
        intent_label = label_parse_result_list['intent']

        slot_type_f1 = s3prl_slot_eval.slot_type_f1(slots_pred_list, transcript_pred_list, slots_label_list, transcript_label_list)
        slot_value_cer = s3prl_slot_eval.slot_value_cer(slots_pred_list, transcript_pred_list, slots_label_list, transcript_label_list)

        result = {}
        result['slot_type_f1'] = slot_type_f1
        result['slot_value_cer'] = slot_value_cer

        eval_instance = MassiveEval()
        intents_pred, slots_pred_all = eval_instance.convert_t2t_batch_to_intents_slots(
            massive_nlu_eval_format_pred, eval_instance.t2t_args)
        intents_lab, slots_lab_all = eval_instance.convert_t2t_batch_to_intents_slots(
            massive_nlu_eval_format_label, eval_instance.t2t_args)

        if not training_args.do_predict:
            random.seed(1)

            temp_rand_idx_list = random.choices(range(0, len(pred_str)), k=num_eval_langs*5)
            rand_idx_list = list(set(temp_rand_idx_list))

            log_folder_path = f'{training_args.output_dir}/eval_log/step-{global_step}'
            Path(log_folder_path).mkdir(parents=True, exist_ok=True)
            eval_log_file_pred = open(f'{log_folder_path}/pred.txt', 'w')
            eval_log_file_label = open(f'{log_folder_path}/label.txt', 'w')
            eval_log_file_source = open(f'{log_folder_path}/source.txt', 'w')
            eval_log_file_audio = open(f'{log_folder_path}/audio.txt', 'w')

            for sample_idx in rand_idx_list:
                lang = raw_datasets['eval'][sample_idx]['locale']
                source_text = raw_datasets['eval'][sample_idx]['annot_utt']
                audio_path = raw_datasets['eval'][sample_idx]['audio']['path']
                audio_filename = str(os.path.basename(audio_path))
                audio_full_dir = os.path.dirname(audio_path)
                audio_dir = str(os.path.basename(audio_full_dir))
                audio_caption = f'{audio_dir}/{audio_filename}'

                eval_log_file_pred.write(f'{sample_idx} - {lang}\t:{pred_str[sample_idx]}\n')
                eval_log_file_label.write(f'{sample_idx} - {lang}\t:{label_str[sample_idx]}\n')
                eval_log_file_source.write(f'{sample_idx} - {lang}\t:{source_text}\n')
                eval_log_file_audio.write(f'{sample_idx} - {lang}\t:{audio_path}\n')
            
            eval_log_file_pred.close()
            eval_log_file_label.close()
            eval_log_file_source.close()
            eval_log_file_audio.close()

        result.update(eval_instance.eval_preds(
            pred_intents=intents_pred,
            lab_intents=intents_lab,
            pred_slots=slots_pred_all,
            lab_slots=slots_lab_all,
            # eval_metrics=metrics,
            eval_metrics='all',
            labels_ignore='Other',
            # labels_ignore=ignore_labels,
            pad='Other'
        ))

        return result

    # 9. Create a single speech processor
    # make sure all processes wait until data is saved
    with training_args.main_process_first():
        # only the main process saves them
        if is_main_process(training_args.local_rank):
            # save feature extractor, tokenizer and config
            feature_extractor.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            config.save_pretrained(training_args.output_dir)

    processor = WhisperProcessor.from_pretrained(training_args.output_dir)

    # 10. Define data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        forward_attention_mask=forward_attention_mask,
    )

    if data_args.add_slu_tag:
        training_args.__setattr__('add_slu_tag', '|')

    # 11. Initialize Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    if custom_args.do_early_stopping:
        trainer.add_callback(EarlyStoppingCallback(custom_args.early_stopping_patience)) # add it to the condition

    # 12. Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the feature extractor too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # 13. Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            metric_key_prefix="eval",
            max_length=training_args.generation_max_length,
            num_beams=training_args.generation_num_beams,
        )

        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        for name, predict_dataset in predict_dataset_dict.items():
            logger.info(f"*** {name} ***")
        
            predict_results = trainer.predict(
                predict_dataset, metric_key_prefix="predict",
                max_length=training_args.generation_max_length,
                num_beams=training_args.generation_num_beams
            )

            metrics = predict_results.metrics
            metrics["predict_samples"] = len(predict_dataset)

            trainer.log_metrics(f"predict_{name}", metrics)
            trainer.save_metrics(f"predict_{name}", metrics)

            if trainer.is_world_process_zero():
                if training_args.predict_with_generate:
                    predictions = predict_results.predictions
                    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
                    predictions = tokenizer.batch_decode(
                        predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    predictions = [pred.strip() for pred in predictions]
                    output_prediction_file = os.path.join(training_args.output_dir, f"{name}_predictions.txt")
                    with open(output_prediction_file, "w", encoding="utf-8") as writer:
                        writer.write("\n".join(predictions))

                    labels = predict_results.label_ids
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                    labels = tokenizer.batch_decode(
                        labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    labels = [label.strip() for label in labels]
                    output_label_file = os.path.join(training_args.output_dir, f"{name}_labels.txt")
                    with open(output_label_file, "w", encoding="utf-8") as writer:
                        writer.write("\n".join(labels))

                    annot_utt_list = predict_dataset['annot_utt']
                    sample_id_list = predict_dataset['id']
                    
                    anoot_utt_w_idx = [f'{sample_id}\t: {annot_utt.strip()}' for annot_utt, sample_id in zip(annot_utt_list, sample_id_list)]
                    output_source_file = os.path.join(training_args.output_dir, f"{name}_sources.txt")
                    with open(output_source_file, "w", encoding="utf-8") as writer:
                        writer.write("\n".join(anoot_utt_w_idx))

    # 14. Write Training Stats
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "automatic-speech-recognition"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    return results


if __name__ == "__main__":
    main()
