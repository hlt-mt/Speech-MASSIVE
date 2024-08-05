#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 FBK and NAVER LABS Europe. All rights reserved.
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

from copy import deepcopy
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union)

import torch
from torch import nn
from torch.utils.data import Dataset
from torch import Tensor
from torch import optim


from transformers.generation.configuration_utils import GenerationConfig
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer import Trainer
from transformers.utils import logging


if TYPE_CHECKING:
    from transformers.data.data_collator import DataCollator
    from transformers.modeling_utils import PreTrainedModel
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from transformers.trainer_callback import TrainerCallback
    from transformers.trainer_utils import EvalPrediction, PredictionOutput
    from transformers.training_args import TrainingArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Trainer):
    def __init__(
            self,
            model: Union["PreTrainedModel", nn.Module] = None,
            args: "TrainingArguments" = None,
            data_collator: Optional["DataCollator"] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            tokenizer: Optional["PreTrainedTokenizerBase"] = None,
            model_init: Optional[Callable[[], "PreTrainedModel"]] = None,
            compute_metrics: Optional[Callable[["EvalPrediction"], Dict]] = None,
            callbacks: Optional[List["TrainerCallback"]] = None,
            optimizers: Tuple[optim.Optimizer, optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Optional[Callable[[Tensor, Tensor], Tensor]] = None):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics)

        # Override self.model.generation_config if a GenerationConfig is specified in args.
        # Priority: args.generation_config > model.generation_config > default GenerationConfig.
        if self.args.generation_config is not None:
            gen_config = self.load_generation_config(self.args.generation_config)
            self.model.generation_config = gen_config

    @staticmethod
    def load_generation_config(gen_config_arg: Union[str, GenerationConfig]) -> GenerationConfig:
        """
        Loads a `~generation.GenerationConfig` from the
        `Seq2SeqTrainingArguments.generation_config` arguments.

        Args:
            gen_config_arg (`str` or [`~generation.GenerationConfig`]):
                `Seq2SeqTrainingArguments.generation_config` argument.

        Returns:
            A `~generation.GenerationConfig`.
        """

        # GenerationConfig provided, nothing to do
        if isinstance(gen_config_arg, GenerationConfig):
            return deepcopy(gen_config_arg)

        # str or Path
        pretrained_model_name = Path(gen_config_arg) if isinstance(gen_config_arg, str) \
            else gen_config_arg
        config_file_name = None

        # Figuring if it is path pointing to a file,
        # pointing to a directoryor else a model id or URL
        # This step is required in order to determine config_file_name
        if pretrained_model_name.is_file():
            config_file_name = pretrained_model_name.name
            pretrained_model_name = pretrained_model_name.parent
        # dir path
        elif pretrained_model_name.is_dir():
            pass
        # model id or URL
        else:
            pretrained_model_name = gen_config_arg

        gen_config = GenerationConfig.from_pretrained(pretrained_model_name, config_file_name)
        return gen_config

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            **gen_kwargs) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they
        are task-dependent (pass it to the init `compute_metrics` argument).
        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`.
                If it is an [`~datasets.Dataset`], columns not accepted by the `model.forward()`
                method are automatically removed. It must implement the `__len__` method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be
                ignored when gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics
                "bleu" will be named "eval_bleu" if the prefix is `"eval"` (default).
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate
                method. 1 means no beam search.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from
            the predictions. The dictionary also contains the epoch number which comes from the
            training state.
        """

        gen_kwargs = gen_kwargs.copy()

        # Use legacy argument setting
        # if a) the option is not explicitly passed;
        # and b) the argument is set in the training args
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None \
                and self.args.generation_max_length is not None:
            gen_kwargs["max_length"] = self.args.generation_max_length
        if gen_kwargs.get("num_beams") is None and self.args.generation_num_beams is not None:
            gen_kwargs["num_beams"] = self.args.generation_num_beams
        # We don't want to drop samples in general
        self.gather_function = self.accelerator.gather
        self._gen_kwargs = gen_kwargs
        return super().evaluate(
            eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix)

    def predict(
            self,
            test_dataset: Dataset,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "test",
            **gen_kwargs) -> "PredictionOutput":
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels.
        In that case, this method will also return metrics, like in `evaluate()`.

        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is a [`~datasets.Dataset`], columns not
                accepted by the `model.forward()` method are automatically removed. Has to
                implement the method `__len__`.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be
                ignored when gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics
                "bleu" will be named "eval_bleu" if the prefix is `"eval"` (default).
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate
                method. 1 means no beam search.
            gen_kwargs:
                Additional `generate` specific kwargs.

        <Tip>

        If your predictions or labels have different sequence lengths (for instance because you
        are doing dynamic padding in a token classification task), the predictions will be padded
        (on the right) to allow for concatenation into one array. The padding index is -100.

        </Tip>

        Returns: *NamedTuple* A namedtuple with the following keys:

            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics
              (if the dataset contained labels).
        """

        gen_kwargs = gen_kwargs.copy()

        # Use legacy argument setting
        # if a) the option is not explicitly passed;
        # and b) the argument is set in the training args
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None \
                and self.args.generation_max_length is not None:
            gen_kwargs["max_length"] = self.args.generation_max_length
        if gen_kwargs.get("num_beams") is None and self.args.generation_num_beams is not None:
            gen_kwargs["num_beams"] = self.args.generation_num_beams

        self.gather_function = self.accelerator.gather
        self._gen_kwargs = gen_kwargs

        return super().predict(
            test_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix)

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
            **gen_kwargs) -> Tuple[Optional[float], Optional[Tensor], Optional[Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model.
                Most models expect the targets under the argument `labels`.
                Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Return:
            Tuple[Optional[float], Optional[Tensor], Optional[Tensor]]:
            A tuple with the loss, logits and labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys)

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # Priority (handled in generate):
        # non-`None` gen_kwargs > model.generation_config > default GenerationConfig()
        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()
        if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
            gen_kwargs.pop("num_beams")
        if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
            gen_kwargs.pop("max_length")

        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is \
            not None else default_synced_gpus

        generation_inputs = inputs.copy()
        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model
        # can freely generate. Otherwise, it would continue generating from the padded
        # `decoder_input_ids`.
        if "labels" in generation_inputs and "decoder_input_ids" in generation_inputs and \
                generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape:
            generation_inputs = {
                k: v
                for k, v in inputs.items()
                if k not in ("decoder_input_ids", "decoder_attention_mask")}

        lang_code_tok = int(generation_inputs["labels"][0][0])
        task_tok = int(generation_inputs["labels"][0][1])
        slu_tok = None

        if self.args.add_slu_tag:
            slu_tok = int(generation_inputs["labels"][0][2])
        else:
            logger.info("add_slu_tag is NOT being added to the forced_decoder_ids")

        try:
            no_timestamp_tok = self.tokenizer.convert_tokens_to_ids(["<|notimestamps|>"])[0]
        except ValueError:
            raise ValueError(
                "<|notimestamps|> token does not exist in the tokenizer. Please check the "
                "tokenizer.")

        forced_decoder_ids = [[1, lang_code_tok], [2, task_tok]]
        if slu_tok:
            forced_decoder_ids.append([3, slu_tok])
            forced_decoder_ids.append([4, no_timestamp_tok])
        else:
            forced_decoder_ids.append([3, no_timestamp_tok])

        logger.warning(
            "Models forced_decoder_ids : "
            f"{self.model.generation_config.forced_decoder_ids}")
        if generation_inputs["labels"].shape[0] > 1:  # bsz 1 for eval case
            logger.warning(
                f"Eval batch size {generation_inputs['labels'].shape[0]} is larger than 1. Below "
                "forced_token_ids will be FORCEFULLY applied to all the samples in this batch")
        else:
            logger.warning(
                "We are FORCING new forced_decoder_ids to support multilingual. Overall "
                f"forced_decoder_ids : {forced_decoder_ids}. Bos token will be automatically "
                "preprended in the modeling code.")

        original_forced_decoder_ids = self.model.generation_config.forced_decoder_ids
        self.model.generation_config.__setattr__("forced_decoder_ids", forced_decoder_ids)
        generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)

        logger.warning("Reverting back to orginal models forced_decoder_ids")
        logger.warning(f"was : {self.model.generation_config.forced_decoder_ids}")

        self.model.generation_config.__setattr__("forced_decoder_ids", original_forced_decoder_ids)
        logger.warning(f"now : {self.model.generation_config.forced_decoder_ids}")

        # Temporary hack to ensure the generation config is not initialized for each iteration
        # of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from
        # a model config is
        # removed in https://github.com/huggingface/transformers/blob
        # /98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens,
                gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < \
                gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens,
                gen_config.max_new_tokens + 1)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                    loss = loss.mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < \
                    gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None

        return loss, generated_tokens, labels

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None \
                else self.tokenizer.eos_token_id
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError(
                    "Pad_token_id must be set in the configuration of the model, in order to pad "
                    "tensors")

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device)
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor
