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

"""
original evaluation code from MASSIVE code repository
https://github.com/alexa/massive/blob/main/src/massive/utils/training_utils.py

"""

from math import sqrt
from seqeval.metrics import f1_score
import sklearn.metrics as sklm


class MassiveEval:

    def __init__(self):
        self.t2t_args = {
            "input_prompt": "Annotate: ",
            "use_output_descrip": False,
            "intent_first": False,
            "slots_mixed": False,
            "toks_in_output": False,
            "sentinels": False,
            "inside_format": "slot_name",
            "outside_label": "Other"}

    def convert_to_bio(self, seq_tags, outside="Other", labels_merge=None):
        """
        Converts a sequence of tags into BIO format. EX:

            ['city', 'city', 'Other', 'country', -100, 'Other']
            to
            ['B-city', 'I-city', 'O', 'B-country', 'I-country', 'O']
            where outside = 'Other' and labels_merge = [-100]

        :param seq_tags: the sequence of tags that should be converted
        :type seq_tags: list
        :param outside: The label(s) to put outside (ignore). Default: 'Other'
        :type outside: str or list
        :param labels_merge: The labels to merge leftward (i.e. for tokenized inputs)
        :type labels_merge: str or list
        :return: a BIO-tagged sequence
        :rtype: list
        """

        seq_tags = [str(x) for x in seq_tags]

        outside = [outside] if type(outside) is not list else outside
        outside = [str(x) for x in outside]

        if labels_merge:
            labels_merge = [labels_merge] if type(labels_merge) is not list else labels_merge
            labels_merge = [str(x) for x in labels_merge]
        else:
            labels_merge = []

        bio_tagged = []
        prev_tag = None
        for tag in seq_tags:
            if prev_tag is None and tag in labels_merge:
                bio_tagged.append("O")
            elif tag in outside:
                bio_tagged.append("O")
                prev_tag = tag
            elif tag != prev_tag and tag not in labels_merge:
                bio_tagged.append("B-" + tag)
                prev_tag = tag
            elif tag == prev_tag or tag in labels_merge:
                if prev_tag in outside:
                    bio_tagged.append("O")
                else:
                    bio_tagged.append("I-" + prev_tag)

        return bio_tagged

    def eval_preds(
            self,
            pred_intents=None,
            lab_intents=None,
            pred_slots=None,
            lab_slots=None,
            eval_metrics="all",
            labels_ignore="Other",
            labels_merge=None,
            pad="Other"):
        """
        Function to evaluate the predictions from a model

        :param pred_intents: a list of predicted intents
        :type pred_intents: list
        :param lab_intents: a list of intents labels (ground truth)
        :type lab_intents: list
        :param pred_slots:
        a list of predicted slots,
        where each entry is a list of token-based slots
        :type pred_slots: list
        :param lab_slots: a list of slots labels (ground truth)
        :type lab_slots: list
        :param eval_metrics: The metrics to include.
                             Options are 'all', 'intent_acc', 'ex_match_acc', 'slot_micro_f1'
        :type eval_metrics: str
        :param labels_ignore: The labels to ignore (prune away). Default: ['Other']
        :type labels_ignore: str or list
        :param labels_merge: The labels to merge leftward (i.e. for tokenized inputs)
        :type labels_merge: str or list
        :param pad: The value to use when padding slot predictions to match
                    the length of ground truth
        :type pad: str
        """

        results = {}

        # Check lengths
        if pred_intents is not None and lab_intents is not None:
            assert len(pred_intents) == len(lab_intents), \
                "pred_intents and lab_intents must be same len"
        if pred_slots is not None and lab_slots is not None:
            assert len(pred_slots) == len(lab_slots), \
                "pred_slots and lab_slots must be same length"

        if ("intent_acc" in eval_metrics) or ("all" in eval_metrics):
            intent_acc = sklm.accuracy_score(lab_intents, pred_intents)
            results["intent_acc"] = intent_acc
            # Assuming normal distribution. Multiply by z (from "z table") to get confidence int
            results["intent_acc_stderr"] = sqrt(intent_acc * (1 - intent_acc) / len(pred_intents))

        if lab_slots is not None and pred_slots is not None:
            bio_slot_labels, bio_slot_preds = [], []
            for lab, pred in zip(lab_slots, pred_slots):

                # Pad or truncate prediction as needed using `pad` arg
                if type(pred) is list:
                    pred = pred[: len(lab)] + [pad] * (len(lab) - len(pred))

                # Fix for Issue 21 -- subwords after the first one from a word should be ignored
                for i, x in enumerate(lab):
                    if x == -100:
                        pred[i] = -100

                # convert to BIO
                bio_slot_labels.append(
                    self.convert_to_bio(lab, outside=labels_ignore, labels_merge=labels_merge))
                bio_slot_preds.append(
                    self.convert_to_bio(pred, outside=labels_ignore, labels_merge=labels_merge))

        if ("slot_micro_f1" in eval_metrics) or ("all" in eval_metrics):

            # from seqeval
            smf1 = f1_score(bio_slot_labels, bio_slot_preds)
            results["slot_micro_f1"] = smf1
            # Assuming normal distribution. Multiply by z (from "z table") to get confidence int
            total_slots = sum([len(x) for x in bio_slot_preds])
            results["slot_micro_f1_stderr"] = sqrt(smf1 * (1 - smf1) / total_slots)

        if ("ex_match_acc" in eval_metrics) or ("all" in eval_metrics):
            # calculate exact match accuracy (~0.01 seconds)
            matches = 0
            denom = 0
            for p_int, p_slot, l_int, l_slot in zip(
                    pred_intents, bio_slot_preds, lab_intents, bio_slot_labels):

                if (p_int == l_int) and (p_slot == l_slot):
                    matches += 1
                denom += 1
            emacc = matches / denom

            results["ex_match_acc"] = emacc
            # Assuming normal distribution. Multiply by z (from "z table") to get confidence int
            results["ex_match_acc_stderr"] = sqrt(emacc * (1 - emacc) / len(pred_intents))

        return results

    def convert_t2t_batch_to_intents_slots(
            self,
            mod_out,
            use_output_descrip=False,
            intent_first=False,
            slots_mixed=False,
            toks_in_output=False,
            sentinels=False,
            inside_format="slot_name",
            outside_label="Other",
            **kwargs):
        """
        Helper function to convert an intent and 0 or more slots to a text-to-text format

        :param model_out: A list of outputs from the model, each a detokenized string
        :type model_out: list
        :param use_output_descrip:
            Whether or not to include descriptive prompts in the output,
            being 'tokens: ' and 'annotations' for non mixed slotting or 'annotation: '
            for mixed slotting. Default: False
        :type use_output_descrip: bool
        :param intent_first:
            Whether to put the intent before the slots and utterance (True) or
            after Default: True
        :type intent_first: bool
        :param slots_mixed:
            Whether to put each slot after its respective token (True) or
            to put all slots after all tokens (False). Default: False
        :type slots_mixed: bool
        :param input_prompt:
            The text prompt for the input. Leave blank for no prompt.
            Default: 'Annotate: '
        :type input_prompt: str
        :param toks_in_output:
            Whether to put tokens in the output or not. Default: False.
            If this is True, then slots_mixed must be False
        :type toks_in_output: bool
        :param sentinels:
            Whether to add T5 sentinels before each token. Overrides toks_in_output and
            slots_mixed. Default: False
            See: https://arxiv.org/pdf/2203.08378.pdf
        :type sentinels: bool
        :param inside_format:
            The slot to use for the inside of a multi-word slot. Options are
            "slot_name", in which the slot name is repeated, "inside_slot_name",
            in which "I-" is added to the slot name, or "inside", in which "I" is
            used on its own.
        :type inside_format: str
        :param outside_label: The word used for non-slotted tokens. Default: Other
        :type outside_label: str

        :return: a list of intents, a list of slot lists
        :rtype: list
        """

        if sentinels:
            # using sentinels is the same as doing slots_mixed and toks_in_output and
            # converting the utterance to a sequence of sentinels
            toks_in_output = True
            slots_mixed = True
            for example in mod_out:
                new_utt, sent_id = [], 0
                for tok in example:
                    new_utt.append("<extra_id_" + str(sent_id) + ">")
                    sent_id += 1
                example = new_utt

        # Get intents
        if intent_first and use_output_descrip:
            # Note: this assumes that the description is one word
            intents_pred = [x.split()[1] if len(x.split()) > 1 else "" for x in mod_out]
        elif intent_first:
            intents_pred = [x.split()[0] for x in mod_out]
        else:
            intents_pred = []
            for x in mod_out:
                try:
                    intents_pred.append(x.split()[-1])
                except IndexError:
                    intents_pred.append("")
            # intents_pred = [x.split()[-1] for x in mod_out]

        # Determine Slots. Note: this assumes that the description is one word
        descrip_shift = 0
        if use_output_descrip:
            descrip_shift = 1

        if intent_first:
            # Everthing after the intent
            slot_chunk_pred = [x.split()[(1 + 2 * descrip_shift):] for x in mod_out]
        else:
            # Everything until the intent
            slot_chunk_pred = [
                x.split()[(descrip_shift): (-1 * (descrip_shift + 1))]
                for x in mod_out]
        if toks_in_output and slots_mixed:
            # Grab every other item
            slots_pred = [x[1::2] for x in slot_chunk_pred]
        elif toks_in_output:
            slots_pred = []
            # Assume equal number of tokens and slots and take second half
            for pred in slot_chunk_pred:
                pred = pred[descrip_shift:]
                mid = len(pred) // 2
                slots_pred.append(pred[mid:])
        else:
            slots_pred = slot_chunk_pred

        # Modify for inside format if needed
        for s_idx, slots in enumerate(slots_pred):
            new_slots = []
            for idx, slot in enumerate(slots):
                if idx > 0 and slot != outside_label:
                    if inside_format == "inside_slot_name":
                        if slot.startswith("I-"):
                            new_slots.append(slots[idx - 1])
                            continue
                    elif inside_format == "inside":
                        if slot == "I":
                            new_slots.append(slots[idx - 1])
                            continue
                new_slots.append(slot)
            slots_pred[s_idx] = new_slots

        return intents_pred, slots_pred
