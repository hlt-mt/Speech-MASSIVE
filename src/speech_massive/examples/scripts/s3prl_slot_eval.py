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
original evaluation code from s3prl code repository
https://github.com/s3prl/s3prl/blob/aa3ba844bfe2b5402b7f345cbebd72b33ef6aeff/s3prl/metric/common.py
https://github.com/s3prl/s3prl/blob/aa3ba844bfe2b5402b7f345cbebd72b33ef6aeff/s3prl/metric/slot_filling.py

Original authors
Commonly used metrics

Authors
  * Shu-wen Yang 2022
  * Heng-Jui Chang 2022
  * Haibin Wu 2022

Metrics for the slot filling SLU task

Authors:
  * Yung-Sung Chuang 2021
  * Heng-Jui Chang 2022
"""

import re
from typing import Dict, List, Tuple, Union
import editdistance as ed
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve


def accuracy(xs, ys, item_same_fn=None):
    if isinstance(xs, (tuple, list)):
        assert isinstance(ys, (tuple, list))
        return _accuracy_impl(xs, ys, item_same_fn)
    elif isinstance(xs, dict):
        assert isinstance(ys, dict)
        keys = sorted(list(xs.keys()))
        xs = [xs[k] for k in keys]
        ys = [ys[k] for k in keys]
        return _accuracy_impl(xs, ys, item_same_fn)
    else:
        raise ValueError


def _accuracy_impl(xs, ys, item_same_fn=None):
    item_same_fn = item_same_fn or (lambda x, y: x == y)
    same = [int(item_same_fn(x, y)) for x, y in zip(xs, ys)]
    return sum(same) / len(same)


def ter(hyps: List[Union[str, List[str]]], refs: List[Union[str, List[str]]]) -> float:
    """Token error rate calculator.

    Args:
        hyps (List[Union[str, List[str]]]): List of hypotheses.
        refs (List[Union[str, List[str]]]): List of references.

    Returns:
        float: Averaged token error rate overall utterances.
    """
    error_tokens = 0
    total_tokens = 0
    for h, r in zip(hyps, refs):
        error_tokens += ed.eval(h, r)
        total_tokens += len(r)
    return float(error_tokens) / float(total_tokens)


def wer(hyps: List[str], refs: List[str]) -> float:
    """Word error rate calculator.

    Args:
        hyps (List[str]): List of hypotheses.
        refs (List[str]): List of references.

    Returns:
        float: Averaged word error rate overall utterances.
    """
    hyps = [h.split(" ") for h in hyps]
    refs = [r.split(" ") for r in refs]
    return ter(hyps, refs)


def per(hyps: List[str], refs: List[str]) -> float:
    """Phoneme error rate calculator.

    Args:
        hyps (List[str]): List of hypotheses.
        refs (List[str]): List of references.

    Returns:
        float: Averaged phoneme error rate overall utterances.
    """
    return wer(hyps, refs)


def cer(hyps: List[str], refs: List[str]) -> float:
    """Character error rate calculator.

    Args:
        hyps (List[str]): List of hypotheses.
        refs (List[str]): List of references.

    Returns:
        float: Averaged character error rate overall utterances.
    """
    return ter(hyps, refs)


def compute_eer(labels: List[int], scores: List[float]):
    """Compute equal error rate.

    Args:
        scores (List[float]): List of hypotheses.
        labels (List[int]): List of references.

    Returns:
        eer (float): Equal error rate.
        treshold (float): The treshold to accept a target trial.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    threshold = interp1d(fpr, thresholds)(eer)
    return eer, threshold


def compute_minDCF(
        labels: List[int],
        scores: List[float],
        p_target: float = 0.01,
        c_miss: int = 1,
        c_fa: int = 1):
    """Compute MinDCF.
    Computes the minimum of the detection cost function.  The comments refer to
    equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.

    Args:
        scores (List[float]): List of hypotheses.
        labels (List[int]): List of references.
        p (float): The prior probability of positive class.
        c_miss (int): The cost of miss.
        c_fa (int): The cost of false alarm.

    Returns:
        min_dcf (float): The calculated min_dcf.
        min_c_det_threshold (float): The treshold to calculate min_dcf.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnr)):
        c_det = c_miss * fnr[i] * p_target + c_fa * fpr[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold


def clean(ref: str) -> str:
    ref = re.sub(r"B\-(\S+) ", "", ref)
    ref = re.sub(r" E\-(\S+)", "", ref)
    return ref


def parse(hyp: str, ref: str) -> Tuple[str, str, str, str]:
    gex = re.compile(r"B\-(\S+) (.+?) E\-\1")

    hyp = re.sub(r" +", " ", hyp)
    ref = re.sub(r" +", " ", ref)

    hyp_slots = gex.findall(hyp)
    ref_slots = gex.findall(ref)

    ref_slots = ";".join([":".join([x[1], x[0]]) for x in ref_slots])
    if len(hyp_slots) > 0:
        hyp_slots = ";".join([":".join([clean(x[1]), x[0]]) for x in hyp_slots])
    else:
        hyp_slots = ""

    ref = clean(ref)
    hyp = clean(hyp)

    return ref, hyp, ref_slots, hyp_slots


def get_slot_dict(
        pred_slot,
        pred_transcript,
        label_slot,
        label_transcript) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    hyp_dict, ref_dict = {}, {}

    for slot_tok, transcript_tok in zip(
            pred_slot.split(), pred_transcript.split()):
        hyp_dict.setdefault(slot_tok, [])
        hyp_dict[slot_tok].append(transcript_tok)

    for slot_tok, transcript_tok in zip(
            label_slot.split(), label_transcript.split()):
        ref_dict.setdefault(slot_tok, [])
        ref_dict[slot_tok].append(transcript_tok)

    return ref_dict, hyp_dict


def slot_type_f1(
        slots_pred_list,
        transcript_pred_list,
        slots_label_list,
        transcript_label_list) -> float:
    F1s = []

    for p_slot, p_trans, t_slot, t_trans in zip(
            slots_pred_list,
            transcript_pred_list,
            slots_label_list,
            transcript_label_list):
        ref_dict, hyp_dict = get_slot_dict(p_slot, p_trans, t_slot, t_trans)

        if len(hyp_dict.keys()) == 0 and len(ref_dict.keys()) == 0:
            F1 = 1.0
        elif len(hyp_dict.keys()) == 0:
            F1 = 0.0
        elif len(ref_dict.keys()) == 0:
            F1 = 0.0
        else:
            P, R = 0.0, 0.0
            for slot in ref_dict:
                if slot in hyp_dict:
                    R += 1
            R = R / len(ref_dict.keys())
            for slot in hyp_dict:
                if slot in ref_dict:
                    P += 1
            P = P / len(hyp_dict.keys())
            F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
        F1s.append(F1)

    return sum(F1s) / len(F1s)


def slot_value_cer(
        slots_pred_list,
        transcript_pred_list,
        slots_label_list,
        transcript_label_list) -> float:
    value_hyps, value_refs = [], []

    for p_slot, p_trans, t_slot, t_trans in zip(
            slots_pred_list,
            transcript_pred_list,
            slots_label_list,
            transcript_label_list):
        ref_dict, hyp_dict = get_slot_dict(p_slot, p_trans, t_slot, t_trans)

        # Slot Value WER/CER evaluation
        unique_slots = list(ref_dict.keys())
        for slot in unique_slots:
            for ref_i, ref_v in enumerate(ref_dict[slot]):
                if slot not in hyp_dict:
                    hyp_v = ""
                    value_refs.append(ref_v)
                    value_hyps.append(hyp_v)
                else:
                    min_cer = 100
                    best_hyp_v = ""
                    for hyp_v in hyp_dict[slot]:
                        tmp_cer = cer([hyp_v], [ref_v])
                        if min_cer > tmp_cer:
                            min_cer = tmp_cer
                            best_hyp_v = hyp_v
                    value_refs.append(ref_v)
                    value_hyps.append(best_hyp_v)

    return cer(value_hyps, value_refs)


def slot_value_wer(hypothesis: List[str], groundtruth: List[str], **kwargs) -> float:
    value_hyps = []
    value_refs = []
    for p, t in zip(hypothesis, groundtruth):
        ref_dict, hyp_dict = get_slot_dict(p, t)

        # Slot Value WER/CER evaluation
        unique_slots = list(ref_dict.keys())
        for slot in unique_slots:
            for ref_i, ref_v in enumerate(ref_dict[slot]):
                if slot not in hyp_dict:
                    hyp_v = ""
                    value_refs.append(ref_v)
                    value_hyps.append(hyp_v)
                else:
                    min_wer = 100
                    best_hyp_v = ""
                    for hyp_v in hyp_dict[slot]:
                        tmp_wer = wer([hyp_v], [ref_v])
                        if min_wer > tmp_wer:
                            min_wer = tmp_wer
                            best_hyp_v = hyp_v
                    value_refs.append(ref_v)
                    value_hyps.append(best_hyp_v)

    return wer(value_hyps, value_refs)


def slot_edit_f1(
        hypothesis: List[str],
        groundtruth: List[str],
        loop_over_all_slot: bool,
        **kwargs) -> float:
    slot2F1 = {}  # defaultdict(lambda: [0,0,0]) # TPs, FNs, FPs
    for p, t in zip(hypothesis, groundtruth):
        ref_dict, hyp_dict = get_slot_dict(p, t)

        # Collecting unique slots
        unique_slots = list(ref_dict.keys())
        if loop_over_all_slot:
            unique_slots += [x for x in hyp_dict if x not in ref_dict]
        # Evaluating slot edit F1
        for slot in unique_slots:
            TP = 0
            FP = 0
            FN = 0
            # this never happens in list(ref_dict.keys())
            if slot not in ref_dict:
                for hyp_v in hyp_dict[slot]:
                    FP += 1
            else:
                for ref_i, ref_v in enumerate(ref_dict[slot]):
                    if slot not in hyp_dict:
                        FN += 1
                    else:
                        match = False
                        for hyp_v in hyp_dict[slot]:
                            # if ref_i < len(hyp_dict[slot]):
                            #    hyp_v = hyp_dict[slot][ref_i]
                            if hyp_v == ref_v:
                                match = True
                                break
                        if match:
                            TP += 1
                        else:
                            FN += 1
                            FP += 1
            slot2F1.setdefault(slot, [0, 0, 0])
            slot2F1[slot][0] += TP
            slot2F1[slot][1] += FN
            slot2F1[slot][2] += FP

    all_TPs, all_FNs, all_FPs = 0, 0, 0
    for slot in slot2F1.keys():
        all_TPs += slot2F1[slot][0]
        all_FNs += slot2F1[slot][1]
        all_FPs += slot2F1[slot][2]

    return 2 * all_TPs / (2 * all_TPs + all_FPs + all_FNs)


def slot_edit_f1_full(hypothesis: List[str], groundtruth: List[str], **kwargs) -> float:
    return slot_edit_f1(
        hypothesis, groundtruth, loop_over_all_slot=True, **kwargs)


def slot_edit_f1_part(hypothesis: List[str], groundtruth: List[str], **kwargs) -> float:
    return slot_edit_f1(
        hypothesis, groundtruth, loop_over_all_slot=False, **kwargs)
