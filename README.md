# Speech-MASSIVE: A Multilingual Speech Dataset for SLU and Beyond



## Introduciton

Speech-MASSIVE is a multilingual Spoken Language Understanding (SLU) dataset comprising the speech counterpart for a portion of the [MASSIVE](https://arxiv.org/abs/2204.08582) textual corpus. Speech-MASSIVE covers 12 languages (Arabic, German, Spanish, French, Hungarian, Korean, Dutch, Polish, European Portuguese, Russian, Turkish, and Vietnamese) from different families and inherits from MASSIVE the annotations for the intent prediction and slot-filling tasks. MASSIVE utterances' labels span 18 domains, with 60 intents and 55 slots. Full train split is provided for French and German, and for all the 12 languages (including French and German), we provide few-shot train, dev, test splits. Few-shot train (115 examples) covers all 18 domains, 60 intents, and 55 slots (including empty slots).

Our extension is prompted by the scarcity of massively multilingual SLU datasets and the growing need for versatile speech datasets to assess foundation models (LLMs, speech encoders) across diverse languages and tasks. To facilitate speech technology advancements, we release Speech-MASSIVE publicly available with [CC-BY-SA-4.0 license](https://creativecommons.org/licenses/by-sa/4.0/deed.en).

Speech-MASSIVE is accepted at INTERSPEECH 2024 (Kos, GREECE).

## Speech-MASSIVE data statistics

| lang | split      | # sample | # hrs | total # spk </br>(Male/Female/Unidentified) |
|:---:|:---:|:---:|:---:|:---:|
| ar-SA | few-shot train | 115 | 0.14 | 8 (4/4/0) |
| | dev | 2033 | 2.12 | 36 (22/14/0) |
| | test | 2974 | 3.23 | 37 (15/17/5) |
| de-DE | train-full | 11514 | 12.61 | 117 (50/63/4) |
| | few-shot train | 115 | 0.15 | 7 (3/4/0) |
| | dev | 2033 | 2.33 | 68 (35/32/1) |
| | test | 2974 | 3.41 | 82 (36/36/10) |
| es-ES | few-shot train | 115 | 0.13 | 7 (3/4/0) |
| | dev | 2033 | 2.53  | 109 (51/53/5) |
| | test | 2974 | 3.61  | 85 (37/33/15) |
| fr-FR | train-full | 11514 | 12.42 | 103 (50/52/1) |
| | few-shot train | 115 | 0.12 | 103 (50/52/1) |
| | dev | 2033 | 2.20 | 55 (26/26/3) |
| | test | 2974 | 2.65 | 75 (31/35/9) |
| hu-HU | few-shot train | 115 | 0.12 | 8 (3/4/1) |
| | dev | 2033 | 2.27 | 69 (33/33/3) |
| | test | 2974 | 3.30 | 55 (25/24/6) |
| ko-KR | few-shot train | 115 | 0.14 | 8 (4/4/0) |
| | dev | 2033 | 2.12 | 21 (8/13/0) |
| | test | 2974 | 2.66 | 31 (10/18/3) |
| nl-NL | few-shot train | 115 | 0.12 | 7 (3/4/0) |
| | dev | 2033 | 2.14 | 37 (17/19/1) |
| | test | 2974 | 3.30 | 100 (48/49/3) |
| pl-PL | few-shot train | 115 | 0.10 | 7 (3/4/0) |
| | dev | 2033 | 2.24 | 105 (50/52/3) |
| | test | 2974 | 3.21 | 151 (73/71/7) |
| pt-PT | few-shot train | 115 | 0.12 | 8 (4/4/0) |
| | dev | 2033 | 2.20 | 107 (51/53/3) |
| | test | 2974 | 3.25 | 102 (48/50/4) |
| ru-RU | few-shot train | 115 | 0.12 | 7 (3/4/0) |
| | dev | 2033 | 2.25 | 40 (7/31/2) |
| | test | 2974 | 3.44 | 51 (25/23/3) |
| tr-TR | few-shot train | 115 | 0.11 | 6 (3/3/0) |
| | dev | 2033 | 2.17 | 71 (36/34/1) |
| | test | 2974 | 3.00 | 42 (17/18/7) |
| vi-VN | few-shot train | 115 | 0.11 | 7 (2/4/1) |
| | dev | 2033 | 2.10 | 28 (13/14/1) |
| | test | 2974 | 3.23 | 30 (11/14/5) |




## Quick Links
- Speech-MASSIVE paper (link to be added)
- Speech-MASSIVE dataset on HuggingFace (link to be added)
- Speech-MASSIVE dataset on Zenodo (link to be added)
- E2E SLU models trained with Speech-MASSIVE on HuggingFace (link to be added)



## Licenses
Speech-MASSIVE dataset is released with CC-BY-SA-4.0 license.

All the codes in this repository are released with Apache License 2.0.

See `LICENSE.txt`, `NOTICE.md`, and `THIRD-PARTY.md`. (To be added)

## Citation

We ask that you cite our Speech-MASSIVE paper (link to be added) and both of the [MASSIVE paper](https://arxiv.org/abs/2204.08582) given that Speech-MASSIVE used text data from MASSIVE as seed data.

Speech-MASSIVE paper:
```
citation to be added
```

MASSIVE paper:
```
@misc{fitzgerald2022massive,
      title={MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages}, 
      author={Jack FitzGerald and Christopher Hench and Charith Peris and Scott Mackie and Kay Rottmann and Ana Sanchez and Aaron Nash and Liam Urbach and Vishesh Kakarala and Richa Singh and Swetha Ranganath and Laurie Crist and Misha Britan and Wouter Leeuwis and Gokhan Tur and Prem Natarajan},
      year={2022},
      eprint={2204.08582},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```