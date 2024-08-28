# Speech-MASSIVE: A Multilingual Speech Dataset for SLU and Beyond



## Introduciton

Speech-MASSIVE is a multilingual Spoken Language Understanding (SLU) dataset comprising the speech counterpart for a portion of the [MASSIVE](https://arxiv.org/abs/2204.08582) textual corpus. Speech-MASSIVE covers 12 languages (Arabic, German, Spanish, French, Hungarian, Korean, Dutch, Polish, European Portuguese, Russian, Turkish, and Vietnamese) from different families and inherits from MASSIVE the annotations for the intent prediction and slot-filling tasks. MASSIVE utterances' labels span 18 domains, with 60 intents and 55 slots. Full train split is provided for French and German, and for all the 12 languages (including French and German), we provide few-shot train, dev, test splits. Few-shot train (115 examples) covers all 18 domains, 60 intents, and 55 slots (including empty slots).

Our extension is prompted by the scarcity of massively multilingual SLU datasets and the growing need for versatile speech datasets to assess foundation models (LLMs, speech encoders) across diverse languages and tasks. To facilitate speech technology advancements, we release Speech-MASSIVE publicly available with [CC-BY-NC-SA-4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en).

Speech-MASSIVE is accepted at INTERSPEECH 2024 (Kos, GREECE).

## Speech-MASSIVE data statistics

| lang | split      | # sample | # hrs | total # spk </br>(Male/Female/Unidentified) |
|:---:|:---:|:---:|:---:|:---:|
| ar-SA | train_115 | 115 | 0.14 | 8 (4/4/0) |
| | validation | 2033 | 2.12 | 36 (22/14/0) |
| | test | 2974 | 3.23 | 37 (15/17/5) |
| de-DE | train | 11514 | 12.61 | 117 (50/63/4) |
| | train_115 | 115 | 0.15 | 7 (3/4/0) |
| | validation | 2033 | 2.33 | 68 (35/32/1) |
| | test | 2974 | 3.41 | 82 (36/36/10) |
| es-ES | train_115 | 115 | 0.13 | 7 (3/4/0) |
| | validation | 2033 | 2.53  | 109 (51/53/5) |
| | test | 2974 | 3.61  | 85 (37/33/15) |
| fr-FR | train | 11514 | 12.42 | 103 (50/52/1) |
| | train_115 | 115 | 0.12 | 103 (50/52/1) |
| | validation | 2033 | 2.20 | 55 (26/26/3) |
| | test | 2974 | 2.65 | 75 (31/35/9) |
| hu-HU | train_115 | 115 | 0.12 | 8 (3/4/1) |
| | validation | 2033 | 2.27 | 69 (33/33/3) |
| | test | 2974 | 3.30 | 55 (25/24/6) |
| ko-KR | train_115 | 115 | 0.14 | 8 (4/4/0) |
| | validation | 2033 | 2.12 | 21 (8/13/0) |
| | test | 2974 | 2.66 | 31 (10/18/3) |
| nl-NL | train_115 | 115 | 0.12 | 7 (3/4/0) |
| | validation | 2033 | 2.14 | 37 (17/19/1) |
| | test | 2974 | 3.30 | 100 (48/49/3) |
| pl-PL | train_115 | 115 | 0.10 | 7 (3/4/0) |
| | validation | 2033 | 2.24 | 105 (50/52/3) |
| | test | 2974 | 3.21 | 151 (73/71/7) |
| pt-PT | train_115 | 115 | 0.12 | 8 (4/4/0) |
| | validation | 2033 | 2.20 | 107 (51/53/3) |
| | test | 2974 | 3.25 | 102 (48/50/4) |
| ru-RU | train_115 | 115 | 0.12 | 7 (3/4/0) |
| | validation | 2033 | 2.25 | 40 (7/31/2) |
| | test | 2974 | 3.44 | 51 (25/23/3) |
| tr-TR | train_115 | 115 | 0.11 | 6 (3/3/0) |
| | validation | 2033 | 2.17 | 71 (36/34/1) |
| | test | 2974 | 3.00 | 42 (17/18/7) |
| vi-VN | train_115 | 115 | 0.11 | 7 (2/4/1) |
| | validation | 2033 | 2.10 | 28 (13/14/1) |
| | test | 2974 | 3.23 | 30 (11/14/5) |


## Primary environments
All the provided codes are tested with below environments. Supports for different versions are not guraranteed.
- Python 3.9.4
- HuggingFace 4.37


## Training an End-to-end SLU model using Speech-MASSIVE dataset with HuggingFace

> [!CAUTION]
> Below examples are provided to work on HuggingFace

1. Clone this github to desired path
```
$ git clone https://github.com/hlt-mt/Speech-MASSIVE /your/speech-massive-code/path
```

2. Set up your own virtual envirnoment (venv, conda and etc) and activate it
```
$ python -m venv /path/to/new/virtual/environment

or

$ conda create --name <my-env>
```

3. First upgrade pip to support pyproject.toml
```
(venv) $ pip install --upgrade pip
```

4. Install Speech-MASSIVE python project
```
(venv) $ pip install -e .
```

5. Check hyper parameter files and modify them accordingly
```
    .
    └── src/speech_massive/examples/hparams
        ├── e2e_slu_zeroshot_fr.yaml    # Hparamer for zero-shot (French train data) setting
        └── e2e_slu_fewshot_fr.yaml     # Hparamer for few-shot (French train data + all langs train_115 data) setting
```

6. Run training
```
# zero-shot training

$ python ${your_path}/src/speech_massive/examples/speech/run_slu_whisper.py {your_path}/src/speech_massive/examples/hparams/e2e_slu_zeroshot_fr.yaml

# few-shot training

$ python ${your_path}/src/speech_massive/examples/speech/run_slu_whisper.py {your_path}/src/speech_massive/examples/hparams/e2e_slu_fewshot_fr.yaml
```


## Quick Links
- [Speech-MASSIVE paper (arXiv)](https://arxiv.org/abs/2408.03900)
- [Speech-MASSIVE train/validation set on HuggingFace](https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE)
- [Speech-MASSIVE test set on HuggingFace](https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE-test)
- [Speech-MASSIVE train/validation/test set zip file directly downloadable](https://mt.fbk.eu/speech-massive)

## Licenses
Speech-MASSIVE dataset is released with CC-BY-NC-SA-4.0 license.

All the codes in this repository are released with Apache License 2.0.

See `LICENSE` and `NOTICE.md`.

## Citation

We ask that you cite our [Speech-MASSIVE paper on arXiv](https://arxiv.org/abs/2408.03900) and also the [MASSIVE paper](https://arxiv.org/abs/2204.08582) given that Speech-MASSIVE used text data from MASSIVE as seed data.



Speech-MASSIVE paper (accepted at INTERSPEECH 2024):
```
@inproceedings{lee2024speechmassivemultilingualspeechdataset,
      title={{Speech-MASSIVE: A Multilingual Speech Dataset for SLU and Beyond}}, 
      author={Beomseok Lee and Ioan Calapodescu and Marco Gaido and Matteo Negri and Laurent Besacier},
      year={2024},
       booktitle={Proc. Interspeech 2024},
}
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