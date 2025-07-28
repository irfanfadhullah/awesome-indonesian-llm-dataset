# 🧠 Natural Language Understanding

This section covers datasets and resources for natural language understanding tasks in Indonesian, including natural language inference, sentiment analysis, text classification, and other NLU tasks.

## IndoNLI: Natural Language Inference Dataset

- **Description**: Human-annotated NLI data for Indonesian with train/val/test splits
- **Size**: Thousands of premise-hypothesis pairs
- **Tasks**: Natural language inference, textual entailment
- **Links**: [GitHub Repository](https://github.com/ir-nlp-csui/indonli) | [HuggingFace](https://huggingface.co/datasets/indonli)
- **Paper**: EMNLP 2021

**Key Features:**
- Expert and lay annotator splits
- Diagnostic subset with linguistic phenomena
- Translated MNLI dataset included

## Indonesian NLP Resources

Comprehensive collection of datasets for various NLP tasks:

- **Repository**: [GitHub](https://github.com/kmkurn/id-nlp-resource)
- **POS Tagging**: [IDN tagged corpus](https://github.com/famrashel/idn-tagged-corpus) (10K sentences, 250K tokens)
- **Sentiment Analysis**:
  - [Hotel Reviews ABSA](https://github.com/jordhy97/final_project) (5K reviews, 78K tokens)
  - [Aspect-Based Sentiment Analysis](https://github.com/annisanurulazhar/absa-playground)
  - [Mongabay](https://huggingface.co/datasets/Datasaur/Mongabay-collection)
- **Text Classification**:
  - [SMS Spam Detection](https://drive.google.com/file/d/1-stKadfTgJLtYsHWqXhGO3nTjKVFxm_Q/view) (1,143 sentences)
  - [Hate Speech Detection](https://github.com/ialfina/id-hatespeech-detection) (713 tweets)
  - [Abusive Language Detection](https://github.com/okkyibrohim/id-abusive-language-detection)
  - [HoASA](https://huggingface.co/datasets/SEACrowd/hoasa)
  - [Indonesian Clickbait Headlines](https://www.kaggle.com/datasets/andikawilliam/clickid)
  - [CASA](https://huggingface.co/datasets/SEACrowd/casa)
  - [SmSA](https://huggingface.co/datasets/SEACrowd/smsa)
  - [WReTE](https://github.com/IndoNLP/indonlu/tree/master/dataset/wrete_entailment-ui)
  - [EmoT](https://huggingface.co/datasets/SEACrowd/emot)
  - [SentiWS](https://www.kaggle.com/datasets/rtatman/sentiment-lexicons-for-81-languages)

## Hate Speech & Abusive Language Detection

- **ID Multi-label Hate Speech**
  - **Description**: 13,169 Indonesian tweets labeled for multi-label hate speech and abusive language
  - **Labels**: Multiple categories of hate speech and abusive content
  - **Links**: [GitHub Repository](https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection)
  - **Paper**: [ACL 2019](https://www.aclweb.org/anthology/W19-3506.pdf)

- **ID Abusive Language Detection**
  - **Description**: 2,016 Indonesian tweets labeled as abusive/not offensive/offensive
  - **Links**: [GitHub Repository](https://github.com/okkyibrohim/id-abusive-language-detection)
  - **Paper**: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1877050918314583)

- **ID Hate Speech Detection**
  - **Description**: 713 Indonesian tweets labeled hate speech/non-hate speech
  - **Links**: [GitHub Repository](https://github.com/ialfina/id-hatespeech-detection)
  - **Paper**: [IEEE](https://ieeexplore.ieee.org/abstract/document/8355039)

## Translated Datasets

- **OpenWebText-10K**: [Download](https://huggingface.co/datasets/irfanfadhullah/OpenWebText-Indonesia-10k/)
- **FineWeb-Edu-25K**: [Download](https://huggingface.co/datasets/irfanfadhullah/FineWeb-Edu-25K/)

## QQPR-Triplets-ID

- **Size**: 86K
- **Link**: [Download](https://huggingface.co/datasets/robinsyihab/QQPR-triplets-ID)

## Citations

```

@inproceedings{mahendra-etal-2021-indonli,
title = "{I}ndo{NLI}: A Natural Language Inference Dataset for {I}ndonesian",
author = "Mahendra, Rahmad and Aji, Alham Fikri and Louvan, Samuel and Rahman, Fahrurrozi and Vania, Clara",
booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
month = nov,
year = "2021",
address = "Online and Punta Cana, Dominican Republic",
publisher = "Association for Computational Linguistics",
url = "https://aclanthology.org/2021.emnlp-main.821",
pages = "10511--10527",
}

@inproceedings{okkyibrohim2019,
title = "Multi-label Hate Speech and Abusive Language Detection in Indonesian Twitter",
author = "Ibrohim, Okky and Budi, Indra",
booktitle = "Proceedings of the Third Workshop on Abusive Language Online",
year = "2019",
publisher = "Association for Computational Linguistics",
url = "https://www.aclweb.org/anthology/W19-3506.pdf",
}

```