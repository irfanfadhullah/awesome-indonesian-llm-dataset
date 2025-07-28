# 🗣️ Local Languages

This section covers datasets and resources for Indonesian local languages including Javanese, Sundanese, Minangkabau, and other regional languages spoken across Indonesia.

## NusaX-MT: Multilingual Translation Dataset

- **Languages**: 12 languages including Indonesian + 11 local languages
- **Local Languages**: Acehnese, Balinese, Banjarese, Buginese, Madurese, Minangkabau, Javanese, Ngaju, Sundanese, Toba Batak, Ambon
- **Size**: Parallel corpus across all language pairs
- **Links**: [GitHub Repository](https://github.com/IndoNLP/nusax/tree/main/datasets/mt) | [Paper](https://arxiv.org/abs/2205.15960)

**Supported Tasks:**
- Machine translation for Indonesian local languages
- Cross-lingual evaluation
- Multilingual NLP research

## Regional Language Resources

### MinangNLP
- **Description**: Comprehensive Minangkabau language resources
- **Content**:
  - Bilingual dictionary: 11,905 Minangkabau-Indonesian word pairs
  - Sentiment analysis: 5,000 parallel texts (1,481 positive, 3,519 negative)
  - Machine translation: 16,371 sentence pairs
- **Paper**: [PACLIC 2020](https://www.aclweb.org/anthology/2020.paclic-1.17.pdf)

### MadureseSet
- **Size**: 17,809 basic + 53,722 substitution lemmata
- **Content**: Complete Madurese-Indonesian dictionary with pronunciation, POS, synonyms
- **Links**: [Mendeley Dataset](https://data.mendeley.com/datasets/nvc3rsf53b/5)
- **Paper**: [Data in Brief 2023](https://doi.org/10.1016/j.dib.2023.109035)

### Javanese Resources
- **Translated Dataset**: [HuggingFace](https://huggingface.co/datasets/ravialdy/javanese-translated)
- **Javanese Corpus**: [Kaggle](https://www.kaggle.com/datasets/hakikidamana/javanese-corpus/data)
- **Large Javanese ASR**: [OpenSLR](https://openslr.org/35/) - 52,000+ utterances
- **VoxLingua107**: [Javanese Speech](https://cs.taltech.ee/staff/tanel.alumae/data/voxlingua107/jw.zip)

### Sundanese Resources
- **AMSunda Dataset**: [Zenodo](https://zenodo.org/records/15494944) - First Sundanese information retrieval dataset
- **SU-CSQA**: [HuggingFace](https://huggingface.co/datasets/rifkiaputri/su-csqa) - Sundanese CommonsenseQA
- **Large Sundanese ASR**: [OpenSLR](https://openslr.org/36/)
- **VoxLingua107**: [Sundanese Speech](https://cs.taltech.ee/staff/tanel.alumae/data/voxlingua107/su.zip)

## NusaWrites Benchmark

- **NusaTranslation**: 72,444 texts across 11 local languages
- **NusaParagraph**: 57,409 paragraphs across 10 local languages
- **Tasks**: Sentiment analysis, emotion classification, machine translation, topic modeling, rhetoric mode classification
- **Links**: [GitHub Repository](https://github.com/IndoNLP/nusa-writes) | [Paper](https://aclanthology.org/2023.ijcnlp-main.60/)

**HuggingFace Access:**
```


# NusaTranslation datasets

datasets.load_dataset('indonlp/nusatranslation_emot')
datasets.load_dataset('indonlp/nusatranslation_senti')
datasets.load_dataset('indonlp/nusatranslation_mt')

# NusaParagraph datasets

datasets.load_dataset('indonlp/nusaparagraph_emot')
datasets.load_dataset('indonlp/nusaparagraph_rhetoric')
datasets.load_dataset('indonlp/nusaparagraph_topic')

```

## Indonesian Speech with Regional Accents

- **Coverage**: 5 ethnic groups (Batak, Malay, Javanese, Sundanese, Papuan)
- **Content**: 320-word standardized Indonesian text per speaker
- **Links**: [Kaggle Dataset](https://www.kaggle.com/datasets/hengkymulyono/indonesian-speech-with-accents-5-ethnic-groups)

## Story Cloze in Local Languages

- **Languages**: Javanese and Sundanese
- **Description**: Culturally nuanced commonsense reasoning dataset
- **Links**: [HuggingFace](https://huggingface.co/datasets/rifoag/javanese_sundanese_story_cloze) | [Paper](https://arxiv.org/abs/2502.12932)

## Citations

```

@misc{winata2022nusax,
title={NusaX: Multilingual Parallel Sentiment Dataset for 10 Indonesian Local Languages},
author={Winata, Genta Indra and Aji, Alham Fikri and Cahyawijaya, Samuel and others},
year={2022},
eprint={2205.15960},
archivePrefix={arXiv},
primaryClass={cs.CL}
}

@inproceedings{koto-koto-2020-towards,
title = "Towards Computational Linguistics in Minangkabau Language: Studies on Sentiment Analysis and Machine Translation",
author = "Koto, Fajri and Koto, Ikhwan",
booktitle = "Proceedings of the 34th Pacific Asia Conference on Language, Information and Computation",
year = "2020"
}

@article{ifada2023madureseset,
title={MadureseSet: Madurese-Indonesian Dataset},
author={Ifada, N. and Rachman, F.H. and others},
journal={Data in Brief},
volume={48},
pages={109035},
year={2023}
}

```