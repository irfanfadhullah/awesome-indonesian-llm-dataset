## MinangNLP

This repository contains three data:
1. Bilingual dictionary: 11,905-size Minangkabau–Indonesian word pairs. (in `resources/kamus_minang_id.dic`)
2. Sentiment Analysis: 5,000-size (1,481 positive and 3,519 negative labels) parallel Minangkabau-Indonesian texts. (in `sentiment/data/folds/`)
3. Machine translation: 16,371-size parallel Minangkabau-Indonesian sentence pairs. (in `translation/wiki_data/`)

Please cite our works if you use our corpus:

 Fajri Koto, and Ikhwan Koto. [_Towards Computational Linguistics in Minangkabau Language: Studies on Sentiment Analysis and Machine Translation_](https://www.aclweb.org/anthology/2020.paclic-1.17.pdf).  In Proceedings of the 34th Pacific Asia Conference on Language, Information and Computation (PACLIC), Vietnam, October, 2020.

## NusaX-MT Dataset

### Table of Contents
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

### Dataset Description

- **Repository:** [GitHub](https://github.com/IndoNLP/nusax/tree/main/datasets/mt)
- **Paper:** [EACL 2022](https://arxiv.org/abs/2205.15960)
- **Point of Contact:** [GitHub](https://github.com/IndoNLP/nusax/tree/main/datasets/mt)

#### Dataset Summary

NusaX is a high-quality multilingual parallel corpus that covers 12 languages, Indonesian, English, and 10 Indonesian local languages, namely Acehnese, Balinese, Banjarese, Buginese, Madurese, Minangkabau, Javanese, Ngaju, Sundanese, and Toba Batak.
NusaX-MT is a parallel corpus for training and benchmarking machine translation models across 10 Indonesian local languages + Indonesian and English. The data is presented in csv format with 12 columns, one column for each language.


#### Supported Tasks and Leaderboards

- Machine translation for Indonesian languages

#### Languages

All possible pairs of the following:

- ace: acehnese,
- ban: balinese,
- bjn: banjarese,
- bug: buginese,
- eng: english,
- ind: indonesian,
- jav: javanese,
- mad: madurese,
- min: minangkabau,
- nij: ngaju,
- sun: sundanese,
- bbc: toba_batak,
### Dataset Creation
#### Curation Rationale
There is a shortage of NLP research and resources for the Indonesian languages, despite the country having over 700 languages. With this in mind, we have created this dataset to support future research for the underrepresented languages in Indonesia.
#### Source Data
##### Initial Data Collection and Normalization
NusaX-MT is a dataset for machine translation in Indonesian langauges that has been expertly translated by native speakers.
##### Who are the source language producers?
The data was produced by humans (native speakers).
#### Annotations
##### Annotation process
NusaX-MT is derived from SmSA, which is the biggest publicly available dataset for Indonesian sentiment analysis. It comprises of comments and reviews from multiple online platforms. To ensure the quality of our dataset, we have filtered it by removing any abusive language and personally identifying information by manually reviewing all sentences. To ensure balance in the label distribution, we randomly picked 1,000 samples through stratified sampling and then translated them to the corresponding languages.
##### Who are the annotators?
Native speakers of both Indonesian and the corresponding languages.
Annotators were compensated based on the number of translated samples.
#### Personal and Sensitive Information
Personal information is removed.
### Considerations for Using the Data
#### Social Impact of Dataset
[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)
#### Discussion of Biases
NusaX is created from review text. These data sources may contain some bias.
#### Other Known Limitations
No other known limitations
### Additional Information
#### Licensing Information
CC-BY-SA 4.0.
Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
ShareAlike — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.
No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.
Please contact authors for any information on the dataset.
#### Citation Information
```
@misc{winata2022nusax,
      title={NusaX: Multilingual Parallel Sentiment Dataset for 10 Indonesian Local Languages},
      author={Winata, Genta Indra and Aji, Alham Fikri and Cahyawijaya,
      Samuel and Mahendra, Rahmad and Koto, Fajri and Romadhony,
      Ade and Kurniawan, Kemal and Moeljadi, David and Prasojo,
      Radityo Eko and Fung, Pascale and Baldwin, Timothy and Lau,
      Jey Han and Sennrich, Rico and Ruder, Sebastian},
      year={2022},
      eprint={2205.15960},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
#### Contributions
Thanks to [@afaji](https://github.com/afaji) for adding this dataset.


## MadureseSet: Madurese-Indonesian Dataset

### Description
MadureseSet is a digitized version of the physical document of Kamus Lengkap Bahasa MaduraIndonesia (The Complete Dictionary of Madurese-Indonesian). It stores the list of lemmata in Madurese, i.e., 17809 basic lemmata and 53722 substitution lemmata, and their translation in Indonesian. The details of each lemma may include its pronunciation, part of speech, synonym and homonym relations, speech level, dialect, and loanword. The framework of dataset creation consists of three stages. First, the data extraction stage processes the physical document results to produce corrected data in a text file. Second, the data structural review stage processes the text file in terms of the paragraph, homonym, synonym, linguistic, poem, short poem, proverb, and metaphor structures to create the data structure that best represents the information in the dictionary. Finally, the database construction stage builds the physical data model and populates the MadureseSet database data. MadureseSet is validated by a Madurese language expert who is also the author of the physical document source of this dataset. Thus, this dataset can be a primary source for Natural Language Processing (NLP) research, especially for the Madurese language.  

### Original Link
https://data.mendeley.com/datasets/nvc3rsf53b/5


Please cite the following paper to acknowledge use of the dataset in publications: 
Ifada, N., Rachman, F.H., Syauqy, M.W.M.A., Wahyuni, S. and Pawitra, A., 2023. MadureseSet: Madurese-Indonesian Dataset. Data in Brief, 48, p.109035. DOI: https://doi.org/10.1016/j.dib.2023.109035

## Javansese Dataset

Translated version from another original dataset

### Link Download
https://huggingface.co/datasets/ravialdy/javanese-translated

## Nusa Writes

### Overview
**NusaWrites** is an in-depth analysis of corpora collection strategy and a comprehensive language modeling benchmark for underrepresented and extremely low-resource Indonesian local languages.

**NusaWrites** benchmark consists of two datasets over 6 different tasks: 
1. NusaTranslation 
	- NusaTranslation is a human translated local languages dataset consists of 72,444 textual data from multiple languages
	- NusaTranslation covers 11 local languages, including Ambon (abs), Batak (btk), Betawi (bew), Bima (bhp), Javanese (jav), Madurese (mad), Makassarese (mak), Minangkabau (min), Palembang / Musi (mui), Rejang (rej), and Sundanese (sun)
	- NusaTranslation support 3 downstream tasks: sentiment analysis, emotion classification, and machine translation
	
2. NusaParagraph
	- NusaParagraph is a human paragraph writing dataset consists of 57,409 paragraphs from multiple Indonesian local languages
	- NusaParagraph covers 10 local languages, including Batak (btk), Betawi (bew), Buginese (bug), Javanese (jav), Madurese (mad), Makassarese (mak), Minangkabau (min), Palembang / Musi (mui), Rejang (rej), and Sundanese (sun)
	- NusaParagraph supports 3 downstream tasks: emotion classification, topic modeling, and rhetoric mode classification

### How to Use

The complete **NusaWrites** dataset can be accessed from our [github repository](https://github.com/IndoNLP/nusa-writes). 
For a more easy-to-use and standardized access of all **NusaWrites** datasets, you can access it though the [Hugging Face `datasets` library]() or our [NusaCrowd library]()

##### Access from Hugging Face `datasets`
```
import datasets

# NusaTranslation (all Languages mixed)
nt_emot_dset = datasets.load_dataset('indonlp/nusatranslation_emot')
nt_senti_dset = datasets.load_dataset('indonlp/nusatranslation_senti')
nt_mt_dset = datasets.load_dataset('indonlp/nusatranslation_mt')

# NusaTranslation (per language)
# Supported lang_code: abs, btk, bew, bhp, jav, mad, mak, min, mui, rej, sun
nt_emot_dset = datasets.load_dataset('indonlp/nusatranslation_emot', name='nusatranslation_emot_{lang_code}_nusantara_text')
nt_senti_dset = datasets.load_dataset('indonlp/nusatranslation_senti', name='nusatranslation_senti_{lang_code}_nusantara_text')
nt_mt_dset = datasets.load_dataset('indonlp/nusatranslation_mt', name='nusatranslation_mt_{lang_code}_nusantara_text')

# NusaParagraph (all Languages mixed)
np_emot_dset = datasets.load_dataset('indonlp/nusaparagraph_emot')
np_rhetoric_dset = datasets.load_dataset('indonlp/nusaparagraph_rhetoric')
np_topic_dset = datasets.load_dataset('indonlp/nusaparagraph_topic')

# NusaParagraph (per language)
# Supported lang_code: btk, bew, bug, jav, mad, mak, min, mui, rej, sun
np_emot_dset = datasets.load_dataset('indonlp/nusaparagraph_emot', name='nusaparagraph_emot_{lang_code}_nusantara_text')
np_rhetoric_dset = datasets.load_dataset('indonlp/nusaparagraph_rhetoric', name='nusaparagraph_rhetoric_{lang_code}_nusantara_text')
np_topic_dset = datasets.load_dataset('indonlp/nusaparagraph_topic', name='nusaparagraph_topic_{lang_code}_nusantara_text')
```

##### Access from NusaCrowd

Loading per task dataset
```
import nusacrowd as nc

# NusaTranslation (all Languages mixed)
nt_emot_dset = nc.load_dataset('nusatranslation_emot')
nt_senti_dset = nc.load_dataset('nusatranslation_senti')
nt_mt_dset = nc.load_dataset('nusatranslation_mt')

# NusaTranslation (per language)
# Supported lang_code: abs, btk, bew, bhp, jav, mad, mak, min, mui, rej, sun
nt_emot_dset = nc.load_dataset('indonlp/nusatranslation_emot', name='nusatranslation_emot_{lang_code}_nusantara_text')
nt_senti_dset = nc.load_dataset('indonlp/nusatranslation_senti', name='nusatranslation_senti_{lang_code}_nusantara_text')
nt_mt_dset = nc.load_dataset('indonlp/nusatranslation_mt', name='nusatranslation_mt_{lang_code}_nusantara_text')

# NusaParagraph (all Languages mixed)
np_emot_dset = nc.load_dataset('indonlp/nusaparagraph_emot')
np_rhetoric_dset = nc.load_dataset('indonlp/nusaparagraph_rhetoric')
np_topic_dset = nc.load_dataset('indonlp/nusaparagraph_topic')

# NusaParagraph (per language)
# Supported lang_code: btk, bew, bug, jav, mad, mak, min, mui, rej, sun
np_emot_dset = nc.load_dataset('indonlp/nusaparagraph_emot', name='nusaparagraph_emot_{lang_code}_nusantara_text')
np_rhetoric_dset = nc.load_dataset('indonlp/nusaparagraph_rhetoric', name='nusaparagraph_rhetoric_{lang_code}_nusantara_text')
np_topic_dset = nc.load_dataset('indonlp/nusaparagraph_topic', name='nusaparagraph_topic_{lang_code}_nusantara_text')
```

Loading the whole benchmark
```
# NusaTranslation
nusa_translation_dsets = nc.load_benchmark('NusaTranslation')

# NusaParagraph
nusa_paragraph_dsets = nc.load_benchmark('NusaParagraph')

# NusaWrites
nusa_writes_dsets = nc.load_benchmark('NusaWrites')
```

### Experiment Code

##### Running LM Experiment

We modify the `run_clm.py` code from Hugging Face and made use of IndoGPT (https://huggingface.co/indobenchmark/indogpt) tokenizer in our LM experiment. 
The code and the run script can be found under the [lm-exp](https://github.com/IndoNLP/nusa-writes/tree/main/lm-exp) folder in the repository.
- `run_clm.py` → https://github.com/IndoNLP/nusa-writes/blob/main/lm-exp/run_clm.py
- Bash runner script (`run_lm_exp.sh`) → https://github.com/IndoNLP/nusa-writes/blob/main/lm-exp/run_lm_exp.sh

##### Running PBSMT Experiment

To run the PBSMT experiment, you can follow the run the code in the following order:
- Generate dataset → https://github.com/IndoNLP/nusa-writes/pbsmt/convert_data.py
- Generate config → https://github.com/IndoNLP/nusa-writes/pbsmt/generate_configs.py
- Training → https://github.com/IndoNLP/nusa-writes/blob/stif-indonesia/run_nusa_menulis_train.sh
- Testing → https://github.com/IndoNLP/nusa-writes/blob/stif-indonesia/run_nusa_menulis_eval.sh 


### Research Paper
Our work has been accepted in AACL 2023 and published [here](https://aclanthology.org/2023.ijcnlp-main.60/).

If you find our work helpful, please cite the following article:
```
@inproceedings{cahyawijaya-etal-2023-nusawrites,
    title = "{N}usa{W}rites: Constructing High-Quality Corpora for Underrepresented and Extremely Low-Resource Languages",
    author = "Cahyawijaya, Samuel  and  Lovenia, Holy  and Koto, Fajri  and  Adhista, Dea  and  Dave, Emmanuel  and  Oktavianti, Sarah  and  Akbar, Salsabil  and  Lee, Jhonson  and  Shadieq, Nuur  and  Cenggoro, Tjeng Wawan  and  Linuwih, Hanung  and  Wilie, Bryan  and  Muridan, Galih  and  Winata, Genta  and  Moeljadi, David  and  Aji, Alham Fikri  and  Purwarianti, Ayu  and  Fung, Pascale",
    editor = "Park, Jong C.  and  Arase, Yuki  and  Hu, Baotian  and  Lu, Wei  and  Wijaya, Derry  and  Purwarianti, Ayu  and  Krisnadhi, Adila Alfa",
    booktitle = "Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = nov,
    year = "2023",
    address = "Nusa Dua, Bali",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.ijcnlp-main.60",
    pages = "921--945",
}
```


## AMSunda: A Novel Dataset for Sundanese Information Retrieval

### Description
The AMSunda dataset was introduced as the first resource designed explicitly for fine-tuning and evaluating embedding models in the Sundanese language. AMSunda dataset consists of two dataset types: (1) triplet data containing a query passage, a positive, and a negative response aimed for fine-tuning embedding models, and (2) BEIR-compatible data structured for evaluating embedding models on retrieval tasks.

### Original Paper
https://www.sciencedirect.com/science/article/pii/S2352340925005232

### Link to the dataset
https://zenodo.org/records/15494944

### Citation
```
@dataset{maesya_2025_15494944,
  author       = {Maesya, Aries and
                  Arifin, Yulyani and
                  Budiharto, Widodo and
                  Amalia, Zahra},
  title        = {AMSunda: A Novel Dataset for Sundanese Information
                   Retrieval
                  },
  month        = may,
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.15494944},
  url          = {https://doi.org/10.5281/zenodo.15494944},
}
```

## Javanese Corpus

### Description
The Javanese Corpus was introduce for pretraining the LLM or machine translation.

### Link to the dataset
https://www.kaggle.com/datasets/hakikidamana/javanese-corpus/data


## Large Javanese ASR training data set

### Link to Download
https://openslr.org/35/

### Description
About this resource:

This data set contains transcribed audio data for Javanese. The data set consists of wave files, and a TSV file. The file utt_spk_text.tsv contains a FileID, UserID and the transcription of audio in the file.
The data set has been manually quality checked, but there might still be errors.

This dataset was collected by Google in collaboration with Reykjavik University and Universitas Gadjah Mada in Indonesia.

See LICENSE.txt file for license information.

Copyright 2016, 2017 Google, Inc.

### Citation
If you use this data in publications, please cite it as follows:
```
  @inproceedings{kjartansson-etal-sltu2018,
    title = {{Crowd-Sourced Speech Corpora for Javanese, Sundanese,  Sinhala, Nepali, and Bangladeshi Bengali}},
    author = {Oddur Kjartansson and Supheakmungkol Sarin and Knot Pipatsrisawat and Martin Jansche and Linne Ha},
    booktitle = {Proc. The 6th Intl. Workshop on Spoken Language Technologies for Under-Resourced Languages (SLTU)},
    year  = {2018},
    address = {Gurugram, India},
    month = aug,
    pages = {52--55},
    URL   = {http://dx.doi.org/10.21437/SLTU.2018-11},
  }
```

## Indonesian Speech with Accents (5 ethnic groups)

### Description
Primary data audio signal of Indonesian speech accent was taken by direct recording. The recording process was carried out by speakers aged 17 - 50 years from the Batak, Malay, Javanese, Sundanese, and Papuan ethnic groups. Each speaker uttered the same 320-word Indonesian text, recorded on their respective mobile phones.

Some additional secondary data was taken from several internet sources like YouTube, podcasts, etc.
(the MFCC files are derived only from primary data).

Here is the exact sentence script used for the primary data recordings:
https://www.kaggle.com/datasets/hengkymulyono/indonesian-speech-with-accents-5-ethnic-groups


## Dataset Card for SU-CSQA and ID
<!-- Provide a quick summary of the dataset. -->

### Dataset Description

<!-- Provide a longer summary of what this dataset is. -->

- **Repository:** [rifkiaputri/id-csqa](https://github.com/rifkiaputri/id-csqa)
- **Paper:** [Can LLM Generate Culturally Relevant Commonsense QA Data? Case Study in Indonesian and Sundanese](https://aclanthology.org/2024.emnlp-main.1145/)
- **Point of Contact:** [rifkiaputri](https://github.com/rifkiaputri)
- **License:** Creative Commons Non-Commercial (CC BY-NC 4.0)

In our [paper](https://arxiv.org/abs/2402.17302/), we investigate the effectiveness of using LLMs in generating culturally relevant CommonsenseQA datasets
for Indonesian and Sundanese languages. To do so, we create datasets using various methods: *(1) Automatic Data Adaptation*, *(2) Manual Data Generation*, and 
*(3) Automatic Data Generation*.

![Data Construction](data_generation_method_v3.jpg)

**Note: This repository contains the *Sundanese* subset of the data. The *Indonesian* version can be found [here](https://huggingface.co/datasets/rifkiaputri/id-csqa).**

### Dataset Structure
Based on the dataset generation methods, we have three data variation:

1. `LLM_Adapt`: LLM-generated* dataset constructed through automatic data adaptation method.
2. `Human_Gen`: human-generated dataset constructed through manual data generation method.
3. `LLM_Gen`: LLM-generated* dataset constructed through automatic data generation method.

_*\) Note: In this data version, we utilized GPT-4 Turbo (11-06) as the LLM._

Generally, each data item consists of a multiple-choice question with five options and one correct answer.

For `Human_Gen` dataset specifically, we provide one answer (`answer_majority`), which is based on the majority voting from: one answer from the question creator 
(`answer_creator`), and answers from other annotators (`answers`). We also provide more metadata related to the answers, such as `answers_uncertainty`, 
`questions_ambiguity`, `option_ambiguity` and `reason` (a freetext explanation in Indonesian language, for why the annotators marked the question or option as ambiguous).

For `LLM_Adapt` and `LLM_Gen` data, we also provide a subset that has been cleaned by humans, which can be found in the `test_clean` split.


### Language and Region Coverage
In terms of language coverage, we were only able to cover Indonesian and Sundanese due to the available resources and the authors’ familiarity with these languages. Additionally, the annotators we recruited
were mostly from Java island, with one annotator from Bali island. Despite our effort to include a range of question concepts from different regions, including those beyond Java and Bali islands, it is
possible that some bias may exist, especially in the Indonesian dataset. This is because the questions were generated primarily by annotators from Java
and Bali, and their perspectives and cultural backgrounds may have influenced the content. Nonetheless, we have taken measures to eliminate potentially harmful or stereotypical questions

### Citation
Please cite this paper if you use any dataset in this repository:
```
@inproceedings{putri-etal-2024-llm,
    title = "Can {LLM} Generate Culturally Relevant Commonsense {QA} Data? Case Study in {I}ndonesian and {S}undanese",
    author = "Putri, Rifki Afina  and
      Haznitrama, Faiz Ghifari  and
      Adhista, Dea  and
      Oh, Alice",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.1145",
    pages = "20571--20590",
}
```

## Javanese SUndanese Story Cloze

### Original Paper
https://arxiv.org/abs/2502.12932

### Linkt ot Download
https://huggingface.co/datasets/rifoag/javanese_sundanese_story_cloze

### Description
Quantifying reasoning capability in low-resource languages remains a challenge in NLP due to data scarcity and limited access to annotators. While LLM-assisted dataset construction has proven useful for medium- and high-resource languages, its effectiveness in low-resource languages, particularly for commonsense reasoning, is still unclear. In this paper, we compare three dataset creation strategies: (1) LLM-assisted dataset generation, (2) machine translation, and (3) human-written data by native speakers, to build a culturally nuanced story comprehension dataset. We focus on Javanese and Sundanese, two major local languages in Indonesia, and evaluate the effectiveness of open-weight and closed-weight LLMs in assisting dataset creation through extensive manual validation. To assess the utility of synthetic data, we fine-tune language models on classification and generation tasks using this data and evaluate performance on a human-written test set. Our findings indicate that LLM-assisted data creation outperforms machine translation.

### Citation
```
@misc{pranida2025syntheticdatagenerationculturally,
      title={Synthetic Data Generation for Culturally Nuanced Commonsense Reasoning in Low-Resource Languages}, 
      author={Salsabila Zahirah Pranida and Rifo Ahmad Genadi and Fajri Koto},
      year={2025},
      eprint={2502.12932},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.12932}, 
}
```