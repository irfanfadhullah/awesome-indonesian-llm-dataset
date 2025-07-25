# 🇮🇩 Awesome Indonesian LLM Dataset

A curated collection of high-quality Indonesian datasets for training Large Language Models (LLMs), Vision-Language Models (VLMs), and multimodal AI systems. This repository serves as a comprehensive resource for researchers and practitioners working on Indonesian natural language processing and AI.

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [📋 TODO](#-todo)
- [📊 Dataset Categories](#-dataset-categories)
    - [🧠 Natural Language Understanding](#-natural-language-understanding)
    - [📚 Knowledge Graphs](#-knowledge-graphs)
    - [🌐 Web Crawl \& Text Corpora](#-web-crawl--text-corpora)
    - [🗣️ Local Languages](#-local-languages)
    - [🖼️ Multimodal \& Vision-Language](#-multimodal--vision-language)
    - [🔄 Paraphrase \& Text Similarity](#-paraphrase--text-similarity)
    - [❓ Question Answering](#-question-answering)
    - [🎙️ Speech \& Audio](#-speech--audio)
    - [📝 Text Summarization](#-text-summarization)
    - [🌍 Machine Translation](#-machine-translation)
    - [📖 Dictionary \& Vocabulary](#-dictionary--vocabulary)
- [🚀 Quick Start](#-quick-start)
- [🤝 Contributing](#-contributing)
- [📚 Key Papers](#-key-papers)
- [⭐ Star This Repository](#-star-this-repository)
- [📧 Contact](#-contact)
- [📄 License](#-license)
- [📖 Citations](#-citations)


## 🎯 Overview

Indonesia, with over 700 local languages and 270 million speakers, represents one of the world's most linguistically diverse regions. This collection aims to support the development of AI systems that understand and generate Indonesian text across various domains and modalities.

**Key Statistics:**

- **15+ dataset categories** covering diverse NLP tasks
- **100+ individual datasets** from text to multimodal
- **Multiple language variants** including Bahasa Indonesia and local languages
- **Various scales** from small specialized datasets to large-scale corpora


## 📋 TODO

### 🚧 Upcoming Features

We are actively working on improving the awesome-indonesian-llm-dataset repository with the following planned features:

#### 🤗 HuggingFace Repository Integration

- **Create centralized HuggingFace organization/repository** for hosting all Indonesian datasets
- **Standardize dataset formats** across all collections for consistency
- **Implement dataset cards** with comprehensive metadata and usage examples
- **Set up automated dataset validation** and quality checks
- **Enable direct dataset loading** via HuggingFace `datasets` library
- **Provide dataset versioning** and update tracking


#### 🔧 Unified Access \& Preprocessing Code

- **Develop `indonesian-nlp` Python package** for easy dataset access
- **Create standardized preprocessing pipelines** for different data types:
    - Text normalization and cleaning
    - Language detection and filtering
    - Format conversion utilities
    - Train/validation/test splitting
- **Implement unified data loaders** with consistent APIs across all datasets
- **Add data validation and quality metrics** for each dataset
- **Provide pre-built tokenization** for Indonesian language models
- **Create benchmark evaluation scripts** for common NLP tasks


#### 📚 Documentation \& Examples

- **Comprehensive usage tutorials** for each dataset category
- **Jupyter notebook examples** demonstrating common use cases
- **Model training templates** using the unified preprocessing code
- **Performance benchmarks** and evaluation metrics
- **Integration guides** with popular NLP frameworks (Transformers, spaCy, etc.)


#### 🔄 Community Features

- **Dataset request system** for missing language resources
- **Quality feedback mechanism** for dataset improvements
- **Contribution guidelines** for adding new datasets
- **Regular dataset updates** and maintenance schedule


### 🤝 Get Involved

Interested in contributing to these developments? We welcome:

- **Dataset contributions** and quality improvements
- **Code contributions** for preprocessing utilities
- **Documentation** and tutorial writing
- **Testing and feedback** on beta features

Please check our [GitHub Issues](https://github.com/irfanfadhullah/awesome-indonesia-llm-dataset/issues) or start a [Discussion](https://github.com/irfanfadhullah/awesome-indonesia-llm-dataset/discussions) to get involved!

## 📊 Dataset Categories

### 🧠 Natural Language Understanding

#### IndoNLI: Natural Language Inference Dataset

- **Description**: Human-annotated NLI data for Indonesian with train/val/test splits
- **Size**: Thousands of premise-hypothesis pairs
- **Tasks**: Natural language inference, textual entailment
- **Links**: [GitHub Repository](https://github.com/ir-nlp-csui/indonli) | [HuggingFace](https://huggingface.co/datasets/indonli)
- **Paper**: EMNLP 2021

**Key Features:**

- Expert and lay annotator splits
- Diagnostic subset with linguistic phenomena
- Translated MNLI dataset included


#### Indonesian NLP Resources

Comprehensive collection of datasets for various NLP tasks:

- **Repository**: [GitHub](https://github.com/kmkurn/id-nlp-resource)
- **POS Tagging**: [IDN tagged corpus](https://github.com/famrashel/idn-tagged-corpus) (10K sentences, 250K tokens)
- **Sentiment Analysis**:
    - [Hotel Reviews ABSA](https://github.com/jordhy97/final_project) (5K reviews, 78K tokens)
    - [Aspect-Based Sentiment Analysis](https://github.com/annisanurulazhar/absa-playground)
- **Text Classification**:
    - [SMS Spam Detection](https://drive.google.com/file/d/1-stKadfTgJLtYsHWqXhGO3nTjKVFxm_Q/view) (1,143 sentences)
    - [Hate Speech Detection](https://github.com/ialfina/id-hatespeech-detection) (713 tweets)
    - [Abusive Language Detection](https://github.com/okkyibrohim/id-abusive-language-detection)
- **Syntactic Parsing**:
    - [Indonesian Treebank](https://github.com/famrashel/idn-treebank) (1K parsed sentences)
    - [UD Indonesian](https://github.com/UniversalDependencies/UD_Indonesian-GSD)


### 📚 Knowledge Graphs

#### IndoWiki: Indonesian Knowledge Graph

- **Description**: Knowledge graph from WikiData aligned with Indonesian Wikipedia
- **Size**: 533K+ entities, 939 relations, 2.6M+ triplets
- **Format**: Both transductive and inductive splits available
- **Links**: [GitHub Repository](https://github.com/IgoRamli/IndoWiki/) | [Google Drive](https://drive.google.com/drive/folders/1V79VrSJ_ljz652iETARjHoB_zEfEIxV1?usp=sharing)

**Data Structure:**

- Transductive setting: Shared entities/relations across splits
- Inductive setting: Disjoint entities between train/test
- Complete pipeline for data processing included


### 🌐 Web Crawl \& Text Corpora

#### Large-Scale Text Collections

**OSCAR Corpus**

- **Size**: 4B word tokens, 2B word types
- **Source**: CommonCrawl extraction
- **Links**: [HuggingFace OSCAR-2301](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301) | [TRACES OSCAR](https://traces1.inria.fr/oscar/#corpus)

**CC-100**

- **Size**: ~4.8B sentences, 6B sentence piece tokens
- **Source**: FAIR's CommonCrawl extraction
- **Links**: [StatMT CC-100](http://data.statmt.org/cc-100/)

**CulturaX**

- **Links**: [HuggingFace CulturaX](https://huggingface.co/datasets/uonlp/CulturaX)

**Indonesian News Corpus**

- **Size**: 150,466 news articles from 2015
- **Links**: [Mendeley Dataset](https://data.mendeley.com/datasets/2zpbjs22k3/1)

**Leipzig Corpora Collection**

- **Size**: 74M+ sentences, 1.2B+ tokens
- **Links**: [Leipzig Corpora](https://corpora.uni-leipzig.de/en?corpusId=ind_mixed_2013)

**IndoNLU Benchmark**

- **Size**: 4B words, 250M sentences
- **Links**: [HuggingFace Models](https://huggingface.co/indobenchmark) | [Dataset](https://storage.googleapis.com/babert-pretraining/IndoNLU_finals/dataset/preprocessed/dataset_all_uncased_blankline.txt.xz) | [GitHub](https://github.com/indobenchmark/indonlu)


#### Specialized Crawl Datasets

**Kaskus WebText**

- **Size**: 7K URLs from Indonesian forum with quality filtering
- **Links**: [Download](https://stor.akmal.dev/kaskus-webtext.tar.zst) | [Source List](https://stor.akmal.dev/kaskus.jsonl.zst)

**Wikipedia Links**

- **Size**: 58K URLs from Indonesian Wikipedia external links
- **Links**: [Download](https://stor.akmal.dev/wikipedia-links.tar.zst)

**Indonesian CommonCrawl News**

- **Description**: Processed CommonCrawl News filtered for Indonesian
- **Links**: [Download](https://stor.akmal.dev/ccnews-id.tar)

**Twitter Collections**

- **Twitter Puisi**: 80K+ poems from [@PelangiPuisi](https://twitter.com/PelangiPuisi)
- **Twitter Crawl**: [Download](https://stor.akmal.dev/twitter-dump/)

**"Clean" mC4**

- **Description**: Indonesian mC4 with spam/NSFW/gambling filtering
- **Source**: Based on [AllenAI Multilingual C4](https://github.com/allenai/allennlp/discussions/5265)

**Javanese Text**

- **Sources**: CC100, data.statmt.org, Wikipedia
- **Links**: [Download](https://stor.akmal.dev/jv-text/)


#### HuggingFace Datasets Collection

Additional datasets available on HuggingFace:

- [Indonesian Wikipedia](https://huggingface.co/datasets/indonesian-nlp/wikipedia-id)
- [IndoWiki by sabilmakbar](https://huggingface.co/datasets/sabilmakbar/indo_wiki)
- [Cendol Collection v1](https://huggingface.co/datasets/indonlp/cendol_collection_v1)
- [Cendol Collection v2](https://huggingface.co/datasets/indonlp/cendol_collection_v2)


### 🗣️ Local Languages

#### NusaX-MT: Multilingual Translation Dataset

- **Languages**: 12 languages including Indonesian + 10 local languages
- **Local Languages**: Acehnese, Balinese, Banjarese, Buginese, Madurese, Minangkabau, Javanese, Ngaju, Sundanese, Toba Batak
- **Size**: Parallel corpus across all language pairs
- **Links**: [GitHub Repository](https://github.com/IndoNLP/nusax/tree/main/datasets/mt) | [Paper](https://arxiv.org/abs/2205.15960)


#### Regional Language Resources

**MinangNLP**

- **Description**: Minangkabau language resources
- **Content**:
    - Bilingual dictionary: 11,905 Minangkabau-Indonesian pairs
    - Sentiment analysis: 5,000 parallel texts
    - Machine translation: 16,371 sentence pairs
- **Paper**: [PACLIC 2020](https://www.aclweb.org/anthology/2020.paclic-1.17.pdf)

**MadureseSet**

- **Size**: 17,809 basic + 53,722 substitution lemmata
- **Content**: Complete Madurese-Indonesian dictionary
- **Links**: [Mendeley Dataset](https://data.mendeley.com/datasets/nvc3rsf53b/5)
- **Paper**: [Data in Brief 2023](https://doi.org/10.1016/j.dib.2023.109035)

**Javanese Datasets**

- **Translated Dataset**: [HuggingFace](https://huggingface.co/datasets/ravialdy/javanese-translated)
- **Javanese Corpus**: [Kaggle](https://www.kaggle.com/datasets/hakikidamana/javanese-corpus/data)
- **Large Javanese ASR**: [OpenSLR](https://openslr.org/35/)

**NusaWrites Benchmark**

- **NusaTranslation**: 72,444 texts across 11 local languages
- **NusaParagraph**: 57,409 paragraphs across 10 local languages
- **Tasks**: Sentiment analysis, emotion classification, machine translation
- **Links**: [GitHub Repository](https://github.com/IndoNLP/nusa-writes) | [Paper](https://aclanthology.org/2023.ijcnlp-main.60/)

**HuggingFace Access for NusaWrites**:

```python
# NusaTranslation datasets
datasets.load_dataset('indonlp/nusatranslation_emot')
datasets.load_dataset('indonlp/nusatranslation_senti') 
datasets.load_dataset('indonlp/nusatranslation_mt')

# NusaParagraph datasets
datasets.load_dataset('indonlp/nusaparagraph_emot')
datasets.load_dataset('indonlp/nusaparagraph_rhetoric')
datasets.load_dataset('indonlp/nusaparagraph_topic')
```

**Sundanese Resources**

- **AMSunda Dataset**: [Zenodo](https://zenodo.org/records/15494944) - First dataset for Sundanese information retrieval
- **SU-CSQA**: [HuggingFace](https://huggingface.co/datasets/rifkiaputri/su-csqa) - Sundanese CommonsenseQA

**Indonesian Speech with Accents**

- **Coverage**: 5 ethnic groups (Batak, Malay, Javanese, Sundanese, Papuan)
- **Links**: [Kaggle Dataset](https://www.kaggle.com/datasets/hengkymulyono/indonesian-speech-with-accents-5-ethnic-groups)

**Story Cloze in Local Languages**

- **Languages**: Javanese and Sundanese
- **Links**: [HuggingFace](https://huggingface.co/datasets/rifoag/javanese_sundanese_story_cloze) | [Paper](https://arxiv.org/abs/2502.12932)


### 🖼️ Multimodal \& Vision-Language

#### Vision-Language Datasets

**Conceptual Captions (Indonesian)**

- **CC3M**: 3M image-caption pairs translated to Indonesian
    - **Links**: [Download](https://stor.akmal.dev/cc3m-train.jsonl.zst) | [Original](https://github.com/google-research-datasets/conceptual-captions)
- **CC12M**: 12M image-caption pairs for vision-language pre-training
    - **Links**: [Download](https://stor.akmal.dev/cc12m.jsonl.zst) | [Original](https://github.com/google-research-datasets/conceptual-12m)

**YFCC100M OpenAI Subset**

- **Size**: 14.8M images with natural language descriptions
- **Links**: [Download](https://stor.akmal.dev/yfcc100.jsonl.zst) | [Original](https://github.com/openai/CLIP/blob/main/data/yfcc100m.md)

**MSVD-Indonesian**

- **Description**: Video-text dataset derived from MSVD
- **Size**: ~80K video-text pairs
- **Tasks**: Text-to-video retrieval, video captioning
- **Links**: [GitHub Repository](https://github.com/willyfh/msvd-indonesian) | [Data](https://github.com/willyfh/msvd-indonesian/blob/main/data/MSVD-indonesian.txt) | [Videos](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/YouTubeClips.tar)
- **Paper**: [arXiv:2306.11341](https://arxiv.org/abs/2306.11341)

**KTP VLM Instruct Dataset**

- **Links**: [HuggingFace](https://huggingface.co/datasets/danielsyahputra/ktp-vlm-instruct-dataset)


#### Educational \& Professional Assessment

**IndoMMLU**

- **Description**: Multi-task language understanding benchmark
- **Size**: 14,906 questions across 63 tasks
- **Levels**: Primary school to university entrance exams
- **Coverage**: 46% Indonesian language + 9 local languages/cultures
- **Links**: [HuggingFace](https://huggingface.co/datasets/indolem/IndoMMLU)

**IndoCareer**

- **Description**: Professional certification exam questions
- **Size**: 8,834 multiple-choice questions
- **Sectors**: Healthcare, finance, design, tourism, education, law
- **Links**: [HuggingFace](https://huggingface.co/datasets/indolem/IndoCareer) | [Paper](https://arxiv.org/pdf/2409.08564)

**IndoCulture**

- **Description**: Cultural knowledge questions about Indonesia
- **Size**: 2,430 questions with multiple choice answers
- **Links**: [HuggingFace](https://huggingface.co/datasets/indolem/IndoCulture)

**IndoCloze**

- **Description**: Commonsense story understanding through cloze evaluation
- **Size**: 2,325 Indonesian stories (4-sentence premise + endings)
- **Split**: 1,000 train / 200 dev / 1,135 test
- **Award**: Best Paper Award at CSRR (ACL 2022)


### 🔄 Paraphrase \& Text Similarity

#### Paraphrase Collections

**PAWS (Indonesian)**

- **Size**: 100K human-labeled pairs
- **Source**: Translated from Google's PAWS dataset
- **Focus**: Word order and structure importance
- **Links**: [GitHub Repository](https://github.com/Wikidepia/indonesia_dataset/tree/master/paraphrase/PAWS) | [Original](https://github.com/google-research-datasets/paws)

**ParaNMT-50M (Indonesian)**

- **Size**: Subset of 50M+ English paraphrase pairs
- **Method**: Neural machine translation approach
- **Links**: [Download](https://stor.akmal.dev/paranmt-5m.jsonl.zst) | [Original](http://www.cs.cmu.edu/~jwieting/)

**ParaBank**

- **Description**: Diverse paraphrastic bitexts via sampling and clustering
- **Links**: [Download](https://stor.akmal.dev/parabank-v2.0.jsonl.zst) | [Original](https://nlp.jhu.edu/parabank/)

**Indonesian MultiNLI \& SNLI**

- **MultiNLI**: [Download](https://stor.akmal.dev/idmultinli/) | [Original](https://cims.nyu.edu/~sbowman/multinli)
- **SNLI**: [Download](https://stor.akmal.dev/idsnli/) | [Original](https://nlp.stanford.edu/projects/snli/)

**SBERT Paraphrase Data**

- **Description**: Various paraphrase datasets compiled by SBERT, translated to Indonesian
- **Links**: [Download](https://storage.depia.wiki/sbert-paraphrase/)


### ❓ Question Answering

#### QA Datasets

**SQuAD (Indonesian)**

- **Description**: Reading comprehension dataset
- **Source**: Translated Stanford Question Answering Dataset
- **Format**: Compatible with HuggingFace training scripts
- **Links**: [Download](https://stor.akmal.dev/squad/) | [Original](https://rajpurkar.github.io/SQuAD-explorer/)

**Mathematics Dataset (Indonesian)**

- **Size**: 1,000 questions per module
- **Content**: School-level mathematical reasoning problems
- **Tasks**: Algebraic reasoning, problem solving
- **Links**: [GitHub Repository](https://github.com/Wikidepia/indonesia_dataset/tree/master/question-answering/mathematics_dataset) | [Original](https://github.com/deepmind/mathematics_dataset)


### 🎙️ Speech \& Audio

#### Speech Recognition

**TITML-IDN Speech Corpus**

- **Size**: 20 speakers (11M, 9F), 343 utterances each
- **Quality**: Phonetically balanced
- **Access**: Academic/non-commercial use
- **Links**: [NII Research](http://research.nii.ac.jp/src/en/TITML-IDN.html)

**Large Javanese ASR Dataset**

- **Description**: Transcribed audio data for Javanese
- **Source**: Google collaboration with universities
- **Links**: [OpenSLR](https://openslr.org/35/)

**Indonesian Speech with Accents**

- **Coverage**: 5 ethnic groups (Batak, Malay, Javanese, Sundanese, Papuan)
- **Content**: 320-word standardized Indonesian text per speaker
- **Links**: [Kaggle Dataset](https://www.kaggle.com/datasets/hengkymulyono/indonesian-speech-with-accents-5-ethnic-groups)

**Indonesian Speech Recognition (Small)**

- **Size**: 50 utterances by single male speaker
- **Links**: [GitHub Repository](https://github.com/frankydotid/Indonesian-Speech-Recognition)
- **Note**: School project - not for production use

**CMU Wilderness Multilingual Speech**

- **Coverage**: 700+ languages including Indonesian
- **Source**: Bible recordings from bible.is
- **Links**: [GitHub Repository](https://github.com/festvox/datasets-CMU_Wilderness)

**Google TTS Dataset**

- **Description**: Automatically generated speech using Google Translate TTS
- **Source**: Indonesian newspaper titles
- **Links**: [Download](https://stor.akmal.dev/gtts-500k.zip)

**Indonesian Unsupervised Speech Dataset**

- **Content**: Podcast (170GB) + YouTube (90GB)
- **Access**: Contact akmal@depia.wiki
- **Note**: Research use only


### 📝 Text Summarization

#### Summarization Corpora

**Liputan6**

- **Size**: 215K+ article-summary pairs (193,883 train, 10,972 dev/test)
- **Period**: 2000-2010 news articles
- **Types**: Both abstractive and extractive summaries
- **Links**: [GitHub Repository](https://github.com/indolem/sum_liputan6/) | [Access Form](https://docs.google.com/forms/d/1bFkimFsZoswKCbUa76yHqi9hizLrJYne-1G_r5unfww/edit)
- **Paper**: AACL-IJCNLP 2020

**IndoSum**

- **Size**: 20K online news article-summary pairs
- **Categories**: 6 categories, 10 sources
- **Features**: Abstractive summaries + extractive labels
- **Links**: [GitHub Repository](https://github.com/kata-ai/indosum) | [Data Download](https://drive.google.com/file/d/1OgYbPfXFAv3TbwP1Qcwt_CC9cVWSJaco/view)
- **Paper**: IALP 2018


#### Translated Summarization Datasets

**Reddit TLDR**

- **Size**: ~4M content-summary pairs
- **Source**: Social media domain summaries
- **Links**: [Download](https://stor.akmal.dev/reddit-tldr.jsonl.zst)

**WikiHow (Indonesian)**

- **Description**: Headline and procedural text pairs
- **Source**: WikiHow Indonesia
- **Links**: [Download](https://stor.akmal.dev/wikihow.json.zst) | [Original](https://github.com/mahnazkoupaee/WikiHow-Dataset)

**GigaWord**

- **Description**: Article headline generation dataset
- **Size**: ~4M article pairs
- **Links**: [Download](https://stor.akmal.dev/gigaword.tar.zst)


### 🌍 Machine Translation

#### Translation Pairs

**OPUS Parallel Corpus**

- **Description**: Indonesian paired with 21+ European languages
- **Source**: European Parliament proceedings, OpenSubtitles
- **Links**: [OPUS Collection](http://opus.nlpl.eu/)

**ParaCrawl v7.1**

- **Description**: Web-scale parallel corpora (41 language pairs)
- **Method**: Bitextor parallel data crawling
- **Links**: [HuggingFace IndoParaCrawl](https://huggingface.co/datasets/Wikidepia/IndoParaCrawl)

**Europarl**

- **Description**: European Parliament proceedings in 21 languages
- **Links**: [Download](https://storage.depia.wiki/europarl.jsonl.zst) | [Original](http://opus.nlpl.eu/)

**EuroPat v2**

- **Description**: Patent translations from USPTO and EPO
- **Links**: [Download](https://storage.depia.wiki/europat.jsonl.zst) | [Original](https://europat.net/)


#### Specialized Translation Datasets

**IWSLT2017**

- **Description**: TEDtalk subtitles (ID-EN)
- **Size**: ~100K sentences
- **Links**: [WIT3 IWSLT2017](https://wit3.fbk.eu/mt.php?release=2017-01-more)

**Asian Language Treebank**

- **Languages**: Indonesian, English, and Asian languages
- **Size**: 20K sentences
- **Links**: [ALT Corpus](http://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/)

**IDENTICv1.0**

- **Languages**: Indonesian-English
- **Size**: 45K sentences, ~1M tokens
- **Links**: [LINDAT Repository](https://lindat.mff.cuni.cz/repository/xmlui/handle/11858/00-097C-0000-0005-BF85-F)


### 📖 Dictionary \& Vocabulary

#### Lexical Resources

**Indonesia Wordlist**

- **Size**: 105,226 words from Kamus Besar Bahasa Indonesia
- **Coverage**: Most common Indonesian vocabulary
- **Links**: [GitHub Repository](https://github.com/Wikidepia/indonesian_datasets/tree/master/dictionary/wordlist/data/wordlist.txt)

**Colloquial Indonesian Lexicon**

- **Size**: 3,592 colloquial tokens → 1,742 lemmas
- **Purpose**: Text normalization for informal language
- **Links**: [GitHub Repository](https://github.com/nasalsabila/kamus-alay)
- **Paper**: [IEEE 2018](https://ieeexplore.ieee.org/abstract/document/8629151)


## 🚀 Quick Start

### Loading Datasets

**From HuggingFace:**

```python
import datasets

# Load IndoNLI
indonli = datasets.load_dataset('indonli')

# Load NusaX datasets
nusax_mt = datasets.load_dataset('indonlp/nusatranslation_mt')
nusax_sentiment = datasets.load_dataset('indonlp/nusatranslation_senti')

# Load OSCAR Indonesian
oscar_id = datasets.load_dataset('oscar-corpus/OSCAR-2301', language='id')

# Load IndoMMLU
indommlu = datasets.load_dataset('indolem/IndoMMLU')

# Load IndoCareer
indocareer = datasets.load_dataset('indolem/IndoCareer')
```

**From GitHub Repositories:**

```bash
# Clone specific dataset repositories
git clone https://github.com/ir-nlp-csui/indonli
git clone https://github.com/kata-ai/indosum
git clone https://github.com/IndoNLP/nusax
git clone https://github.com/indolem/sum_liputan6
git clone https://github.com/IgoRamli/IndoWiki
```


### Example Usage

**Training a Sentiment Classifier:**

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load Indonesian sentiment data
dataset = load_dataset('indonlp/nusatranslation_senti')

# Use Indonesian BERT
tokenizer = AutoTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
model = AutoModelForSequenceClassification.from_pretrained('indobenchmark/indobert-base-p1')

# Training code here...
```


## 🤝 Contributing

We welcome contributions to expand this collection! Please:

1. **Fork** this repository
2. **Add** new datasets with proper documentation
3. **Include** dataset descriptions, sizes, and access links
4. **Provide** citation information when available
5. **Submit** a pull request with your additions

### Dataset Submission Guidelines

When adding new datasets, please include:

- **Dataset name** and description
- **Size** and format information
- **Download links** or access instructions
- **License** and usage restrictions
- **Citation** information if available
- **Example usage** code when possible


## 📚 Key Papers

- **IndoNLI**: Mahendra et al., EMNLP 2021
- **NusaX**: Winata et al., EACL 2022
- **IndoSum**: Kurniawan \& Louvan, IALP 2018
- **Liputan6**: Koto et al., AACL-IJCNLP 2020
- **IndoMMLU**: Koto et al., EMNLP 2023
- **MSVD-Indonesian**: Hendria, arXiv 2023
- **NusaWrites**: Cahyawijaya et al., AACL-IJCNLP 2023
- **IndoCareer**: Koto, NAACL HLT 2025
- **IndoCloze**: Koto et al., CSRR at ACL 2022 (**Best Paper Award**)


## ⭐ Star This Repository

**🌟 Star this repository** if you find it useful for your Indonesian NLP research!

## 📧 Contact

**📧 Contact**: Open an issue for questions, suggestions, or dataset additions.

## 📄 License

This repository serves as a collection of links and information about publicly available datasets. Each dataset has its own license terms:

- **Most academic datasets**: Non-commercial research use only
- **News-based datasets**: Often restricted by copyright law
- **Government/public domain**: More permissive licensing
- **Translated datasets**: Subject to original dataset licenses

⚠️ **Important**: Always check individual dataset licenses before use, especially for commercial applications.

## 📖 Citations

If you use datasets from this collection, please cite the relevant papers and acknowledge the original dataset creators. See individual dataset sections for specific citation information.

## 📖 Citations

### Repository Citation

When using datasets from this collection, please cite this repository:

```bibtex
@misc{awesome-indonesian-llm-dataset,
  title={Awesome Indonesian LLM Dataset: A Curated Collection for Indonesian AI},
  author={Muhamad Irfan Fadhullah},
  year={2025},
  url={https://github.com/irfanfadhullah/awesome-indonesian-llm-dataset}
}
```


### Individual Dataset Citations

**IndoNLI:**

```bibtex
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
```

**NusaX:**

```bibtex
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

**IndoSum:**

```bibtex
@inproceedings{kurniawan2018,
  place={Bandung, Indonesia},
  title={IndoSum: A New Benchmark Dataset for Indonesian Text Summarization},
  url={https://ieeexplore.ieee.org/document/8629109},
  DOI={10.1109/IALP.2018.8629109},
  booktitle={2018 International Conference on Asian Language Processing (IALP)},
  publisher={IEEE},
  author={Kurniawan, Kemal and Louvan, Samuel},
  year={2018},
  month={Nov},
  pages={215-220}
}
```

**Liputan6:**

```bibtex
@inproceedings{koto-etal-2020-liputan6,
    title = "Liputan6: A Large-scale Indonesian Dataset for Text Summarization",
    author = "Koto, Fajri and Lau, Jey Han and Baldwin, Timothy",
    booktitle = "Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing",
    year = "2020"
}
```

**IndoMMLU:**

```bibtex
@inproceedings{koto-etal-2023-indommlu,
    title = "Large Language Models Only Pass Primary School Exams in {I}ndonesia: A Comprehensive Test on {I}ndo{MMLU}",
    author = "Fajri Koto and Nurul Aisyah and Haonan Li and Timothy Baldwin",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = December,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
}
```

**MSVD-Indonesian:**

```bibtex
@article{Hendria2023MSVDID,
  title={{MSVD}-{I}ndonesian: A Benchmark for Multimodal Video-Text Tasks in Indonesian},
  author={Willy Fitra Hendria},
  journal={arXiv preprint arXiv:2306.11341},
  year={2023}
}
```

**NusaWrites:**

```bibtex
@inproceedings{cahyawijaya-etal-2023-nusawrites,
    title = "{N}usa{W}rites: Constructing High-Quality Corpora for Underrepresented and Extremely Low-Resource Languages",
    author = "Cahyawijaya, Samuel  and  Lovenia, Holy  and Koto, Fajri  and  Adhista, Dea  and  Dave, Emmanuel  and  Oktavianti, Sarah  and  Akbar, Salsabil  and  Lee, Jhonson  and  Shadieq, Nuur  and  Cenggoro, Tjeng Wawan  and  Linuwih, Hanung  and  Wilie, Bryan  and  Muridan, Galih  and  Winata, Genta  and  Moeljadi, David  and  Aji, Alham Fikri  and  Purwarianti, Ayu  and  Fung, Pascale",
    booktitle = "Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = nov,
    year = "2023",
    address = "Nusa Dua, Bali",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.ijcnlp-main.60",
    pages = "921--945",
}
```

**MinangNLP:**

```bibtex
@inproceedings{koto-koto-2020-towards,
    title = "Towards Computational Linguistics in Minangkabau Language: Studies on Sentiment Analysis and Machine Translation",
    author = "Koto, Fajri and Koto, Ikhwan",
    booktitle = "Proceedings of the 34th Pacific Asia Conference on Language, Information and Computation",
    year = "2020"
}
```

**MadureseSet:**

```bibtex
@article{ifada2023madureseset,
  title={MadureseSet: Madurese-Indonesian Dataset},
  author={Ifada, N. and Rachman, F.H. and Syauqy, M.W.M.A. and Wahyuni, S. and Pawitra, A.},
  journal={Data in Brief},
  volume={48},
  pages={109035},
  year={2023},
  doi={https://doi.org/10.1016/j.dib.2023.109035}
}
```

**IndoCareer:**

```bibtex
@inproceedings{koto2025cracking,
  title={Cracking the Code: Multi-domain LLM Evaluation on Real-World Professional Exams in Indonesia},
  author={"Fajri Koto"},
  booktitle={Proceedings of the 2025 Conference of the North American Chapter of the Association for Computational Linguistics – Human Language Technologies (NAACL HLT 2025), Industry Track},
  year={2025}
}
```

**PAWS (Indonesian):**

```bibtex
@misc{zhang2019paws,
      title={PAWS: Paraphrase Adversaries from Word Scrambling}, 
      author={Yuan Zhang and Jason Baldridge and Luheng He},
      year={2019},
      eprint={1904.01130},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

**CC-100:**

```bibtex
@inproceedings{wenzek-etal-2020-ccnet,
    title = "{CCN}et: Extracting High Quality Monolingual Datasets from Web Crawl Data",
    author = "Wenzek, Guillaume  and
      Lachaux, Marie-Anne  and
      Conneau, Alexis  and
      Chaudhary, Vishrav  and
      Guzm{\'a}n, Francisco  and
      Joulin, Armand  and
      Grave, {\'E}douard",
    booktitle = "Proceedings of The 12th Language Resources and Evaluation Conference",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2020.lrec-1.494",
    pages = "4003--4012",
}
```

**OSCAR:**

```bibtex
@inproceedings{abadji-etal-2022-towards,
    title = "Towards a Cleaner Document-Oriented Multilingual Crawled Corpus",
    author = "Abadji, Julien  and
      Ortiz Suarez, Pedro  and
      Romary, Laurent  and
      Sagot, Benoit",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.463",
    pages = "4344--4355",
}
```

**Conceptual Captions:**

```bibtex
@inproceedings{sharma2018conceptual,
  title = {Conceptual Captions: A Cleaned, Hypernymed, Image Alt-text Dataset For Automatic Image Captioning},
  author = {Sharma, Piyush and Ding, Nan and Goodman, Sebastian and Soricut, Radu},
  booktitle = {Proceedings of ACL},
  year = {2018},
}
```

**Large Javanese ASR Dataset:**

```bibtex
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


**OSCAR Corpus:**

```bibtex
@inproceedings{abadji-etal-2022-towards,
    title = "Towards a Cleaner Document-Oriented Multilingual Crawled Corpus",
    author = "Abadji, Julien  and
      Ortiz Suarez, Pedro  and
      Romary, Laurent  and
      Sagot, Benoit",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.463",
    pages = "4344--4355",
}
```

**CCNet (CC-100):**

```bibtex
@inproceedings{wenzek-etal-2020-ccnet,
    title = "{CCN}et: Extracting High Quality Monolingual Datasets from Web Crawl Data",
    author = "Wenzek, Guillaume  and
      Lachaux, Marie-Anne  and
      Conneau, Alexis  and
      Chaudhary, Vishrav  and
      Guzm{\'a}n, Francisco  and
      Joulin, Armand  and
      Grave, {\'E}douard",
    booktitle = "Proceedings of The 12th Language Resources and Evaluation Conference",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2020.lrec-1.494",
    pages = "4003--4012",
}
```

**Conceptual Captions:**

```bibtex
@inproceedings{sharma2018conceptual,
  title = {Conceptual Captions: A Cleaned, Hypernymed, Image Alt-text Dataset For Automatic Image Captioning},
  author = {Sharma, Piyush and Ding, Nan and Goodman, Sebastian and Soricut, Radu},
  booktitle = {Proceedings of ACL},
  year = {2018},
}
```

**Conceptual 12M:**

```bibtex
@inproceedings{changpinyo2021cc12m,
  title = {{Conceptual 12M}: Pushing Web-Scale Image-Text Pre-Training To Recognize Long-Tail Visual Concepts},
  author = {Changpinyo, Soravit and Sharma, Piyush and Ding, Nan and Soricut, Radu},
  booktitle = {CVPR},
  year = {2021},
}
```

**YFCC100M:**

```bibtex
@article{Thomee_2016,
   title={YFCC100M},
   volume={59},
   ISSN={1557-7317},
   url={http://dx.doi.org/10.1145/2812802},
   DOI={10.1145/2812802},
   number={2},
   journal={Communications of the ACM},
   publisher={Association for Computing Machinery (ACM)},
   author={Thomee, Bart and Shamma, David A. and Friedland, Gerald and Elizalde, Benjamin and Ni, Karl and Poland, Douglas and Borth, Damian and Li, Li-Jia},
   year={2016},
   month={Jan},
   pages={64–73}
}
```

**MultiNLI:**

```bibtex
@misc{williams2018broadcoverage,
      title={A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference}, 
      author={Adina Williams and Nikita Nangia and Samuel R. Bowman},
      year={2018},
      eprint={1704.05426},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

**ParaBank:**

```bibtex
@inproceedings{hu-etal-2019-large,
    title = "Large-Scale, Diverse, Paraphrastic Bitexts via Sampling and Clustering",
    author = "Hu, J. Edward  and
      Singh, Abhinav  and
      Holzenberger, Nils  and
      Post, Matt  and
      Van Durme, Benjamin",
    booktitle = "Proceedings of the 23rd Conference on Computational Natural Language Learning (CoNLL)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/K19-1005",
    doi = "10.18653/v1/K19-1005",
    pages = "44--54",
}
```

**ParaNMT-50M:**

```bibtex
@inproceedings{wieting-gimpel-2018-paranmt,
    title = "{P}ara{NMT}-50{M}: Pushing the Limits of Paraphrastic Sentence Embeddings with Millions of Machine Translations",
    author = "Wieting, John  and
      Gimpel, Kevin",
    booktitle = "Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2018",
    address = "Melbourne, Australia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P18-1042",
    doi = "10.18653/v1/P18-1042",
    pages = "451--462",
}
```

**SNLI:**

```bibtex
@inproceedings{snli:emnlp2015,
	Author = {Bowman, Samuel R. and Angeli, Gabor and Potts, Christopher, and Manning, Christopher D.},
	Booktitle = {Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
	Publisher = {Association for Computational Linguistics},
	Title = {A large annotated corpus for learning natural language inference},
	Year = {2015}
}
```

**Stanford Question Answering Dataset (SQuAD):**

```bibtex
@misc{rajpurkar2018know,
      title={Know What You Don't Know: Unanswerable Questions for SQuAD}, 
      author={Pranav Rajpurkar and Robin Jia and Percy Liang},
      year={2018},
      eprint={1806.03822},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

**Mathematics Dataset:**

```bibtex
@inproceedings{saxton2019analysing,
  title={Analysing Mathematical Reasoning Abilities of Neural Models},
  author={Saxton, David and Grefenstette, Edward and Hill, Felix and Kohli, Pushmeet},
  booktitle={International Conference on Learning Representations},
  year={2019}
}
```

**Reddit TLDR:**

```bibtex
@inproceedings{volske-etal-2017-tl,
    title = "{TL};{DR}: Mining {R}eddit to Learn Automatic Summarization",
    author = {V{\"o}lske, Michael  and
      Potthast, Martin  and
      Syed, Shahbaz  and
      Stein, Benno},
    booktitle = "Proceedings of the Workshop on New Frontiers in Summarization",
    month = sep,
    year = "2017",
    address = "Copenhagen, Denmark",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W17-4508",
    doi = "10.18653/v1/W17-4508",
    pages = "59--63",
}
```

**WikiHow:**

```bibtex
@misc{koupaee2018wikihow,
      title={WikiHow: A Large Scale Text Summarization Dataset}, 
      author={Mahnaz Koupaee and William Yang Wang},
      year={2018},
      eprint={1810.09305},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

**GigaWord:**

```bibtex
@article{graff2003english,
  title={English gigaword},
  author={Graff, David and Kong, Junbo and Chen, Ke and Maeda, Kazuaki},
  journal={Linguistic Data Consortium, Philadelphia},
  volume={4},
  number={1},
  pages={34},
  year={2003}
}
```

**Europarl:**

```bibtex
@inproceedings{koehn-2005-europarl,
    title = "{E}uroparl: A Parallel Corpus for Statistical Machine Translation",
    author = "Koehn, Philipp",
    booktitle = "Proceedings of Machine Translation Summit X: Papers",
    month = sep # " 13-15",
    year = "2005",
    address = "Phuket, Thailand",
    url = "https://aclanthology.org/2005.mtsummit-papers.11",
    pages = "79--86",
}
```

**ParaCrawl:**

```bibtex
@inproceedings{espla-etal-2019-paracrawl,
    title = "{P}ara{C}rawl: Web-scale parallel corpora for the languages of the {EU}",
    author = "Espl{\`a}, Miquel  and
      Forcada, Mikel  and
      Ram{\'\i}rez-S{\'a}nchez, Gema  and
      Hoang, Hieu",
    booktitle = "Proceedings of Machine Translation Summit XVII Volume 2: Translator, Project and User Tracks",
    month = aug,
    year = "2019",
    address = "Dublin, Ireland",
    publisher = "European Association for Machine Translation",
    url = "https://www.aclweb.org/anthology/W19-6721",
    pages = "118--119",
}
```

**AMSunda:**

```bibtex
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

**SU-CSQA:**

```bibtex
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

**Javanese Sundanese Story Cloze:**

```bibtex
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

**IndoCloze:**

```bibtex
@inproceedings{koto-etal-2022-cloze,
    title = "Cloze Evaluation for Deeper Understanding of Commonsense Stories in Indonesian",
    author = "Koto, Fajri  and
      Baldwin, Timothy  and
      Lau, Jey Han",
    booktitle = "Proceedings of the First Workshop on Commonsense Representation and Reasoning (CSRR 2022)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.csrr-1.2",
    pages = "9--25",
}
```

**Kompas Online Collection:**

```bibtex
@misc{kompas2002,
  title={Kompas Online Collection},
  author={University of Amsterdam},
  year={2002},
  url={http://ilps.science.uva.nl/resources/bahasa/}
}
```

**Tempo Online Collection:**

```bibtex
@misc{tempo2002,
  title={Tempo Online Collection},
  author={University of Amsterdam},
  year={2002},
  url={http://ilps.science.uva.nl/resources/bahasa/}
}
```

**Leipzig Corpora Collection:**

```bibtex
@misc{leipzig2013,
  title={Indonesian Mixed Corpus},
  author={Leipzig University},
  year={2013},
  url={https://corpora.uni-leipzig.de/en?corpusId=ind_mixed_2013}
}
```

**Indonesian News Corpus:**

```bibtex
@dataset{indonesian_news_2015,
  title={Indonesian News Corpus},
  author={Mendeley Data},
  year={2015},
  url={https://data.mendeley.com/datasets/2zpbjs22k3/1}
}
```

**IDN Tagged Corpus:**

```bibtex
@misc{rashel2018idn,
  title={IDN Tagged Corpus},
  author={Rashel, Fam},
  year={2018},
  url={https://github.com/famrashel/idn-tagged-corpus}
}
```

**Aspect and Opinion Terms Extraction:**

```bibtex
@article{jordhy2019aspect,
  title={Aspect and Opinion Terms Extraction for Hotel Reviews},
  author={Jordhy, Jordhy},
  journal={arXiv preprint arXiv:1908.04899},
  year={2019}
}
```

**Indonesian Treebank:**

```bibtex
@misc{rashel2018treebank,
  title={Indonesian Treebank},
  author={Rashel, Fam},
  year={2018},
  url={https://github.com/famrashel/idn-treebank}
}
```

**Universal Dependencies Indonesian:**

```bibtex
@misc{mcdonald2013universal,
  title={Universal Dependencies Indonesian-GSD},
  author={McDonald, Ryan and Nivre, Joakim and Yvonne, Quirmbach-Brundage and others},
  year={2013},
  url={https://github.com/UniversalDependencies/UD_Indonesian-GSD}
}
```

**IDENTICv1.0:**

```bibtex
@inproceedings{larasati2012identic,
  title={IDENTICv1.0: Indonesian-English Parallel Corpus},
  author={Larasati, Septina Dian and Kubon, Vladislav and Zeman, Daniel},
  booktitle={Proceedings of the Eighth International Conference on Language Resources and Evaluation (LREC'12)},
  year={2012},
  pages={644}
}
```

**CMU Wilderness Multilingual Speech Dataset:**

```bibtex
@inproceedings{black2019cmu,
  title={CMU Wilderness Multilingual Speech Dataset},
  author={Black, Alan W},
  booktitle={ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={5971--5975},
  year={2019},
  organization={IEEE}
}
```

**Colloquial Indonesian Lexicon:**

```bibtex
@inproceedings{nasalsabila2018,
  title={Colloquial Indonesian Lexicon},
  author={Nasalsabila, Nasal and others},
  booktitle={2018 International Conference on Asian Language Processing (IALP)},
  year={2018},
  organization={IEEE}
}
```

**TITML-IDN Speech Corpus:**

```bibtex
@misc{titml2018,
  title={TITML-IDN Speech Corpus},
  author={National Institute of Informatics},
  year={2018},
  url={http://research.nii.ac.jp/src/en/TITML-IDN.html}
}
```

**SMS Spam Dataset:**

```bibtex
@misc{wibisono2018sms,
  title={Indonesian SMS Spam Dataset},
  author={Wibisono, Yudi},
  year={2018}
}
```

**Hate Speech Detection:**

```bibtex
@misc{alfina2018hate,
  title={Indonesian Hate Speech Detection},
  author={Alfina, Ika and others},
  year={2018},
  url={https://github.com/ialfina/id-hatespeech-detection}
}
```

**Abusive Language Detection:**

```bibtex
@misc{ibrohim2018abusive,
  title={Indonesian Abusive Language Detection},
  author={Ibrohim, Okky and others},
  year={2018},
  url={https://github.com/okkyibrohim/id-abusive-language-detection}
}
```

**EuroPat v2:**

```bibtex
@misc{europat2019,
  title={EuroPat: Parallel Patent Corpus},
  author={European Patent Organisation},
  year={2019},
  url={https://europat.net/}
}
```
**Rush et al. 2015 (Neural Attention Model for GigaWord):**

```bibtex
@article{Rush_2015,
   title={A Neural Attention Model for Abstractive Sentence Summarization},
   url={http://dx.doi.org/10.18653/v1/D15-1044},
   DOI={10.18653/v1/d15-1044},
   journal={Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing},
   publisher={Association for Computational Linguistics},
   author={Rush, Alexander M. and Chopra, Sumit and Weston, Jason},
   year={2015}
}
```

**Radford et al. 2021 (CLIP - Learning Transferable Visual Models):**

```bibtex
@misc{radford2021learning,
      title={Learning Transferable Visual Models From Natural Language Supervision}, 
      author={Alec Radford and Jong Wook Kim and Chris Hallacy and Aditya Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
      year={2021},
      eprint={2103.00020},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

**Saxton et al. 2019 (Mathematics Dataset):**

```bibtex
@inproceedings{saxton2019analysing,
  title={Analysing Mathematical Reasoning Abilities of Neural Models},
  author={Saxton, David and Grefenstette, Edward and Hill, Felix and Kohli, Pushmeet},
  booktitle={International Conference on Learning Representations},
  year={2019}
}
```

**Raffel et al. 2020 (T5 - Exploring the Limits of Transfer Learning):**

```bibtex
@article{Raffel2020ExploringTL,
  title={Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer},
  author={Colin Raffel and Noam M. Shazeer and Adam Roberts and Katherine Lee and Sharan Narang and Michael Matena and Yanqi Zhou and W. Li and Peter J. Liu},
  journal={ArXiv},
  year={2020},
  volume={abs/1910.10683}
}
```

**Cheng \& Lapata 2016 (Neural Summarization referenced in IndoSum):**

```bibtex
@inproceedings{cheng-lapata-2016-neural,
    title = "Neural Summarization by Extracting Sentences and Words",
    author = "Cheng, Jianpeng  and
      Lapata, Mirella",
    booktitle = "Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2016",
    address = "Berlin, Germany",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P16-1046",
    doi = "10.18653/v1/P16-1046",
    pages = "484--494",
}
```
