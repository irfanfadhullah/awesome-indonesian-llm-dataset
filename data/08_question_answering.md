# ❓ Question Answering

This section covers datasets for question answering tasks including reading comprehension, factoid QA, and mathematical reasoning in Indonesian.

## Reading Comprehension Datasets

### SQuAD (Indonesian)
- **Description**: Reading comprehension dataset translated from Stanford Question Answering Dataset
- **Content**: Question-answer pairs based on Wikipedia passages
- **Translation Methods**: 
  - Google Translate alignment (similar to Swedish SQuAD approach)
  - TranslateAlignRetrieve with MarianMT
- **Links**: [Download](https://stor.akmal.dev/squad/) | [Original](https://rajpurkar.github.io/SQuAD-explorer/)
- **Paper**: Stanford Question Answering Dataset (SQuAD) 2.0

**Usage Note**: If using with HuggingFace `run_qa.py`, run the `convert_huggingface.py` script for compatibility.

### Mathematics Dataset (Indonesian)
- **Description**: Mathematical question-answer pairs covering school-level difficulty
- **Size**: 1,000 questions per module
- **Content**: Algebraic reasoning, problem solving across various mathematical domains
- **Links**: [GitHub Repository](https://github.com/Wikidepia/indonesia_dataset/tree/master/question-answering/mathematics_dataset) | [Original](https://github.com/deepmind/mathematics_dataset)
- **Generation**: Created using [mathematics_dataset_id](https://github.com/Wikidepia/mathematics_dataset_id)

**Skills Tested:**
- Mathematical learning capability
- Algebraic reasoning
- Problem-solving across mathematical domains

## Indonesian NLP Benchmark QA Tasks

### FacQA
- **Description**: Factoid question answering from news articles
- **Task**: Find answers from provided short passages
- **Links**: [GitHub](https://github.com/IndoNLP/indonlu/tree/master/dataset/facqa_qa-factoid-itb) | [HuggingFace](https://huggingface.co/datasets/SEACrowd/facqa)

### TyDi QA
- **Description**: Typologically diverse question answering dataset
- **Coverage**: 11 languages including Indonesian
- **Size**: 204K question-answer pairs across all languages
- **Links**: [GitHub](https://github.com/google-research-datasets/tydiqa) | [HuggingFace](https://huggingface.co/datasets/SEACrowd/tydiqa)

**Key Features:**
- Natural information-seeking questions
- No translation artifacts
- Cross-lingual evaluation

### mLAMA
- **Description**: Multilingual Language Model Analysis dataset
- **Purpose**: Probing factual knowledge in multilingual language models
- **Links**: [HuggingFace](https://huggingface.co/datasets/cis-lmu/m_lama)

### QED (Educational Domain)
- **Description**: Question-answer pairs from educational content
- **Coverage**: Multilingual educational corpus including Indonesian
- **Links**: [Opus](https://opus.nlpl.eu/QED.php)

## Dataset Characteristics

| Dataset | Type | Size | Domain | Source Language |
|---------|------|------|---------|-----------------|
| SQuAD-ID | Reading Comprehension | ~100K pairs | Wikipedia | Translated from English |
| Mathematics-ID | Mathematical Reasoning | 1K per module | Education | Generated |
| FacQA | Factoid QA | Variable | News | Indonesian |
| TyDi QA | Reading Comprehension | 204K total | Wikipedia | Native Indonesian |

## Applications

**Reading Comprehension:**
- Educational assessment systems
- Information retrieval from documents
- Automated question answering services

**Mathematical Reasoning:**
- Educational technology platforms
- Automated math tutoring systems
- Mathematical problem solving evaluation

**Factoid QA:**
- Knowledge base question answering
- Information extraction systems
- News comprehension tasks

## Training Recommendations

**For Reading Comprehension Models:**
- Start with SQuAD-ID for general comprehension
- Add TyDi QA for cross-lingual robustness
- Fine-tune on domain-specific data (FacQA for news)

**For Mathematical Reasoning:**
- Use Mathematics Dataset for training
- Augment with translated mathematical problems
- Focus on step-by-step reasoning capabilities

## Data Quality Notes

- **SQuAD-ID**: Two versions available - test both for optimal results
- **TyDi QA**: Native Indonesian questions without translation artifacts
- **Mathematics Dataset**: Limited to 1K questions per module
- **Translation Quality**: Varies by method - manual review recommended

## Citations

```

@misc{rajpurkar2018know,
title={Know What You Don't Know: Unanswerable Questions for SQuAD},
author={Pranav Rajpurkar and Robin Jia and Percy Liang},
year={2018},
eprint={1806.03822},
archivePrefix={arXiv},
primaryClass={cs.CL}
}

@inproceedings{saxton2019analysing,
title={Analysing Mathematical Reasoning Abilities of Neural Models},
author={Saxton, David and Grefenstette, Edward and Hill, Felix and Kohli, Pushmeet},
booktitle={International Conference on Learning Representations},
year={2019}
}

@article{clark-etal-2020-tydi,
title = "{T}y{D}i {QA}: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages",
author = "Clark, Jonathan H. and others",
journal = "Transactions of the Association for Computational Linguistics",
volume = "8",
year = "2020",
pages = "454--470",
}

```