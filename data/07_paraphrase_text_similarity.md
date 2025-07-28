# 🔄 Paraphrase & Text Similarity

This section covers datasets for paraphrase identification, text similarity measurement, and natural language inference tasks in Indonesian.

## Paraphrase Collections

### PAWS (Indonesian)
- **Size**: 100K human-labeled paraphrase pairs
- **Source**: Translated from Google's PAWS dataset using Google Translate
- **Focus**: Word order and structural importance in paraphrase detection
- **Content**: High-quality pairs with human judgments on paraphrasing and fluency
- **Links**: [GitHub Repository](https://github.com/Wikidepia/indonesia_dataset/tree/master/paraphrase/PAWS) | [Original](https://github.com/google-research-datasets/paws)

**Dataset Splits:**
- **PAWS-Wiki Labeled (Final)**: Train/Dev/Test splits with complete human annotations
- **PAWS-Wiki Labeled (Swap-only)**: Additional high-quality pairs for auxiliary training

### ParaNMT-50M (Indonesian)
- **Size**: Subset of 50M+ English paraphrase pairs translated to Indonesian
- **Method**: Neural machine translation approach for paraphrase generation
- **Links**: [Download](https://stor.akmal.dev/paranmt-5m.jsonl.zst) | [Original](http://www.cs.cmu.edu/~jwieting/)

### ParaBank
- **Description**: Diverse paraphrastic bitexts via sampling and clustering
- **Method**: Guided backtranslation with constraints for paraphrase generation
- **Size**: Large-scale diverse paraphrase pairs
- **Links**: [Download](https://stor.akmal.dev/parabank-v2.0.jsonl.zst) | [Original](https://nlp.jhu.edu/parabank/)

### Quora Paraphrasing ID
- **Description**: Indonesian adaptation of Quora paraphrase pairs
- **Content**: Question-question paraphrase pairs
- **Links**: [GitHub Repository](https://github.com/louisowen6/quora_paraphrasing_id)

### SBERT Paraphrase Data
- **Description**: Various paraphrase datasets compiled by SBERT, translated to Indonesian
- **Content**: Multiple paraphrase datasets for sentence embedding training
- **Links**: [Download](https://storage.depia.wiki/sbert-paraphrase/)
- **Note**: May require data cleaning before use

## Natural Language Inference

### Indonesian MultiNLI
- **Description**: Multi-genre natural language inference corpus
- **Size**: 433K sentence pairs with textual entailment annotations
- **Genres**: Multiple genres of spoken and written text
- **Links**: [Download](https://stor.akmal.dev/idmultinli/) | [Original](https://cims.nyu.edu/~sbowman/multinli)

### Indonesian SNLI
- **Description**: Stanford Natural Language Inference corpus in Indonesian
- **Size**: 570K human-written sentence pairs
- **Labels**: Entailment, contradiction, neutral
- **Links**: [Download](https://stor.akmal.dev/idsnli/) | [Original](https://nlp.stanford.edu/projects/snli/)

## Dataset Characteristics

| Dataset | Size | Task Type | Annotation Quality | Domain |
|---------|------|-----------|-------------------|---------|
| PAWS-ID | 100K | Paraphrase Detection | Human-labeled | Wikipedia |
| ParaNMT-ID | 5M | Paraphrase Generation | Automatic | General |
| ParaBank | 2M+ | Paraphrase Pairs | Semi-automatic | Multi-domain |
| MultiNLI-ID | 433K | NLI | Human-labeled | Multi-genre |
| SNLI-ID | 570K | NLI | Human-labeled | Image descriptions |

## Applications

**Paraphrase Detection:**
- Duplicate content identification
- Plagiarism detection
- Information retrieval enhancement
- Question answering systems

**Text Similarity:**
- Semantic textual similarity
- Document clustering
- Content recommendation
- Search result ranking

**Natural Language Inference:**
- Reading comprehension
- Textual entailment
- Logic reasoning
- Question answering

## Training Recommendations

**For Paraphrase Models:**
- Start with PAWS for structural understanding
- Add ParaNMT for scale
- Fine-tune on domain-specific data

**For NLI Models:**
- Begin with SNLI for basic entailment
- Extend with MultiNLI for genre diversity
- Combine with IndoNLI for cultural relevance

## Data Quality Notes

- **PAWS**: High-quality human annotations, focus on adversarial examples
- **ParaNMT**: Large-scale but automatic generation, may contain noise
- **ParaBank**: Good balance of scale and diversity
- **Translation Quality**: Varies by dataset; manual review recommended for critical applications

## Citations

```

@misc{zhang2019paws,
title={PAWS: Paraphrase Adversaries from Word Scrambling},
author={Yuan Zhang and Jason Baldridge and Luheng He},
year={2019},
eprint={1904.01130},
archivePrefix={arXiv},
primaryClass={cs.CL}
}

@inproceedings{wieting-gimpel-2018-paranmt,
title = "{P}ara{NMT}-50{M}: Pushing the Limits of Paraphrastic Sentence Embeddings with Millions of Machine Translations",
author = "Wieting, John and Gimpel, Kevin",
booktitle = "Proceedings of ACL",
year = "2018"
}

@inproceedings{hu-etal-2019-large,
title = "Large-Scale, Diverse, Paraphrastic Bitexts via Sampling and Clustering",
author = "Hu, J. Edward and Singh, Abhinav and others",
booktitle = "Proceedings of CoNLL",
year = "2019"
}

@misc{williams2018broadcoverage,
title={A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference},
author={Adina Williams and Nikita Nangia and Samuel R. Bowman},
year={2018},
eprint={1704.05426},
archivePrefix={arXiv},
primaryClass={cs.CL}
}

```