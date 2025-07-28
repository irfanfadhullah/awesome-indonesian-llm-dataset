# 📝 Text Summarization

This section covers datasets for both abstractive and extractive text summarization in Indonesian, ranging from news articles to social media content.

## News Summarization Datasets

### Liputan6
- **Description**: Large-scale Indonesian abstractive and extractive summarization corpus
- **Period**: News articles from 2000-2010
- **Size**: 215K+ article-summary pairs
- **Types**: Both abstractive summaries and extractive labels
- **Links**: [GitHub Repository](https://github.com/indolem/sum_liputan6/) | [Access Form](https://docs.google.com/forms/d/1bFkimFsZoswKCbUa76yHqi9hizLrJYne-1G_r5unfww/edit)
- **Paper**: AACL-IJCNLP 2020

**Dataset Splits:**

| Data Split | Train | Dev | Test |
|------------|-------|-----|------|
| Canonical | 193,883 | 10,972 | 10,972 |
| Xtreme | 193,883 | 4,948 | 3,862 |

**Performance Benchmarks:**

| Model | R1 | R2 | RL |
|-------|----|----|-----|
| Lead-2 | 36.68 | 20.23 | 33.71 |
| BertExtAbs (indoBERT) | **41.08** | 22.85 | **38.01** |
| BertAbs (indoBERT) | 40.94 | **23.01** | 37.89 |

### IndoSum
- **Description**: Multi-source online news article-summary pairs
- **Size**: 20K article-summary pairs
- **Categories**: 6 categories across 10 sources
- **Features**: Abstractive summaries + extractive sentence labels
- **Links**: [GitHub Repository](https://github.com/kata-ai/indosum) | [Data Download](https://drive.google.com/file/d/1OgYbPfXFAv3TbwP1Qcwt_CC9cVWSJaco/view)
- **Paper**: IALP 2018

**Key Features:**
- Multi-source diversity
- Quality-controlled summaries
- Both abstractive and extractive annotations

### XLSum
- **Description**: Large-scale multilingual abstractive summarization
- **Coverage**: 45 languages including Indonesian
- **Size**: 47,802 total (Indonesian: 38,242 train / 4,780 dev / 4,780 test)
- **Quality**: Highly abstractive, concise, and high quality
- **Links**: [HuggingFace](https://huggingface.co/datasets/csebuetnlp/xlsum)

### WikiLingua
- **Description**: Cross-lingual summarization dataset from WikiHow
- **Size**: ~770K article-summary pairs across 18 languages
- **Features**: Gold-standard article-summary alignments across languages
- **Links**: [GitHub Repository](https://github.com/esdurmus/Wikilingua) | [Data Download](https://drive.google.com/file/d/1sTCB5NDPq6vUOlxR29DbvSssErvXLD1d/view?usp=sharing)

## Social Media and Web Content

### Reddit TLDR (Indonesian)
- **Description**: Social media domain summaries translated from Reddit
- **Size**: ~4M content-summary pairs
- **Source**: Reddit TL;DR posts (2006-2016)
- **Links**: [Download](https://stor.akmal.dev/reddit-tldr.jsonl.zst)
- **Note**: Translation quality may vary due to Google Translate character limits

### WikiHow (Indonesian)
- **Description**: Headline and procedural text pairs from WikiHow Indonesia
- **Content**: How-to articles with step-by-step instructions
- **Links**: [Download](https://stor.akmal.dev/wikihow.json.zst) | [Original](https://github.com/mahnazkoupaee/WikiHow-Dataset)

## News Headlines and Short Summaries

### GigaWord (Indonesian)
- **Description**: Headline generation dataset from news articles
- **Size**: ~4M article-headline pairs
- **Task**: Article → headline summarization
- **Links**: [Download](https://stor.akmal.dev/gigaword.tar.zst)
- **Processing**: Uses improved format from Microsoft UniLM

### Indonesian Simple Summaries (Our Works)
- **Description**: Indonesian version of Simple Summaries dataset
- **Size**: ~10K pairs
- **Links**: [Download](https://huggingface.co/datasets/irfanfadhullah/indonesian-simple-summaries)

## Dataset Characteristics

### By Domain

| Dataset | Domain | Size | Type | Quality |
|---------|--------|------|------|---------|
| Liputan6 | News | 215K | Abstractive + Extractive | High |
| IndoSum | News | 20K | Abstractive + Extractive | High |
| XLSum | News | 47K | Abstractive | High |
| Reddit TLDR | Social Media | 4M | Abstractive | Medium |
| WikiHow | Educational | Variable | Procedural | High |
| GigaWord | News | 4M | Headlines | Medium |

### By Task Type

**Abstractive Summarization:**
- Liputan6, IndoSum, XLSum
- Reddit TLDR, WikiHow
- Require generation of new text

**Extractive Summarization:**
- Liputan6 (with extractive labels)
- IndoSum (with extractive labels)
- Select important sentences from source

**Headline Generation:**
- GigaWord
- Specialized short summarization task

## Applications

**News Summarization:**
- Automated news digest generation
- Article preview systems
- Content curation platforms

**Educational Content:**
- Automatic summary generation for learning materials
- Document summarization for research

**Social Media:**
- Thread summarization
- Content highlight extraction
- Social media monitoring

## Training Recommendations

**For News Summarization:**
- Start with Liputan6 for Indonesian-specific patterns
- Add XLSum for multilingual capabilities
- Use IndoSum for multi-source robustness

**For Social Media:**
- Use Reddit TLDR for informal language
- Combine with news data for domain adaptation

**For Educational Content:**
- WikiHow for procedural text
- Combine with academic content

## Data Quality and Preprocessing

**High-Quality Datasets:**
- Liputan6: Professional news summaries
- IndoSum: Multi-source verification
- XLSum: Quality-controlled multilingual data

**Moderate Quality:**
- Reddit TLDR: Translation artifacts possible
- GigaWord: Automatic headline matching

**Preprocessing Notes:**
- Check translation quality for translated datasets
- Verify extractive labels for consistency
- Consider domain adaptation for cross-domain use

## Legal and Ethical Considerations

**Copyright Notice:**
- Liputan6: Non-commercial academic research only
- News-based datasets: Check original source licenses
- Social media data: Respect platform terms of service

**Usage Guidelines:**
- Academic research encouraged
- Commercial use may require additional permissions
- Respect original content creators' rights

## Citations

```

@inproceedings{koto-etal-2020-liputan6,
title = "Liputan6: A Large-scale Indonesian Dataset for Text Summarization",
author = "Koto, Fajri and Lau, Jey Han and Baldwin, Timothy",
booktitle = "Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing",
year = "2020"
}

@inproceedings{kurniawan2018,
title={IndoSum: A New Benchmark Dataset for Indonesian Text Summarization},
author={Kurniawan, Kemal and Louvan, Samuel},
booktitle={2018 International Conference on Asian Language Processing (IALP)},
year={2018},
publisher={IEEE}
}

@inproceedings{hasan-etal-2021-xl,
title = "{XL}-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages",
author = "Hasan, Tahmid and others",
booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
year = "2021"
}

@inproceedings{volske-etal-2017-tl,
title = "{TL};{DR}: Mining {R}eddit to Learn Automatic Summarization",
author = "Völske, Michael and others",
booktitle = "Proceedings of the Workshop on New Frontiers in Summarization",
year = "2017"
}

```