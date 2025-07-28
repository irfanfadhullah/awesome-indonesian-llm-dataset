# 🌍 Machine Translation

This section covers parallel corpora and translation datasets for Indonesian paired with various languages, including both European and local Indonesian languages.

## European Language Pairs

### OPUS Parallel Corpus Collection
- **Description**: Large collection of parallel corpora for Indonesian paired with European languages
- **Coverage**: Indonesian with 21+ European languages
- **Sources**: European Parliament proceedings, OpenSubtitles, various domains
- **Links**: [OPUS Collection](http://opus.nlpl.eu/)

**Major Language Pairs:**
- Indonesian-English, Indonesian-German, Indonesian-French
- Indonesian-Spanish, Indonesian-Italian, Indonesian-Dutch
- And many more European language combinations

### ParaCrawl v7.1
- **Description**: Web-scale parallel corpora from crawled web data
- **Method**: Bitextor parallel data crawling with Bicleaner filtering
- **Coverage**: 41 language pairs, primarily with English
- **Links**: [HuggingFace IndoParaCrawl](https://huggingface.co/datasets/Wikidepia/IndoParaCrawl)
- **Download**: Requires `git lfs` for large files

**Data Processing Pipeline:**
1. Document downloading and preprocessing
2. Document and segment alignment
3. Noise filtering via Bicleaner
4. Quality assessment and validation

### Europarl
- **Description**: European Parliament proceedings in multiple languages
- **Coverage**: 21 European languages including Indonesian translations
- **Links**: [Download](https://storage.depia.wiki/europarl.jsonl.zst) | [Original](http://opus.nlpl.eu/)
- **Period**: Parliamentary proceedings from multiple years

### EuroPat v2
- **Description**: Patent translations from major patent offices
- **Sources**: USPTO (United States) and EPO (European Patent Organisation)
- **Domain**: Technical and legal patent documents
- **Links**: [Download](https://storage.depia.wiki/europat.jsonl.zst) | [Original](https://europat.net/)

## Specialized Translation Datasets

### IWSLT2017
- **Description**: TED talk subtitles for Indonesian-English translation
- **Size**: ~100K sentences
- **Domain**: Spoken language from educational presentations
- **Links**: [WIT3 IWSLT2017](https://wit3.fbk.eu/mt.php?release=2017-01-more)
- **Note**: Test set contains some training data overlap

### Asian Language Treebank (ALT)
- **Coverage**: Indonesian, English, and Asian languages
- **Size**: 20K sentences
- **Domain**: News articles
- **Focus**: South East Asian language pairs
- **Links**: [ALT Corpus](http://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/)

### IDENTICv1.0
- **Languages**: Indonesian-English parallel corpus
- **Size**: 45K sentences (~1M tokens Indonesian)
- **Domains**: Science, sports, international affairs, economy, news, movie subtitles
- **Links**: [LINDAT Repository](https://lindat.mff.cuni.cz/repository/xmlui/handle/11858/00-097C-0000-0005-BF85-F)
- **Formats**: Raw, tokenized, and CoNLL format available

## Multilingual and Cross-lingual Datasets

### TAPACO
- **Description**: Translation of paraphrases across languages
- **Links**: [HuggingFace](https://huggingface.co/datasets/tapaco)

### opus100
- **Description**: Multilingual parallel corpus covering 100 languages
- **Links**: [HuggingFace](https://huggingface.co/datasets/opus100)

### PANL BPPT
- **Description**: Indonesian translation corpus
- **Links**: [HuggingFace](https://huggingface.co/datasets/id_panl_bppt)

### Bible-uedin
- **Description**: Biblical text translations across languages
- **Links**: [Opus](https://opus.nlpl.eu/bible-uedin.php)

### Open Subtitles
- **Description**: Movie and TV subtitle translations
- **Links**: [HuggingFace](https://huggingface.co/datasets/open_subtitles)

## Local Language Translation

### NusaX-MT
- **Coverage**: Indonesian + 10 local Indonesian languages
- **Languages**: Acehnese, Balinese, Banjarese, Buginese, Madurese, Minangkabau, Javanese, Ngaju, Sundanese, Toba Batak
- **Size**: Parallel corpus across all language pairs
- **Links**: [GitHub Repository](https://github.com/IndoNLP/nusax/tree/main/datasets/mt)

## Dataset Statistics

### By Size and Quality

| Dataset | Size | Quality | Domain | Alignment Method |
|---------|------|---------|---------|------------------|
| ParaCrawl | Web-scale | Medium | Web content | Automatic |
| OPUS Collection | Large | High | Mixed | Manual + Automatic |
| Europarl | Medium | High | Parliamentary | Manual |
| IWSLT2017 | ~100K | High | Educational | Manual |
| IDENTICv1.0 | 45K | High | News + Mixed | Manual |
| ALT | 20K | High | News | Manual |

### By Language Pairs

**Indonesian-English:** Most resources available
- ParaCrawl, OPUS, IWSLT2017, IDENTICv1.0, ALT

**Indonesian-European Languages:** Good coverage via OPUS
- Europarl, ParaCrawl, Patent data

**Indonesian-Local Languages:** Limited but growing
- NusaX-MT (10 local languages)

## Applications

**Cross-lingual NLP:**
- Machine translation systems
- Cross-lingual information retrieval
- Multilingual language models

**Domain-Specific Translation:**
- Technical translation (patents)
- News translation
- Educational content translation

**Local Language Preservation:**
- Indonesian local language MT
- Language documentation
- Cultural preservation

## Training Recommendations

**For High-Resource Pairs (ID-EN):**
- Start with OPUS collection for variety
- Add ParaCrawl for scale
- Fine-tune on domain-specific data (IWSLT for spoken, patents for technical)

**For European Language Pairs:**
- Use Europarl for formal language
- Add OPUS subtitles for informal language
- Include patent data for technical domains

**For Local Languages:**
- Use NusaX-MT as primary resource
- Augment with monolingual data
- Consider back-translation for data augmentation

## Data Quality and Preprocessing

**High-Quality Sources:**
- Manual alignments (IWSLT, ALT, IDENTICv1.0)
- Curated collections (OPUS)

**Medium-Quality Sources:**
- Automatic alignments (ParaCrawl)
- May require additional filtering

**Preprocessing Recommendations:**
1. Language identification and filtering
2. Length ratio filtering
3. Deduplication
4. Domain-specific filtering
5. Quality scoring and thresholding

## Legal and Usage Considerations

**Open Access:**
- Most OPUS datasets
- Academic research datasets

**Restricted Access:**
- Some patent datasets
- Commercial subtitle data

**Best Practices:**
- Check individual dataset licenses
- Respect original content creators
- Follow academic use guidelines for research

## Citations

```

@inproceedings{espla-etal-2019-paracrawl,
title = "{P}ara{C}rawl: Web-scale parallel corpora for the languages of the {EU}",
author = "Esplà, Miquel and others",
booktitle = "Proceedings of Machine Translation Summit XVII Volume 2: Translator, Project and User Tracks",
year = "2019"
}

@inproceedings{koehn-2005-europarl,
title = "{E}uroparl: A Parallel Corpus for Statistical Machine Translation",
author = "Koehn, Philipp",
booktitle = "Proceedings of Machine Translation Summit X: Papers",
year = "2005"
}

@inproceedings{larasati2012identic,
title={IDENTICv1.0: Indonesian-English Parallel Corpus},
author={Larasati, Septina Dian and Kubon, Vladislav and Zeman, Daniel},
booktitle={Proceedings of the Eighth International Conference on Language Resources and Evaluation (LREC'12)},
year={2012}
}

@misc{winata2022nusax,
title={NusaX: Multilingual Parallel Sentiment Dataset for 10 Indonesian Local Languages},
author={Winata, Genta Indra and others},
year={2022},
archivePrefix={arXiv},
primaryClass={cs.CL}
}

```