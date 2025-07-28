# 🌐 Web Crawl & Text Corpora

This section covers large-scale text collections, web crawl datasets, and general text corpora for Indonesian language modeling and NLP research.

## Large-Scale Text Collections

### OSCAR Corpus
- **Size**: 4B word tokens, 2B word types
- **Source**: CommonCrawl extraction
- **Links**: [HuggingFace OSCAR-2301](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301) | [TRACES OSCAR](https://traces1.inria.fr/oscar/#corpus)

### CC-100
- **Size**: ~4.8B sentences, 6B sentence piece tokens
- **Source**: FAIR's CommonCrawl extraction
- **Links**: [StatMT CC-100](http://data.statmt.org/cc-100/)

### CulturaX
- **Links**: [HuggingFace CulturaX](https://huggingface.co/datasets/uonlp/CulturaX)

### Indonesian News Corpus
- **Size**: 150,466 news articles from 2015
- **Links**: [Mendeley Dataset](https://data.mendeley.com/datasets/2zpbjs22k3/1)

### Leipzig Corpora Collection
- **Size**: 74M+ sentences, 1.2B+ tokens
- **Links**: [Leipzig Corpora 2013](https://corpora.uni-leipzig.de/en?corpusId=ind_mixed_2013) | [Leipzig Corpora 2022](https://corpora.wortschatz-leipzig.de/en?corpusId=ind_news_2022)

### IndoNLU Benchmark
- **Size**: 4B words, 250M sentences
- **Links**: [HuggingFace Models](https://huggingface.co/indobenchmark) | [Dataset](https://storage.googleapis.com/babert-pretraining/IndoNLU_finals/dataset/preprocessed/dataset_all_uncased_blankline.txt.xz) | [GitHub](https://github.com/indobenchmark/indonlu)

## Other Large Corpora

- **WiLI-2018**: [Zenodo Link](https://zenodo.org/records/841984) - 235000 paragraphs of 235 languages
- **WikiANN**: [DropBox Link](https://www.dropbox.com/s/12h3qqog6q4bjve/panx_dataset.tar?dl=1) - Supports 176 languages
- **Tatoeba**: [HuggingFace Link](https://huggingface.co/datasets/Helsinki-NLP/tatoeba) - 32G translation units in 2,539 bitexts
- **QCRI Educational Domain Corpus**: [Link](https://opus.nlpl.eu/QED/en&es/v2.0a/QED) - 225 languages, 271,558 files

## Specialized Crawl Datasets

### Kaskus WebText
- **Size**: 7K URLs from Indonesian forum with quality filtering
- **Links**: [Download](https://stor.akmal.dev/kaskus-webtext.tar.zst) | [Source List](https://stor.akmal.dev/kaskus.jsonl.zst)

### Wikipedia Links
- **Size**: 58K URLs from Indonesian Wikipedia external links
- **Links**: [Download](https://stor.akmal.dev/wikipedia-links.tar.zst)

### Indonesian CommonCrawl News
- **Description**: Processed CommonCrawl News filtered for Indonesian
- **Links**: [Download](https://stor.akmal.dev/ccnews-id.tar)

### Twitter Collections
- **Twitter Puisi**: 80K+ poems from [@PelangiPuisi](https://twitter.com/PelangiPuisi)
- **Twitter Crawl**: [Download](https://stor.akmal.dev/twitter-dump/)

### Clean mC4
- **Description**: Indonesian mC4 with spam/NSFW/gambling filtering
- **Source**: Based on [AllenAI Multilingual C4](https://github.com/allenai/allennlp/discussions/5265)

### Javanese Text
- **Sources**: CC100, data.statmt.org, Wikipedia
- **Links**: [Download](https://stor.akmal.dev/jv-text/)

## Historical News Corpora

### Kompas Online Collection (2001–2002)
- **Links**: [Download](http://ilps.science.uva.nl/ilps/wp-content/uploads/sites/6/files/bahasaindonesia/kompas.zip)

### Tempo Online Collection (2000–2002)
- **Links**: [Download](http://ilps.science.uva.nl/ilps/wp-content/uploads/sites/6/files/bahasaindonesia/tempo.zip)

## HuggingFace Datasets Collection

Additional datasets available on HuggingFace:
- [Indonesian Wikipedia](https://huggingface.co/datasets/indonesian-nlp/wikipedia-id)
- [IndoWiki by sabilmakbar](https://huggingface.co/datasets/sabilmakbar/indo_wiki)
- [Cendol Collection v1](https://huggingface.co/datasets/indonlp/cendol_collection_v1)
- [Cendol Collection v2](https://huggingface.co/datasets/indonlp/cendol_collection_v2)
- [IndoNLG](https://github.com/indobenchmark/indonlg) - Natural language generation datasets
- [IndoBERTweet](https://github.com/indolem/IndoBERTweet) - Indonesian Twitter corpus
- [ID Clickbait News](https://huggingface.co/datasets/id_clickbait)
- [ID Newspapers 2018](https://huggingface.co/datasets/id_newspapers_2018)

## Citations

```

@inproceedings{abadji-etal-2022-towards,
title = "Towards a Cleaner Document-Oriented Multilingual Crawled Corpus",
author = "Abadji, Julien and others",
booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
year = "2022",
pages = "4344--4355",
}

@inproceedings{wenzek-etal-2020-ccnet,
title = "{CCN}et: Extracting High Quality Monolingual Datasets from Web Crawl Data",
author = "Wenzek, Guillaume and others",
booktitle = "Proceedings of The 12th Language Resources and Evaluation Conference",
year = "2020",
pages = "4003--4012",
}

@misc{kompas2002,
title={Kompas Online Collection},
author={University of Amsterdam},
year={2002},
url={http://ilps.science.uva.nl/resources/bahasa/}
}

@misc{tempo2002,
title={Tempo Online Collection},
author={University of Amsterdam},
year={2002},
url={http://ilps.science.uva.nl/resources/bahasa/}
}

```