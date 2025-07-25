# Crawl Dataset
Source of this information: https://github.com/Wikidepia/indonesian_datasets

## Kaskus WebText

[[Update Link for Download](https://stor.akmal.dev/kaskus-webtext.tar.zst)]


[[List of the Source](https://stor.akmal.dev/kaskus.jsonl.zst)]


Scrape URLs from Kaskus (Starter only), filter to 3 or more cendol (karma). There are only 7000 URLs on Kaskus with 3 or more cendol.

## Twitter Puisi

This dataset contains loosely filtered poem from various user on Twitter. 

| User                                             | Poem |
| ------------------------------------------------ | ---- |
| [PelangiPuisi](https://twitter.com/PelangiPuisi) | 80k+ |


# Wikipedia Links

[[Update Link for Download](https://stor.akmal.dev/wikipedia-links.tar.zst)]

Wikipedia have a lot of references & citations from internet. It should contain some high quality web content, this dataset contains 58k urls from Indonesian Wikipedia external links dump.

You will need to install [lm_dataformat](https://pypi.org/project/lm-dataformat/) to use it.

```python
from lm_dataformat import Reader

rdr = Reader('wikipedia-links')

for doc in rdr.stream_data():
    print(doc)
```


## Indonesian CommonCrawl News

[[Update Link for Download](https://stor.akmal.dev/ccnews-id.tar)]

Processed CommonCrawl News Dataset filtered to Indonesian Language using CLD2. Content extracted using Trafilatura.

You will need to use [lm_dataformat](https://github.com/leogao2/lm_dataformat) to load this data.


## Javanese Text

[[Update Link for Download](https://stor.akmal.dev/jv-text/)]

Extracting Javanese text from available data. The data used comes from:

- CC100
- [data.statmt.org](http://data.statmt.org/ngrams/raw/)
- Wikipedia

### Process

1. Detect language with CLD3
2. Score sentence with KenLM with Javanese Wikipedia
3. Dedupe with simple awk

### Citations

```bibtex
@inproceedings{wenzek2020ccnet,
  title={CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data},
  author={Wenzek, Guillaume and Lachaux, Marie-Anne and Conneau, Alexis and Chaudhary, Vishrav and Guzm{\'a}n, Francisco and Joulin, Armand and Grave, {\'E}douard},
  booktitle={Proceedings of The 12th Language Resources and Evaluation Conference},
  pages={4003--4012},
  year={2020}
}
```

## "Clean" mC4

[[Original Dataset](https://github.com/allenai/allennlp/discussions/5265)]]

Indonesian mC4 with filtering from spam, NSFW, and Gambling. Big thanks to AllenAI for releasing Multilingual C4 🙏.

### Filtering Steps

1. Remove text from CCAligned domain (Why? CCAligned sometimes contains machine translated website.)
2. Find porn-y and gambling words, if it exceeds certain threshold it wont be included.
3. Train fasttext clasifier from (NSFW & Gambling) and (Indonesian CC-NEWS) text and classify text with it.

### Citations

```bibtex
@article{Raffel2020ExploringTL,
  title={Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer},
  author={Colin Raffel and Noam M. Shazeer and Adam Roberts and Katherine Lee and Sharan Narang and Michael Matena and Yanqi Zhou and W. Li and Peter J. Liu},
  journal={ArXiv},
  year={2020},
  volume={abs/1910.10683}
}
```

## Twitter Crawl

[[Update Link for Download](https://stor.akmal.dev/twitter-dump/)]

This is a raw dataset, formatted as JSON line. Contains user data, tweet, etc.
