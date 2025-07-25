# Translation Dataset
Source of this information: https://github.com/Wikidepia/indonesian_datasets


## Europarl

[[Dataset Download](https://storage.depia.wiki/europarl.jsonl.zst)] [[Original Paper](https://aclanthology.org/2005.mtsummit-papers.11//)]

The Europarl parallel corpus is extracted from the proceedings of the European Parliament. It includes versions in 21 European languages: Romanic (French, Italian, Spanish, Portuguese, Romanian), Germanic (English, Dutch, German, Danish, Swedish), Slavik (Bulgarian, Czech, Polish, Slovak, Slovene), Finni-Ugric (Finnish, Hungarian, Estonian), Baltic (Latvian, Lithuanian), and Greek.

### Citation

@inproceedings{koehn-2005-europarl,
    title = "{E}uroparl: A Parallel Corpus for Statistical Machine Translation",
    author = "Koehn, Philipp",
    booktitle = "Proceedings of Machine Translation Summit X: Papers",
    month = sep # " 13-15",
    year = "2005",
    address = "Phuket, Thailand",
    url = "https://aclanthology.org/2005.mtsummit-papers.11",
    pages = "79--86",
    abstract = "We collected a corpus of parallel text in 11 languages from the proceedings of the European Parliament, which are published on the web. This corpus has found widespread use in the NLP community. Here, we focus on its acquisition and its application as training data for statistical machine translation (SMT). We trained SMT systems for 110 language pairs, which reveal interesting clues into the challenges ahead.",
}


## EuroPat v2

[[Dataset Download](https://storage.depia.wiki/europat.jsonl.zst)] 

[[Original Dataset](https://europat.net/)]

Parallel corpora of patents from the United States Patent and Trademark Office and from the European Patent Organisation compiled into aligned data sets available from https://europat.net/


## ParaCrawl

[[Dataset Download](https://huggingface.co/datasets/Wikidepia/IndoParaCrawl)] 

[[Original Paper](https://www.aclweb.org/anthology/W19-6721/)]

ParaCrawl v.7.1 is a parallel dataset with 41 language pairs primarily aligned with English (39 out of 41) and mined using the parallel-data-crawling tool Bitextor which includes downloading documents, preprocessing and normalization, aligning documents and segments, and filtering noisy data via Bicleaner. ParaCrawl focuses on European languages, but also includes 9 lower-resource, non-European language pairs in v7.1.

### Download

To download IndoParaCrawl you will need `git lfs`.

```bash
git lfs install
git clone https://huggingface.co/datasets/Wikidepia/IndoParaCrawl
```

### Citation

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
