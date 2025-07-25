# Summarization Dataset
Source of this information: https://github.com/Wikidepia/indonesian_datasets


## GigaWord

[[Update Link for Download](https://stor.akmal.dev/gigaword.tar.zst)]

Headline-generation on a corpus of article pairs from Gigaword consisting of around 4 million articles. Use the 'org_data' provided by https://github.com/microsoft/unilm/ which is identical to https://github.com/harvardnlp/sent-summary but with better format.

### Citations

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


## Reddit TLDR

[[Original Paper](https://www.aclweb.org/anthology/W17-4508/)] 

[[Update Link for Download](https://stor.akmal.dev/reddit-tldr.jsonl.zst)]

The Webis TLDR Corpus (2017) consists of approximately 4 Million content-summary pairs extracted for Abstractive Summarization, from the Reddit dataset for the years 2006-2016. This corpus is first of its kind from the social media domain in English and has been created to compensate the lack of variety in the datasets used for abstractive summarization research using deep learning models.

### Caution

Translation might seem a bit bad because of Google Translate character limit. So I need to split paragraph to sentences and merge them together.

### Citation

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
    abstract = "Recent advances in automatic text summarization have used deep neural networks to generate high-quality abstractive summaries, but the performance of these models strongly depends on large amounts of suitable training data. We propose a new method for mining social media for author-provided summaries, taking advantage of the common practice of appending a {``}TL;DR{''} to long posts. A case study using a large Reddit crawl yields the Webis-TLDR-17 dataset, complementing existing corpora primarily from the news genre. Our technique is likely applicable to other social media sites and general web crawls.",
}
```


## WikiHow

[[Original Paper](https://arxiv.org/abs/1810.09305)] 

[[Original Code](https://github.com/mahnazkoupaee/WikiHow-Dataset)] 

[[Update Link for Download](https://stor.akmal.dev/wikihow.json.zst)]

This dataset contains Headline and Text from WikiHow Indonesia. Using the same data processing as original WikiHow dataset.

### Citation

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
