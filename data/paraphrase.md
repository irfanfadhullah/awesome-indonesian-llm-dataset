# Paraphrase Dataset
Source of this information: https://github.com/Wikidepia/indonesian_datasets

## The Multi-Genre Natural Language Inference

[[Original Paper](https://arxiv.org/abs/1704.05426)]

[[Original Dataset](https://cims.nyu.edu/~sbowman/multinli)]

[[Update Link for Download](https://stor.akmal.dev/idmultinli/)]


The Multi-Genre Natural Language Inference (MultiNLI) corpus is a crowd-sourced collection of 433k sentence pairs annotated with textual entailment information. The corpus is modeled on the SNLI corpus, but differs in that covers a range of genres of spoken and written text, and supports a distinctive cross-genre generalization evaluation. The corpus served as the basis for the shared task of the RepEval 2017 Workshop at EMNLP in Copenhagen.

### Citation

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

## ParaBank

[[Original Dataset](https://nlp.jhu.edu/parabank/)] 

[[Update Link for Download](https://stor.akmal.dev/parabank-v2.0.jsonl.zst)]

The ParaBank project consists of a series of efforts exploring the potential for guided backtranslation for the purpose of paraphrasing with constraints. This work is spiritually connected to prior efforts at JHU in paraphrasing, in particular projects surrounding the ParaPhrase DataBase (PPDB).

### Citations

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
    abstract = "Producing diverse paraphrases of a sentence is a challenging task. Natural paraphrase corpora are scarce and limited, while existing large-scale resources are automatically generated via back-translation and rely on beam search, which tends to lack diversity. We describe ParaBank 2, a new resource that contains multiple diverse sentential paraphrases, produced from a bilingual corpus using negative constraints, inference sampling, and clustering.We show that ParaBank 2 significantly surpasses prior work in both lexical and syntactic diversity while being meaning-preserving, as measured by human judgments and standardized metrics. Further, we illustrate how such paraphrastic resources may be used to refine contextualized encoders, leading to improvements in downstream tasks.",
}
```

## ParaNMT

[[Original Dataset](http://www.cs.cmu.edu/~jwieting/)] 

[[Update Link for Download](https://stor.akmal.dev/paranmt-5m.jsonl.zst)]

We describe PARANMT-50M, a dataset of more than 50 million English-English sentential paraphrase pairs. We generated the pairs automatically by using neural machine translation to translate the non-English side of a large parallel corpus, following Wieting et al. (2017). Our hope is that ParaNMT-50M can be a valuable resource for paraphrase generation and can provide a rich source of semantic knowledge to improve downstream natural language understanding tasks. To show its utility, we use ParaNMT-50M to train paraphrastic sentence embeddings that outperform all supervised systems on every SemEval semantic textual similarity competition, in addition to showing how it can be used for paraphrase generation.

### Citations

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
    abstract = "We describe ParaNMT-50M, a dataset of more than 50 million English-English sentential paraphrase pairs. We generated the pairs automatically by using neural machine translation to translate the non-English side of a large parallel corpus, following Wieting et al. (2017). Our hope is that ParaNMT-50M can be a valuable resource for paraphrase generation and can provide a rich source of semantic knowledge to improve downstream natural language understanding tasks. To show its utility, we use ParaNMT-50M to train paraphrastic sentence embeddings that outperform all supervised systems on every SemEval semantic textual similarity competition, in addition to showing how it can be used for paraphrase generation.",
}
```


## Paraphrase Adversaries from Word Scrambling

[[Original Paper](https://arxiv.org/abs/1904.01130)] 

[[Original Dataset](https://github.com/google-research-datasets/paws)] 

[[Dataset Download](https://github.com/Wikidepia/indonesia_dataset/tree/master/paraphrase/paws/data)]

This dataset contains 100k human-labeled pairs that feature the importance of modeling structure, context, and word order information for the problem of paraphrase identification.

All translated pairs are sourced from examples in [PAWS-Wiki](https://github.com/google-research-datasets/paws#paws-wiki).

- [`PAWS-Wiki Labeled (Final)`](https://github.com/Wikidepia/indonesia_dataset/tree/master/paraphrase/PAWS/data/final): containing pairs that are generated from both word swapping and back translation methods. All pairs have human judgements on both paraphrasing and fluency and they are split into Train/Dev/Test sections.

- [`PAWS-Wiki Labeled (Swap-only)`](https://github.com/Wikidepia/indonesia_dataset/tree/master/paraphrase/PAWS/data/swap): containing pairs that have no back translation counterparts and therefore they are not included in the first set. Nevertheless, they are high-quality pairs with human judgements on both paraphrasing and fluency, and they can be included as an auxiliary training set.

Translated to Indonesia using Google Translate API. Translate script is included.

### Citation

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

## SBERT Paraphrase Data

[[Dataset Download](https://storage.depia.wiki/sbert-paraphrase/)]

Various paraphrase datasets compiled by [SBERT](https://www.sbert.net/examples/training/paraphrases/README.html) translated to Indonesian.

**NOTE** : You might need to cleanup some data to use this dataset.

## The Stanford Natural Language Inference (SNLI) Corpus

[[Original Paper](https://arxiv.org/abs/1508.05326)] 

[[Original Dataset](https://nlp.stanford.edu/projects/snli/)] 

[[Update Link for Download](https://stor.akmal.dev/idsnli/)]

The SNLI corpus (version 1.0) is a collection of 570k human-written English sentence pairs manually labeled for balanced classification with the labels entailment, contradiction, and neutral, supporting the task of natural language inference (NLI), also known as recognizing textual entailment (RTE).

### Citation

```bibtex
@inproceedings{snli:emnlp2015,
	Author = {Bowman, Samuel R. and Angeli, Gabor and Potts, Christopher, and Manning, Christopher D.},
	Booktitle = {Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
	Publisher = {Association for Computational Linguistics},
	Title = {A large annotated corpus for learning natural language inference},
	Year = {2015}
}
```
