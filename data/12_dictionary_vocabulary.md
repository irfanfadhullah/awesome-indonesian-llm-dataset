# 📖 Dictionary & Vocabulary

This section covers lexical resources including dictionaries, word lists, sentiment lexicons, and linguistic resources for Indonesian language processing.

## Core Vocabulary Resources

### Indonesia Wordlist
- **Description**: Comprehensive Indonesian vocabulary from official dictionary
- **Size**: 105,226 words from Kamus Besar Bahasa Indonesia (KBBI)
- **Coverage**: Most common Indonesian vocabulary
- **Links**: [GitHub Repository](https://github.com/Wikidepia/indonesian_datasets/tree/master/dictionary/wordlist/data/wordlist.txt)

### Colloquial Indonesian Lexicon
- **Description**: Mapping of colloquial/slang words to formal Indonesian
- **Size**: 3,592 colloquial tokens → 1,742 lemmas
- **Purpose**: Text normalization for informal language processing
- **Links**: [GitHub Repository](https://github.com/nasalsabila/kamus-alay)
- **Paper**: [IEEE 2018](https://ieeexplore.ieee.org/abstract/document/8629151)

## Semantic Lexical Resources

### Tesaurus
- **Description**: Indonesian thesaurus with synonym, antonym, and word relationships
- **Content**: Semantic relations between Indonesian words
- **Links**: [GitHub Repository](https://github.com/victoriasovereigne/tesaurus)

### WordNet Bahasa
- **Description**: Indonesian-Malay semantic dictionary based on Princeton WordNet
- **Structure**: Synsets, hypernyms, hyponyms, and semantic relations
- **Links**: [SourceForge](https://sourceforge.net/p/wn-msa/tab/HEAD/tree/trunk/)

### id-wordnet npm package
- **Description**: JavaScript/Node.js package for Indonesian WordNet
- **Links**: [NPM Package](https://www.npmjs.com/package/id-wordnet)

### KAWAT (Word Analogy Dataset)
- **Description**: Indonesian word analogy dataset for semantic and syntactic relations
- **Content**: Word pairs demonstrating analogical relationships
- **Links**: [GitHub Repository](https://github.com/kata-ai/kawat)

## Formal-Informal Language Mapping

### STIF-Indonesia
- **Description**: Dataset for Indonesian formal-informal style conversion
- **Links**: [GitHub Repository](https://github.com/haryoa/stif-indonesia)

### IndoCollex
- **Description**: Comprehensive lexical collection for formal-informal mapping
- **Links**: [GitHub Repository](https://github.com/haryoa/indo-collex)

### Indonesian Slang Mapping
- **Description**: Large dictionary mapping slang/alay words to formal Indonesian
- **Links**: [CSV File](https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection/blob/master/new_kamusalay.csv)

## Sentiment Lexicons

### Comprehensive Sentiment Collections

**Negative Sentiment Words:**
- [negatif_ta2.txt](https://github.com/ramaprakoso/analisis-sentimen/blob/master/kamus/negatif_ta2.txt)
- [negative_add.txt](https://github.com/ramaprakoso/analisis-sentimen/blob/master/kamus/negative_add.txt)
- [negative_keyword.txt](https://github.com/ramaprakoso/analisis-sentimen/blob/master/kamus/negative_keyword.txt)
- [ID-OpinionWords negative](https://github.com/masdevid/ID-OpinionWords/blob/master/negative.txt)

**Positive Sentiment Words:**
- [positif_ta2.txt](https://github.com/ramaprakoso/analisis-sentimen/blob/master/kamus/positif_ta2.txt)
- [positive_add.txt](https://github.com/ramaprakoso/analisis-sentimen/blob/master/kamus/positive_add.txt)
- [positive_keyword.txt](https://github.com/ramaprakoso/analisis-sentimen/blob/master/kamus/positive_keyword.txt)
- [ID-OpinionWords positive](https://github.com/masdevid/ID-OpinionWords/blob/master/positive.txt)

**Score-based Lexicons:**
- [SentiStrengthID](https://github.com/agusmakmun/SentiStrengthID/blob/master/id_dict/sentimentword.txt)
- [InSet Lexicon](https://github.com/fajri91/InSet)
- [HuggingFace senti_lex](https://huggingface.co/datasets/senti_lex)

## Morphological Resources

### Root Words
- [rootword.txt](https://github.com/agusmakmun/SentiStrengthID/blob/master/id_dict/rootword.txt)
- [kata-dasar.original.txt](https://github.com/sastrawi/sastrawi/blob/master/data/kata-dasar.original.txt)
- [kata-dasar.txt](https://github.com/sastrawi/sastrawi/blob/master/data/kata-dasar.txt)
- [kamus-kata-dasar.csv](https://github.com/prasastoadi/serangkai/blob/master/serangkai/kamus/data/kamus-kata-dasar.csv)
- [Combined root words](https://github.com/louisowen6/NLP_bahasa_resources/blob/master/combined_root_words.txt)

### Slang Words
- [kbba.txt](https://github.com/ramaprakoso/analisis-sentimen/blob/master/kamus/kbba.txt)
- [slangword.txt](https://github.com/agusmakmun/SentiStrengthID/blob/master/id_dict/slangword.txt)
- [formalizationDict.txt](https://github.com/panggi/pujangga/blob/master/resource/formalization/formalizationDict.txt)
- [Combined slang words](https://github.com/louisowen6/NLP_bahasa_resources/blob/master/combined_slang_words.txt)

### Stopwords
- [stopwordsID.txt](https://github.com/yasirutomo/python-sentianalysis-id/blob/master/data/feature_list/stopwordsID.txt)
- [stopword.txt](https://github.com/ramaprakoso/analisis-sentimen/blob/master/kamus/stopword.txt)
- [elang stopwords-list](https://github.com/abhimantramb/elang/tree/master/word2vec/utils/stopwords-list)
- [Combined stopwords](https://github.com/louisowen6/NLP_bahasa_resources/blob/master/combined_stop_words.txt)

## Expression and Communication

### Emoticon & Emoji
- [emoticon.txt](https://github.com/ramaprakoso/analisis-sentimen/blob/master/kamus/emoticon.txt)
- [Emoji synonyms Indonesian](https://github.com/jolicode/emoji-search/blob/master/synonyms/cldr-emoji-annotation-synonyms-id.txt)
- [SentiStrengthID Emoticon](https://github.com/agusmakmun/SentiStrengthID/blob/master/id_dict/emoticon.txt)

### Acronyms
- [ramaprakoso acronyms](https://github.com/ramaprakoso/analisis-sentimen/blob/master/kamus/acronym.txt)
- [pujangga acronyms](https://github.com/panggi/pujangga/blob/master/resource/sentencedetector/acronym.txt)
- [Wiktionary Indonesian Acronyms](https://id.wiktionary.org/wiki/Lampiran:Daftar_singkatan_dan_akronim_dalam_bahasa_Indonesia#A)

## Geographic and Administrative Data

### Indonesia Regions & Administrative Data
- [elang region](https://github.com/abhimantramb/elang/blob/master/word2vec/utils/indonesian-region.txt)
- [Wilayah Administratif Indonesia CSV](https://github.com/edwardsamuel/Wilayah-Administratif-Indonesia/tree/master/csv)
- [Indonesia Postal Code CSV](https://github.com/pentagonal/Indonesia-Postal-Code/tree/master/Csv)
- [Country list](https://github.com/panggi/pujangga/blob/master/resource/netagger/contextualfeature/country.txt)
- [Region list](https://github.com/panggi/pujangga/blob/master/resource/netagger/contextualfeature/lpre.txt)

### Named Entity Resources
- [Title of Name](https://github.com/panggi/pujangga/blob/master/resource/netagger/contextualfeature/ppre.txt)
- [Organization titles](https://github.com/panggi/pujangga/blob/master/resource/reference/opre.txt)
- [Gender by Name dataset](https://github.com/seuriously/genderprediction/blob/master/namatraining.txt)

## Resource Statistics

| Resource Type | Count | Coverage | Quality |
|---------------|-------|----------|---------|
| Basic Vocabulary | 105K words | High | Official |
| Colloquial Mapping | 3.6K tokens | Medium | Curated |
| Sentiment Words | Multiple sets | Good | Community |
| Root Words | Multiple sources | High | Linguistic |
| Administrative | Complete | Full | Official |

## Applications

**Text Preprocessing:**
- Normalization of informal language
- Sentiment analysis preprocessing
- Named entity recognition

**Linguistic Analysis:**
- Morphological analysis
- Semantic similarity computation
- Text classification features

**Information Extraction:**
- Location extraction
- Person name recognition
- Organization identification

## Integration Examples

**Text Normalization Pipeline:**
```


# Load colloquial lexicon

colloquial_dict = load_colloquial_mapping()

# Load slang words

slang_dict = load_slang_mapping()

# Normalize text

def normalize_text(text):
\# Apply colloquial mapping
\# Apply slang normalization
\# Remove stopwords
return normalized_text

```

**Sentiment Analysis Enhancement:**
```


# Load sentiment lexicons

positive_words = load_positive_lexicon()
negative_words = load_negative_lexicon()

# Apply sentiment scoring

def sentiment_score(text):
\# Count positive/negative words
\# Apply emoticon mapping
return sentiment_score

```

## Data Quality and Maintenance

**High-Quality Resources:**
- Official KBBI vocabulary
- Manually curated mappings
- Linguistically validated resources

**Community-Maintained:**
- Sentiment lexicons
- Slang mappings
- Emoticon dictionaries

**Usage Recommendations:**
- Combine multiple sources for robustness
- Regular updates for slang and colloquial terms
- Validation against current usage patterns

## Citations

```

@inproceedings{nasalsabila2018,
title={Colloquial Indonesian Lexicon},
author={Nasalsabila, Nasal and others},
booktitle={2018 International Conference on Asian Language Processing (IALP)},
year={2018},
organization={IEEE}
}

@misc{kawat2019,
title={KAWAT: Word Analogy Dataset for Indonesian},
author={Kata.ai},
year={2019},
url={https://github.com/kata-ai/kawat}
}

@misc{tesaurus2019,
title={Indonesian Thesaurus},
author={Victoria Sovereignty},
year={2019},
url={https://github.com/victoriasovereigne/tesaurus}
}

```