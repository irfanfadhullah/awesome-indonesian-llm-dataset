# 🛠️ Tools & Libraries

This section covers software tools, libraries, and frameworks for Indonesian natural language processing, including preprocessing utilities, analysis tools, and NLP frameworks.

## Core NLP Processing Tools

### Sastrawi Stemmer
- **Description**: Indonesian stemmer library for word stemming
- **Language**: Python, PHP, JavaScript
- **Features**: Rule-based stemming, Indonesian morphology handling
- **Links**: [GitHub Repository](https://github.com/sastrawi/sastrawi)

**Key Features:**
- Removes prefixes, suffixes, and infixes
- Handles Indonesian morphological variations
- Multiple programming language support
- High accuracy for Indonesian text

**Usage Example:**
```

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()
sentence = "Perekonomian Indonesia sedang mengalami pertumbuhan"
stemmed = stemmer.stem(sentence)

```

### Pujangga REST API
- **Description**: Comprehensive Indonesian NLP API service
- **Features**: Multiple NLP tasks in single API
- **Links**: [GitHub Repository](https://github.com/panggi/pujangga)

**Supported Tasks:**
- Sentence segmentation
- Tokenization
- Part-of-speech tagging
- Named entity recognition
- Sentiment analysis

### NLP-ID
- **Description**: Indonesian NLP toolkit with various utilities
- **Developer**: Kumparan engineering team
- **Links**: [GitHub Repository](https://github.com/kumparan/nlp-id)

**Features:**
- Text preprocessing utilities
- Indonesian-specific normalization
- Sentiment analysis tools
- Named entity extraction

## Morphological Analysis Tools

### MorphInd
- **Description**: Indonesian morphological analyzer
- **Capabilities**: Detailed morphological parsing
- **Developer**: Academic research project
- **Links**: [Website](http://septinalarasati.com/morphind/)

**Features:**
- Morpheme segmentation
- Part-of-speech analysis
- Root word identification
- Morphological feature extraction

### INDRA
- **Description**: Indonesian dependency parser
- **Task**: Syntactic parsing and dependency analysis
- **Links**: [GitHub Repository](https://github.com/davidmoeljadi/INDRA)

**Capabilities:**
- Dependency tree generation
- Syntactic role identification
- Indonesian grammatical analysis

## Text Processing and Correction

### Indonesian Typo Checker
- **Description**: Spelling correction tool for Indonesian
- **Features**: Typo detection and correction suggestions
- **Links**: [GitHub Repository](https://github.com/mamat-rahmat/checker_id)

**Functionality:**
- Indonesian spelling validation
- Correction suggestions
- Dictionary-based verification

### Text Normalization Tools
- Various libraries for Indonesian text normalization
- Handle informal language, slang, and social media text
- Convert abbreviated forms to standard language

## Framework Integration

### Flair NLP
- **Description**: NLP framework with Indonesian language support
- **Features**: State-of-the-art NLP models
- **Links**: [GitHub Repository](https://github.com/flairNLP/flair)

**Indonesian Support:**
- Pre-trained Indonesian models
- Named entity recognition
- Part-of-speech tagging
- Text classification

**Usage Example:**
```

from flair.data import Sentence
from flair.models import SequenceTagger

# Load Indonesian NER model

tagger = SequenceTagger.load('ner-indonesian')

# Make prediction

sentence = Sentence('Saya tinggal di Jakarta, Indonesia.')
tagger.predict(sentence)

```

### spaCy Bahasa Indonesia
- **Description**: spaCy extension for Indonesian language processing
- **Features**: Industrial-strength NLP for Indonesian
- **Links**: [spaCy Official](https://github.com/explosion/spaCy)
- **Tutorial**: [spaCy Bahasa Indonesia Guide](https://bagas.me/spacy-bahasa-indonesia.html)

**Capabilities:**
- Tokenization
- Part-of-speech tagging
- Named entity recognition
- Dependency parsing
- Word vectors

## Research and Experimental Tools

### nlp-experiments
- **Description**: Collection of Indonesian NLP experiments
- **Content**: Various research implementations
- **Links**: [GitHub Repository](https://github.com/yohanesgultom/nlp-experiments)

### python-sentianalysis-id
- **Description**: Indonesian sentiment analysis toolkit
- **Features**: Sentiment classification tools
- **Links**: [GitHub Repository](https://github.com/yasirutomo/python-sentianalysis-id)

### Analisis-Sentimen-ID
- **Description**: Indonesian sentiment analysis implementation
- **Links**: [GitHub Repository](https://github.com/riochr17/Analisis-Sentimen-ID)

### indonesia-ner
- **Description**: Named entity recognition for Indonesian
- **Links**: [GitHub Repository](https://github.com/yusufsyaifudin/indonesia-ner)

## Creative Text Generation

### Indonesian Poetry & Pantun Generator
- **Description**: Generator for Indonesian traditional poetry
- **Content**: Pantun and puisi generation tools
- **Links**: [GitHub Repository](https://github.com/ilhamfp/puisi-pantun-generator)

**Features:**
- Traditional Indonesian poetry patterns
- Rhyme scheme preservation
- Cultural authenticity

## Data Collection Tools

### GetOldTweets3
- **Description**: Twitter scraping without API requirements
- **Use Case**: Collecting Indonesian Twitter data
- **Links**: [GitHub Repository](https://github.com/Mottl/GetOldTweets3)

**Features:**
- Historical tweet collection
- No API limitations
- Flexible search parameters

### Tweepy
- **Description**: Twitter API wrapper for Python
- **Use Case**: Indonesian social media data collection
- **Documentation**: [Tweepy Docs](http://docs.tweepy.org/en/latest/)
- **Tutorial**: [Tweet Scraping Guide](https://towardsdatascience.com/how-to-scrape-tweets-from-twitter-59287e20f0f1)

## Spelling and Grammar Tools

### Norvig's Spell Corrector (Indonesian)
- **Description**: Indonesian adaptation of Norvig's spelling corrector
- **Method**: Statistical approach to spelling correction
- **Original**: [Norvig's Algorithm](https://norvig.com/spell-correct.html)

**Approach:**
- Statistical language model
- Edit distance calculations
- Frequency-based corrections

## Installation and Setup

### Basic Installation
```


# Install Sastrawi stemmer

pip install Sastrawi

# Install spaCy with Indonesian model

pip install spacy
python -m spacy download id_core_news_sm

# Install Flair

pip install flair

# Install other common tools

pip install tweepy
pip install fasttext

```

### Usage Examples

**Complete NLP Pipeline:**
```
# Indonesian NLP pipeline using multiple tools

import sastrawi
import flair
from indonlp import tokenize, normalize

def indonesian_nlp_pipeline(text):
\# 1. Text normalization
normalized = normalize(text)

    # 2. Tokenization
    tokens = tokenize(normalized)
    
    # 3. Stemming
    stemmer = sastrawi.create_stemmer()
    stemmed = [stemmer.stem(token) for token in tokens]
    
    # 4. NER
    sentence = flair.Sentence(normalized)
    ner_tagger.predict(sentence)
    
    return {
        'normalized': normalized,
        'tokens': tokens,
        'stemmed': stemmed,
        'entities': sentence.get_spans('ner')
    }
```

## Performance and Scalability

### Tool Performance Comparison

| Tool | Speed | Accuracy | Memory | Use Case |
|------|-------|----------|---------|----------|
| Sastrawi | Fast | High | Low | Stemming |
| Flair | Medium | Very High | High | Complete NLP |
| spaCy | Fast | High | Medium | Production |
| MorphInd | Slow | Very High | Medium | Research |

### Production Deployment

**High-Performance Setup:**
- spaCy for production systems
- Flair for research applications
- Custom APIs for specific tasks

**Resource Requirements:**
- CPU: Most tools are CPU-optimized
- Memory: 1-4GB depending on models
- Storage: Model files 100MB-1GB

## Citations

```
@misc{sastrawi2016,
title={Sastrawi: Indonesian Stemmer},
author={Sastrawi Contributors},
year={2016},
url={https://github.com/sastrawi/sastrawi}
}

@misc{pujangga2019,
title={Pujangga: Indonesian NLP REST API},
author={Pujangga Contributors},
year={2019},
url={https://github.com/panggi/pujangga}
}

@misc{nlpid2020,
title={NLP-ID: Indonesian NLP Toolkit},
author={Kumparan Engineering Team},
year={2020},
url={https://github.com/kumparan/nlp-id}
}
```