# 🤖 Pre-trained Models

This section covers pre-trained language models, embeddings, and other machine learning models specifically designed for Indonesian NLP tasks.

## Transformer-based Language Models

### Indo-BERT
- **Description**: Indonesian BERT model for various NLP tasks
- **Architecture**: BERT-base architecture adapted for Indonesian
- **Training Data**: Large Indonesian corpus from IndoNLU
- **Repository**: [GitHub](https://github.com/indobenchmark/indonlu)
- **Models**: [HuggingFace](https://huggingface.co/indobenchmark/indobert-base-p1)
- **Paper**: IndoNLU benchmark paper

**Available Variants:**
- `indobert-base-p1`: Primary Indonesian BERT model
- `indobert-base-p2`: Alternative training configuration
- Cased and uncased versions available

**Performance:**
- State-of-the-art results on IndoNLU benchmark
- Superior performance compared to multilingual models on Indonesian tasks

### Indo-BERTweet
- **Description**: BERT model specifically trained on Indonesian tweets
- **Specialization**: Social media text, informal language, Twitter-specific vocabulary
- **Training**: Indonesian Twitter corpus with careful preprocessing
- **Repository**: [GitHub](https://github.com/indolem/IndoBERTweet)
- **Model**: [HuggingFace](https://huggingface.co/indolem/indobertweet-base-uncased)

**Key Features:**
- Optimized for social media text
- Handles Indonesian Twitter slang and informal expressions
- Better performance on Twitter-based NLP tasks

### Indonesian Language Model Collection
- **Description**: Collection of various transformer models for Indonesian
- **Models**: RoBERTa, DistilBERT, and other transformer variants
- **Links**: [GitHub Repository](https://github.com/cahya-wirawan/indonesian-language-models/tree/master/Transformers)

**Available Models:**
- Indonesian RoBERTa variants
- Indonesian DistilBERT for efficiency
- Various model sizes and configurations

## Word Embeddings

### FastText Indonesian
- **Description**: Pre-trained FastText word vectors for Indonesian
- **Training**: Large Indonesian corpus with subword information
- **Dimension**: 300-dimensional vectors
- **Coverage**: Extensive vocabulary including out-of-vocabulary words
- **Links**: [Official FastText](https://fasttext.cc/docs/en/crawl-vectors.html)

**Features:**
- Subword embeddings for handling morphological variants
- Good coverage of Indonesian vocabulary
- Compatible with standard FastText libraries

### Word2Vec Indonesian
- **Description**: Pre-trained Word2Vec embeddings
- **Training Data**: Indonesian Wikipedia and news corpora
- **Variants**: CBOW and Skip-gram models available
- **Dimensions**: Multiple sizes (100, 200, 300 dimensions)

### Polyglot Embeddings
- **Description**: Multilingual embeddings including Indonesian
- **Coverage**: Indonesian as part of multilingual embedding space
- **Usage**: Cross-lingual applications and transfer learning
- **Links**: [Polyglot Project](https://sites.google.com/site/rmyeid/projects/polyglot)

## Domain-Specific Models

### Indonesian GPT Models
- **IndoGPT**: Generative model for Indonesian text generation
- **Training**: Large-scale Indonesian text corpus
- **Applications**: Text generation, dialogue systems, creative writing
- **Links**: [HuggingFace](https://huggingface.co/indobenchmark/indogpt)

### Indonesian ALBERT
- **Description**: Lightweight version of BERT for Indonesian
- **Efficiency**: Reduced parameters while maintaining performance
- **Training**: Similar corpus to IndoBERT with parameter sharing
- **Links**: Available through IndoNLU benchmark collection

## Multilingual Models with Indonesian Support

### mBERT (Multilingual BERT)
- **Coverage**: 104 languages including Indonesian
- **Usage**: Cross-lingual transfer and multilingual applications
- **Performance**: Good baseline for Indonesian tasks
- **Links**: [HuggingFace](https://huggingface.co/bert-base-multilingual-cased)

### XLM-R (Cross-lingual Language Model - RoBERTa)
- **Coverage**: 100 languages including Indonesian
- **Architecture**: RoBERTa-based multilingual model
- **Performance**: Often outperforms mBERT on Indonesian tasks
- **Links**: [HuggingFace](https://huggingface.co/xlm-roberta-base)

### mT5 (Multilingual T5)
- **Description**: Text-to-text transfer transformer for multiple languages
- **Coverage**: Indonesian included in training languages
- **Applications**: Various text generation and understanding tasks
- **Links**: [HuggingFace](https://huggingface.co/google/mt5-base)

## Model Performance Comparison

### On IndoNLU Benchmark

| Model | Sentiment | NER | POS | QA | Average |
|-------|-----------|-----|-----|----|----|
| IndoBERT | **84.13** | **82.15** | **95.44** | **73.49** | **83.80** |
| mBERT | 81.32 | 78.23 | 93.21 | 69.85 | 80.65 |
| XLM-R | 82.44 | 79.67 | 94.12 | 71.23 | 81.87 |

### Model Sizes and Efficiency

| Model | Parameters | Memory | Inference Speed |
|-------|------------|---------|-----------------|
| IndoBERT-base | 110M | Standard | Baseline |
| Indonesian DistilBERT | 66M | Reduced | 60% faster |
| IndoBERTweet | 110M | Standard | Baseline |
| Indonesian ALBERT | 12M | Small | 80% faster |

## Usage Examples

### Loading IndoBERT
```

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
model = AutoModel.from_pretrained('indobenchmark/indobert-base-p1')

```

### Loading IndoBERTweet
```

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('indolem/indobertweet-base-uncased')
model = AutoModel.from_pretrained('indolem/indobertweet-base-uncased')

```

### Loading Indonesian Word Embeddings
```

import fasttext

# Load FastText model

ft_model = fasttext.load_model('path/to/indonesian/fasttext/model')
word_vector = ft_model.get_word_vector('indonesia')

```

## Citations

```

@inproceedings{indonlu2020,
title = "IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding",
author = "Bryan Wilie and others",
booktitle = "Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing",
year = "2020"
}

@inproceedings{indobertweet2020,
title = "IndoBERTweet: A Pretrained Language Model for Indonesian Twitter with Effective Domain-Specific Vocabulary Initialization",
author = "Fajri Koto and others",
year = "2020"
}

@misc{indonesian-language-models,
title = "Indonesian Language Models",
author = "Cahya Wirawan",
year = "2020",
url = "https://github.com/cahya-wirawan/indonesian-language-models"
}

```