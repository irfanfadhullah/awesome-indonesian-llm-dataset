# 🖼️ Multimodal & Vision-Language

This section covers datasets that combine text with visual modalities, including image-text pairs, video captioning, and vision-language understanding tasks for Indonesian.

## Vision-Language Datasets

### Conceptual Captions (Indonesian)

**CC3M Indonesian**
- **Size**: 3M image-caption pairs translated to Indonesian
- **Source**: Google's Conceptual Captions dataset
- **Links**: [Download](https://stor.akmal.dev/cc3m-train.jsonl.zst) | [Original](https://github.com/google-research-datasets/conceptual-captions)

**CC12M Indonesian**
- **Size**: 12M image-caption pairs for vision-language pre-training
- **Purpose**: Large-scale multimodal training
- **Links**: [Download](https://stor.akmal.dev/cc12m.jsonl.zst) | [Original](https://github.com/google-research-datasets/conceptual-12m)

### YFCC100M OpenAI Subset
- **Size**: 14.8M images with natural language descriptions
- **Format**: Image URLs with Indonesian captions
- **Links**: [Download](https://stor.akmal.dev/yfcc100.jsonl.zst) | [Original](https://github.com/openai/CLIP/blob/main/data/yfcc100m.md)

### MSVD-Indonesian
- **Description**: Video-text dataset derived from MSVD
- **Size**: ~80K video-text pairs
- **Tasks**: Text-to-video retrieval, video captioning, video-to-text retrieval
- **Languages**: Indonesian translations of video descriptions
- **Links**: [GitHub Repository](https://github.com/willyfh/msvd-indonesian) | [Data](https://github.com/willyfh/msvd-indonesian/blob/main/data/MSVD-indonesian.txt)
- **Paper**: [arXiv:2306.11341](https://arxiv.org/abs/2306.11341)

**Qualitative Results:**
- Text-to-Video Retrieval benchmarks
- Video-to-Text Retrieval evaluation
- Video Captioning in Indonesian

### KTP VLM Instruct Dataset
- **Description**: Vision-language instruction dataset for Indonesian ID card processing
- **Links**: [HuggingFace](https://huggingface.co/datasets/danielsyahputra/ktp-vlm-instruct-dataset)

## Educational & Professional Assessment

### IndoMMLU
- **Description**: Multi-task language understanding benchmark
- **Size**: 14,906 questions across 63 tasks
- **Levels**: Primary school to university entrance exams
- **Coverage**: 46% Indonesian language + 9 local languages/cultures  
- **Categories**: STEM, Social Science, Humanities, Indonesian Language, Local Languages
- **Links**: [HuggingFace](https://huggingface.co/datasets/indolem/IndoMMLU)

### IndoCareer
- **Description**: Professional certification exam questions
- **Size**: 8,834 multiple-choice questions
- **Sectors**: Healthcare, finance, design, tourism, education, law
- **Focus**: Real-world professional competency evaluation
- **Links**: [HuggingFace](https://huggingface.co/datasets/indolem/IndoCareer) | [Paper](https://arxiv.org/pdf/2409.08564)

### IndoCulture
- **Size**: 2,430 questions with multiple choice answers
- **Content**: Cultural knowledge about Indonesia
- **Purpose**: Evaluating cultural understanding
- **Links**: [HuggingFace](https://huggingface.co/datasets/indolem/IndoCulture)

### IndoCloze
- **Description**: Commonsense story understanding through cloze evaluation
- **Size**: 2,325 Indonesian stories (4-sentence premise + endings)
- **Split**: 1,000 train / 200 dev / 1,135 test
- **Award**: Best Paper Award at CSRR (ACL 2022)
- **Tasks**: Story completion, commonsense reasoning

## Dataset Statistics

| Dataset | Type | Size | Languages | Tasks |
|---------|------|------|-----------|-------|
| CC3M-ID | Image-Text | 3M pairs | Indonesian | Captioning, Retrieval |
| CC12M-ID | Image-Text | 12M pairs | Indonesian | Pre-training |
| MSVD-ID | Video-Text | 80K pairs | Indonesian | Video Understanding |
| IndoMMLU | Text-Only | 14.9K questions | Indonesian + 9 local | QA, Classification |
| IndoCareer | Text-Only | 8.8K questions | Indonesian | Professional Assessment |

## Applications

**Vision-Language Models:**
- Image captioning in Indonesian
- Visual question answering
- Text-to-image retrieval
- Multimodal pre-training

**Educational Assessment:**
- Automated grading systems
- Cultural competency evaluation
- Professional certification testing
- Multilingual education support

## Citations

```

@inproceedings{sharma2018conceptual,
title = {Conceptual Captions: A Cleaned, Hypernymed, Image Alt-text Dataset For Automatic Image Captioning},
author = {Sharma, Piyush and Ding, Nan and Goodman, Sebastian and Soricut, Radu},
booktitle = {Proceedings of ACL},
year = {2018},
}

@article{Hendria2023MSVDID,
title={{MSVD}-{I}ndonesian: A Benchmark for Multimodal Video-Text Tasks in Indonesian},
author={Willy Fitra Hendria},
journal={arXiv preprint arXiv:2306.11341},
year={2023}
}

@inproceedings{koto-etal-2023-indommlu,
title = "Large Language Models Only Pass Primary School Exams in {I}ndonesia: A Comprehensive Test on {I}ndo{MMLU}",
author = "Koto, Fajri and Aisyah, Nurul and Li, Haonan and Baldwin, Timothy",
booktitle = "Proceedings of EMNLP",
year = "2023"
}

@inproceedings{koto2025cracking,
title={Cracking the Code: Multi-domain LLM Evaluation on Real-World Professional Exams in Indonesia},
author={Koto, Fajri},
booktitle={Proceedings of NAACL HLT 2025, Industry Track},
year={2025}
}

```