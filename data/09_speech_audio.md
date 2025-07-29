# 🎙️ Speech & Audio

This section covers speech recognition datasets, text-to-speech resources, and audio corpora for Indonesian and local languages.

## Speech Recognition Datasets

### TITML-IDN Speech Corpus

- **Description**: High-quality phonetically balanced Indonesian speech corpus
- **Size**: 20 speakers (11 male, 9 female), 343 utterances each
- **Quality**: Phonetically balanced for comprehensive phoneme coverage
- **Access**: Academic/non-commercial use only (formal request required)
- **Links**: [NII Research](http://research.nii.ac.jp/src/en/TITML-IDN.html)
- **Request Process**: [Registration](http://research.nii.ac.jp/src/en/register.html)

**Technical Details:**

- Professional recording quality
- Balanced phonetic distribution
- Suitable for ASR model training


### LUMINA (Linguistic Unified Multimodal Indonesian Natural Audio-Visual)

- **Description**: Constrained audio-visual dataset for speech perception research- **Size**: 14 native speakers (9 male, 5 female), ~1,000 sentences each- **Total**: Approximately 14,000 utterances
- **Quality**: High-quality facial recordings with controlled environment- **Access**: Open access via Creative Commons Attribution 4.0 International- **Links**: [Mendeley Data](https://data.mendeley.com/datasets/8fw93k4rny/4)- **DOI**: 10.17632/8fw93k4rny.4

**Technical Details:**
- **Audio Format**: .wav files, 16kHz sampling rate
- **Video Format**: .mp4 files, CRF28 compression
- **Video Specifications**: 250×150 pixels, cropped and centered on mouth
- **Duration**: 3.3 seconds per clip
- **Recording Environment**: Soundproof room with controlled lighting
- **Applications**: Lip reading, speech synthesis, face processing


### Mozilla Common Voice (Indonesian)

- **Description**: Crowdsourced open speech recognition dataset
- **Content**: Community-contributed voice recordings
- **Links**: [HuggingFace](https://huggingface.co/datasets/common_voice)


### CoVoST2

- **Description**: Cross-lingual speech translation dataset
- **Coverage**: Multiple language pairs including Indonesian
- **Links**: [HuggingFace](https://huggingface.co/datasets/covost2)


### CMU Wilderness Multilingual Speech Dataset

- **Coverage**: 700+ languages including Indonesian
- **Source**: Bible recordings from bible.is
- **Content**: Transcribed audio with alignments
- **Links**: [GitHub Repository](https://github.com/festvox/datasets-CMU_Wilderness)


## Regional Language Speech Datasets

### Large Javanese ASR Dataset

- **Description**: Transcribed audio data for Javanese ASR training
- **Size**: 52,000+ utterances
- **Collaboration**: Google with Reykjavik University and Universitas Gadjah Mada
- **Quality**: Manually quality checked
- **Links**: [OpenSLR](https://openslr.org/35/)


### Large Sundanese ASR Dataset

- **Description**: Sundanese speech recognition training data
- **Links**: [OpenSLR](https://openslr.org/36/)


### VoxLingua107 Dataset

- **Description**: Multilingual spoken language identification dataset
- **Coverage**: 107 languages including Indonesian and local languages
- **Purpose**: Training spoken language identification models
- **Links**:
    - [Indonesian](https://cs.taltech.ee/staff/tanel.alumae/data/voxlingua107/id.zip)
    - [Javanese](https://cs.taltech.ee/staff/tanel.alumae/data/voxlingua107/jw.zip)
    - [Sundanese](https://cs.taltech.ee/staff/tanel.alumae/data/voxlingua107/su.zip)


## Accent and Dialect Datasets

### Indonesian Speech with Regional Accents

- **Coverage**: 5 ethnic groups (Batak, Malay, Javanese, Sundanese, Papuan)
- **Content**: Standardized Indonesian text (320 words) spoken by different ethnic groups
- **Age Range**: Speakers aged 17-50 years
- **Format**: Mobile phone recordings + derived MFCC files
- **Links**: [Kaggle Dataset](https://www.kaggle.com/datasets/hengkymulyono/indonesian-speech-with-accents-5-ethnic-groups)

**Applications:**

- Accent recognition systems
- Dialectal variation studies
- Robust ASR development


## Multimodal and Audio-Visual Datasets

### LUMINA Dataset Applications

The LUMINA dataset serves multiple research purposes beyond traditional ASR:
**Lip Reading Research:**

- Visual speech recognition from silent video
- Cross-modal speech synthesis
- Audio-visual speech enhancement

**Face Processing:**

- Facial expression analysis during speech
- Visual feature extraction for speech tasks
- Computer vision applications in speech technology


## Synthetic and Generated Speech

### Google TTS Dataset

- **Description**: Automatically generated speech using Google Translate TTS
- **Source**: Indonesian newspaper titles from id_newspapers_2018
- **Size**: 500K utterances
- **Links**: [Download](https://stor.akmal.dev/gtts-500k.zip)
- **Format**: Audio files with base64-encoded text in filenames


### Indonesian Unsupervised Speech Dataset

- **Size**: 260GB total (170GB podcast + 90GB YouTube)
- **Access**: Contact akmal@depia.wiki
- **Usage**: Research purposes only
- **Disclaimer**: Content ownership belongs to original creators

**Content Sources:**

- Indonesian podcast recordings
- YouTube channel audio


## Educational and Research Datasets

### Indonesian Speech Recognition (Small)

- **Size**: 50 utterances by single male speaker
- **Links**: [GitHub Repository](https://github.com/frankydotid/Indonesian-Speech-Recognition)
- **Note**: School project - not recommended for production use
- **Disclaimer**: Author not responsible for results


## Dataset Statistics

| Dataset | Size | Languages | Quality | Access | Modality |
| :-- | :-- | :-- | :-- | :-- | :-- |
| TITML-IDN | 6,860 utterances | Indonesian | Professional | Academic Request | Audio |
| LUMINA | 14,000 utterances | Indonesian | High-quality AV | Open | Audio-Visual |
| Javanese ASR | 52K+ utterances | Javanese | Manual QC | Open | Audio |
| Regional Accents | 5 ethnic groups | Indonesian | Mobile Quality | Open | Audio |
| Google TTS | 500K utterances | Indonesian | Synthetic | Open | Audio |
| Unsupervised | 260GB | Indonesian | Variable | Contact Required | Audio |

## Applications

**Speech Recognition:**

- Indonesian ASR system development
- Multi-accent recognition models
- Regional language ASR

**Speech Synthesis:**

- Text-to-speech systems
- Voice cloning applications
- Multilingual speech generation

**Audio-Visual Applications:**

- Lip reading systems- Visual speech synthesis- Cross-modal speech processing
- Silent speech interfaces

**Language Identification:**

- Automatic language detection
- Dialect classification
- Code-switching detection


## Usage Guidelines

**For ASR Development:**

- Use TITML-IDN for high-quality baseline
- Add regional accent data for robustness
- Include local language data for multilingual models

**For Audio-Visual Applications:**

- LUMINA provides synchronized audio-visual data- Suitable for lip reading and visual speech synthesis
- Controlled recording environment ensures data quality

**For TTS Development:**

- Start with high-quality recordings (TITML-IDN, LUMINA)
- Use synthetic data for data augmentation
- Consider regional variations


## Data Quality and Licensing

- **TITML-IDN**: Highest quality, academic license required
- **LUMINA**: High-quality audio-visual, Creative Commons 4.0- **OpenSLR datasets**: Good quality, open access
- **Synthetic datasets**: Lower quality but large scale
- **Regional accents**: Mobile quality but valuable for diversity


## Citations

```

@article{SETYANINGSIH2024110279,
title = {LUMINA: Linguistic unified multimodal Indonesian natural audio-visual dataset},
journal = {Data in Brief},
volume = {54},
pages = {110279},
year = {2024},
issn = {2352-3409},
doi = {https://doi.org/10.1016/j.dib.2024.110279},
url = {https://www.sciencedirect.com/science/article/pii/S2352340924002488},
author = {Eka Rahayu Setyaningsih and Anik Nur Handayani and Wahyu Sakti Gunawan Irianto and Yosi Kristian and Christian Trisno Sen Long Chen},
keywords = {Constrained audio-visual dataset, Lips reading, Speech synthesis, Face processing, Computer vision},
abstract = {The LUMINA (Linguistic Unified Multimodal Indonesian Natural Audio-Visual) Dataset is a carefully curated constrained audio-visual dataset designed to support research in the field of speech perception. Spoken exclusively in Indonesian, LUMINA contains high-quality audio-visual recordings featuring 14 native speakers, including 9 males and 5 females. Each speaker contributes approximately 1,000 sentences, producing a rich and diverse data collection. The recorded videos focus on facial recordings, capturing essential visual cues and expressions that accompany speech. This extensive dataset provides a valuable resource for understanding how humans perceive and process spoken language, paving the way for speech recognition and synthesis technology advancements.}
}

@inproceedings{kjartansson-etal-sltu2018,
title = {{Crowd-Sourced Speech Corpora for Javanese, Sundanese, Sinhala, Nepali, and Bangladeshi Bengali}},
author = {Oddur Kjartansson and Supheakmungkol Sarin and Knot Pipatsrisawat and Martin Jansche and Linne Ha},
booktitle = {Proc. The 6th Intl. Workshop on Spoken Language Technologies for Under-Resourced Languages (SLTU)},
year = {2018},
address = {Gurugram, India},
month = aug,
pages = {52--55},
URL = {http://dx.doi.org/10.21437/SLTU.2018-11},
}

@inproceedings{valk2021slt,
title={{VoxLingua107}: a Dataset for Spoken Language Recognition},
author={Jörgen Valk and Tanel Alumäe},
booktitle={Proc. IEEE SLT Workshop},
year={2021},
}

@inproceedings{black2019cmu,
title={CMU Wilderness Multilingual Speech Dataset},
author={Black, Alan W},
booktitle={ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
pages={5971--5975},
year={2019},
organization={IEEE}
}

```