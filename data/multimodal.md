# Multimodal Dataset
Source of this information (part of it): https://github.com/Wikidepia/indonesian_datasets

## Conceptual Captions 3M

[[Original Dataset](https://github.com/google-research-datasets/conceptual-captions)] 

[[Update Link for Download](https://stor.akmal.dev/cc3m-train.jsonl.zst)]

Conceptual Captions is a dataset containing (image-URL, caption) pairs designed for the training and evaluation of machine learned image captioning systems.

### Citations

```bibtex
@inproceedings{sharma2018conceptual,
  title = {Conceptual Captions: A Cleaned, Hypernymed, Image Alt-text Dataset For Automatic Image Captioning},
  author = {Sharma, Piyush and Ding, Nan and Goodman, Sebastian and Soricut, Radu},
  booktitle = {Proceedings of ACL},
  year = {2018},
}
```

## Conceptual Captions 12M

[[Original Dataset](https://github.com/google-research-datasets/conceptual-12m)]

[[Update Link for Download](https://stor.akmal.dev/cc12m.jsonl.zst)]

Conceptual 12M (CC12M), a dataset with ~12 million image-text pairs meant to be used for vision-and-language pre-training. It is larger and covers a much more diverse set of visual concepts than the Conceptual Captions (CC3M), a dataset that is widely used for pre-training and end-to-end training of image captioning models. Check our paper for further details.

#$# Citations

```bibtex
@inproceedings{changpinyo2021cc12m,
  title = {{Conceptual 12M}: Pushing Web-Scale Image-Text Pre-Training To Recognize Long-Tail Visual Concepts},
  author = {Changpinyo, Soravit and Sharma, Piyush and Ding, Nan and Soricut, Radu},
  booktitle = {CVPR},
  year = {2021},
}
```


# YFCC100M OpenAI Subset

[[Original Dataset](https://github.com/openai/CLIP/blob/main/data/yfcc100m.md)]

[[Update Link for Download](https://stor.akmal.dev/yfcc100.jsonl.zst)]

The subset contains 14,829,396 images, about 15% of the full dataset, which have been filtered to only keep those with natural language titles and/or descriptions in English.

# Citations

```bibtex
@article{Thomee_2016,
   title={YFCC100M},
   volume={59},
   ISSN={1557-7317},
   url={http://dx.doi.org/10.1145/2812802},
   DOI={10.1145/2812802},
   number={2},
   journal={Communications of the ACM},
   publisher={Association for Computing Machinery (ACM)},
   author={Thomee, Bart and Shamma, David A. and Friedland, Gerald and Elizalde, Benjamin and Ni, Karl and Poland, Douglas and Borth, Damian and Li, Li-Jia},
   year={2016},
   month={Jan},
   pages={64–73}
}
@misc{radford2021learning,
      title={Learning Transferable Visual Models From Natural Language Supervision}, 
      author={Alec Radford and Jong Wook Kim and Chris Hallacy and Aditya Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
      year={2021},
      eprint={2103.00020},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


## MSVD-Indonesian

MSVD-Indonesian (Paper: [link](https://arxiv.org/abs/2306.11341)) is derived from the MSVD dataset, which is obtained with the help of a machine translation service. This dataset can be used for multimodal video-text tasks, including text-to-video retrieval, video-to-text retrieval, and video captioning. Same as the original English dataset, the MSVD-Indonesian dataset contains about 80k video-text pairs.

### Data

Indonesian (Bahasa Indonesia) sentences: [link](https://github.com/willyfh/msvd-indonesian/blob/main/data/MSVD-indonesian.txt)

Raw videos: [link](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/YouTubeClips.tar)

### Qualitative Results

#### Text-to-Video  Retrieval
![Text-to-Video Retrieval](https://raw.githubusercontent.com/willyfh/msvd-indonesian/main/figures/qualitative-results-t2v-ret.png)

#### Video-to-Text Retrieval
![Video-to-Text Retrieval](https://raw.githubusercontent.com/willyfh/msvd-indonesian/main/figures/qualitative-results-v2t-ret.png)

#### Video Captioning
![Video Captioning](https://raw.githubusercontent.com/willyfh/msvd-indonesian/main/figures/qualitative-results-v2t-cap.png)

### Citation
If you find our work useful in your research, please cite:

```bibtex
@article{Hendria2023MSVDID,
  title={{MSVD}-{I}ndonesian: A Benchmark for Multimodal Video-Text Tasks in Indonesian},
  author={Willy Fitra Hendria},
  journal={arXiv preprint arXiv:2306.11341},
  year={2023}
}
```

### Acknowledgments
Our experimental results are obtained utilizing the resources from [X-CLIP](https://github.com/xuguohai/X-CLIP) and [VNS-GRU](https://github.com/WingsBrokenAngel/delving-deeper-into-the-decoder-for-video-captioning). We thank the original authors for their open-sourcing.


## KTP VLM Instruct Dataset
https://huggingface.co/datasets/danielsyahputra/ktp-vlm-instruct-dataset

## IndoCareer

### Introduction

IndoCareer is a dataset comprising 8,834 multiple-choice questions designed to evaluate performance in vocational and professional certification exams across various fields. With a focus on Indonesia, IndoCareer provides rich local contexts, spanning six key sectors: (1) healthcare, (2) insurance and finance, (3) creative and design, (4) tourism and hospitality, (5) education and training, and (6) law.
<p align="left"> <img src="https://raw.githubusercontent.com/fajri91/eval_picts/refs/heads/master/indocareer_pie.png" style="width: 40%;" id="title-icon">       </p>

## Data
Each question in the dataset is a multiple-choice question with up to 5 choices and only one choice as the correct answer. 

```
import datasets
data = datasets.load_dataset('indolem/IndoCareer', 'all')
```

## Examples

These questions are written in Indonesian.

<p align="left"> 
    <img src="https://raw.githubusercontent.com/fajri91/eval_picts/refs/heads/master/indocareer_example.png" style="width: 40%;" id="title-icon"> 
  
</p>
## Evaluation

We evaluated one closed-source model (GPT-4o) and 26 open-weight LLMs:

<p align="left"> <img src="https://raw.githubusercontent.com/fajri91/eval_picts/refs/heads/master/indocareer_result.png" style="width: 70%;" id="title-icon">       </p>


## Citation
Please find out paper 📄<a href="https://arxiv.org/pdf/2409.08564" target="_blank" style="margin-right: 15px; margin-left: 10px">here.</a>
```
@inproceedings{koto2025cracking,
  title={Cracking the Code: Multi-domain LLM Evaluation on Real-World Professional Exams in Indonesia},
  author={"Fajri Koto"},
  booktitle={Proceedings of the 2025 Conference of the North American Chapter of the Association for Computational Linguistics – Human Language Technologies (NAACL HLT 2025), Industry Track},
  year={2025}
}
```

## IndoMMLU

### Introduction

We introduce IndoMMLU, the first multi-task language understanding benchmark for Indonesian culture and languages, 
which consists of questions from primary school to university entrance exams in Indonesia. By employing professional teachers, 
we obtain 14,906 questions across 63 tasks and education levels, with 46\% of the questions focusing on assessing proficiency 
in the Indonesian language and knowledge of nine local languages and cultures in Indonesia.
<p align="left"> <img src="https://github.com/fajri91/eval_picts/blob/master/IndoMMLU-dist.png?raw=true" style="width: 500px;" id="title-icon">       </p>

### Subjects

 | Level     | Subjects                           | 
 |-----------|------------------------------------|
 | SD (Primary School)  | Science, Social science, Civics, Indonesian Language, Balinese, Makassarese, Banjarese, Lampungic, Madurese, Sundanese, Javanese, Dayak Ngaju, Minangkabau culture, Art, Sports, Islam religion, Christian religion, Hindu religion |
 | SMP (Junior High School) | Science, Social science, Civics, Indonesian Language, Balinese, Makassarese, Banjarese, Lampungic, Madurese, Sundanese, Javanese, Minangkabau culture, Art, Sports, Islam religion, Christian religion, Hindu religion | 
 | SMA (Senior High School) | Physics, Chemistry, Biology, Geography, Sociology, Economics, History, Civics, Indonesian Language, Balinese, Makassarese, Banjarese, Lampungic, Madurese, Sundanese, Javanese, Art, Sports, Islam religion, Christian religion, Hindu religion | 
 University Entrance Test | Chemistry, Biology, Geography, Sociology, Economics, History, Indonesian Language |

We categorize the collected questions into different subject areas, including: (1) STEM (Science, Technology, Engineering, and Mathematics); (2) Social Science; (3) Humanities; (4) Indonesian Language; and (5) Local Languages and Cultures. 

### Examples

These questions are written in Indonesian. For local language subjects, some are written in the local languages. The English version is for illustrative purposes only.

<p align="left"> 
    <img src="https://github.com/fajri91/eval_picts/blob/master/min_example.png?raw=true" style="width: 400px;" id="title-icon"> 
</p>

### Evaluation

We evaluate 24 multilingual LLMs of different sizes in zero-shot and few-shot settings. This includes [GPT-3.5 (ChatGPT)](https://chat.openai.com/), [XGLM](https://arxiv.org/abs/2112.10668), [Falcon](https://falconllm.tii.ae/), [BLOOMZ](https://huggingface.co/bigscience/bloomz), [mT0](https://huggingface.co/bigscience/bloomz), [LLaMA](https://arxiv.org/abs/2302.13971), and [Bactrian-X](https://github.com/mbzuai-nlp/bactrian-x). Prior to the question and multiple-choice options, we add a simple prompt in the Indonesian language:

```
 Ini adalah soal [subject] untuk [level]. Pilihlah salah satu jawaban yang dianggap benar!
 English Translation: This is a [subject] question for [level]. Please choose the correct answer!
```

### Citation
```
@inproceedings{koto-etal-2023-indommlu,
    title = "Large Language Models Only Pass Primary School Exams in {I}ndonesia: A Comprehensive Test on {I}ndo{MMLU}",
    author = "Fajri Koto and Nurul Aisyah and Haonan Li and Timothy Baldwin",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = December,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
}
```

### License

The IndoMMLU dataset is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).


## IndoCulture

There are 2430 dataset that contain question and multiple choise answer about the culture in Indonesia.

https://huggingface.co/datasets/indolem/IndoCulture


## IndoCloze

### About
Indolem hired seven Indonesian university students to each write 500 short stories over a period of one month. This paper wins **Best Paper Award at CSRR (ACL 2022)**.

### Paper
Fajri Koto, Timothy Baldwin, and Jey Han Lau. [_Cloze Evaluation for Deeper Understanding of Commonsense Stories in Indonesian_](https://aclanthology.org/2022.csrr-1.2.pdf). 
In In Proceedings of Commonsense Representation and Reasoning Workshop 2022 (**CSRR at ACL 2022**), Dublin, Ireland. 

### Dataset

A story in our dataset consists of four-sentence premise, one-sentence correct ending, and one-sentence incorrect ending. In total, we have created 2,325 Indonesian stories with the train/dev/test split 1,000/200/1,135. Please see some examples of our data below, and note that the English translation is only for the illustratrive purposes.

<h3 align="center">
<img src="https://raw.githubusercontent.com/fajri91/eval_picts/master/indocloze.png" width="850">
</h3>