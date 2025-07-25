# IndoWiki
https://github.com/IgoRamli/IndoWiki/
IndoWiki is a knowledge-graph dataset taken from [WikiData](https://www.wikidata.org/) and aligned with [Wikipedia Bahasa Indonesia](https://id.wikipedia.org/) as it's corpus. IndoWiki is an Indonesian version of [Wikidata5M](https://deepgraphlearning.github.io/project/wikidata5m) - a knowledge graph dataset used in [KEPLER](https://arxiv.org/pdf/1911.06136.pdf).

## Data Structure

IndoWiki dataset consists of several files:
- indowiki_text.txt
- indowiki_transductive_train.txt
- indowiki_transductive_valid.txt
- indowiki_transductive_test.txt
- indowiki_inductive_train.txt
- indowiki_inductive_valid.txt
- indowiki_inductive_test.txt

The main dataset can be accesed in [this](https://drive.google.com/drive/folders/1V79VrSJ_ljz652iETARjHoB_zEfEIxV1?usp=sharing) directory. The dataset is created using WikiData JSON dump downloaded at 2021-09-30 and Wikipedia Bahasa Indonesia dump downloaded at 2021-10-01. Due to the large size of each dump (\~100GB for WikiData, \~600MB for Wikipedia Bahasa Indonesia), they will not be included in this repository.

| Setting      | Split | #Entity | #Relation | #Triplet |
|--------------|-------|---------|-----------|----------|
| Transductive | Train | 533611  | 939       | 2629235  |
|              | Valid | 18745   | 386       | 13245    |
|              | Test  | 20774   | 409       | 14818    |
| Inductive    | Train | 513695  | 931       | 1968029  |
|              | Valid | 2890    | 209       | 13245    |
|              | Test  | 4639    | 238       | 14818    |

## Pipeline

IndoWiki dataset is built from a series of data processing to transform WikiData and Wikipedia dumps into a series of entity descriptions and triplets. This process is illustrated by the pipeline diagram below:

![](indowiki-pipeline.png)

### Filtering Entities

IndoWiki requires every items in the knowledge graph to have an entry in Wikipedia. This usually means an article with paragraph format in Wikipedia Bahasa Indonesia. In a raw WikiData JSON dump, entities contained does not always meet this criteria. Some entities may not have any articles on Wikipedia Bahasa Indonesia. To create a new dump of entities that has an entry in a specific Wikipedia site, run

```bash
python filter_entities.py dump.json.gz --sitelink=idwiki
gzip dump-idwiki.json
```
Where dump.json.gz is the name of the raw dump file
The sitelink parameter can be modified depending on the Wikipedia site that is going to be used. For example, to filter entities with an article in English Wikipedia, set sitelink value to 'enwiki'.

### Extracting Wikipedia Dumps

In general, Wikisites provides a regularly updated dump of their site contents. For example, to get the latest version of all articles from Wikipedia Bahasa Indonesia, call

```
wget https://dumps.wikimedia.org/idwiki/latest/idwiki-latest-pages-articles.xml.bz2
```

The content given will be in a raw XML format. Transforming this raw format into a 'plain text' format requires the use of Wikipedia's TextExtractor. To run TextExtractor on Wikipedia dumps, we need to use [WikiExtractor](https://github.com/apertium/WikiExtractor). To download and run WikiExtractor on our Wikipedia dump,
```
wget https://raw.githubusercontent.com/apertium/WikiExtractor/master/WikiExtractor.py
WikiExtractor.py --infn idwiki-latest-pages-articles.xml.bz2
```
The result is a text file named wiki.txt. Each article starts with a blank line, followed with a line containing the article's title and a colon (:). Subsequent non-blank line contains the text content of a Wikipedia article.

### Extracting Titles

WikiData dumps only store the title of an entity. Therefore, a mapping of entity IDs and Wikipedia titles needs to be created
```
python extract_titles.py dump-idwiki.json.gz --sitelink=idwiki --out-file=titles.txt
```

### Mapping Descriptions

With a file containing contents for each title, and a mapping between entity IDs and titles, we can create the indowiki_text.txt file
```
python map_entities.py wiki.txt titles.txt --out-file=indowiki_text.txt --missing-file=missing.txt --unused-file=unused.txt
```
Sometimes, some entities with a title in WikiData dumps may not have an article in our Wikipedia dumps. There may be several reasons for this:
- The Wikipedia dump is outdated and the entity title has changed
- The title given in the WikiData dump is a redirect and the title given by WikiData does not match the title at Wikipedia
- The given WikiData title has no article. This can happen for entities whose title does not lead to it's main article, such as 'Category:History of Benin'.
In any case, the aforementioned entity will not be written and is considered invalid. If the missing-file parameter is defined (like in the example command above), a list of missing entities will be written in the missing file log.
Conversely, there may be cases where a Wikipedia article does not have any WikiData entity linked to it. In this case, if the unused-file parameter is defined, a list of unused article titles will be written in the unused file log.

### Extracting Triplets

WikiData entity with an entry in the indowiki_text.txt file means that entity ca be used in our knowledge graph. With the list of valid entities at hand, we can create a list of valid triplets.
```
python extract_triplets.py dump-idwiki.json.gz indowiki_text.txt  --out-file=indowiki_triplets.txt
```
Triplets will be shuffled in the resulting file. Shuffling list of triplets minimizes the bias created when slicing dataset.

### Creating a Split

There are two settings of Knowledge Graph splitting:
- Transductive: Entities and relations are shared across splits. Only triplets are disjointed between splits. Formally, there are no pair of triplets <img src="https://render.githubusercontent.com/render/math?math=(h_1, r_1, t_1) \in KG_{train}, (h_2, r_2, t_2) \in KG_{test}"> such that <img src="https://render.githubusercontent.com/render/math?math=(h_1, r_1, t_1) = (h_2, r_2, t_2)">
- Inductive: Entities, relations, and triplets are disjointed between splits. Formally, for every pair of triplets <img src="https://render.githubusercontent.com/render/math?math=(h_1, r_1, t_1) \in KG_{train}, (h_2, r_2, t_2) \in KG_{test}">, <img src="https://render.githubusercontent.com/render/math?math=h_1 \neq h_2"> and <img src="https://render.githubusercontent.com/render/math?math=t_1 \neq t_2">


Perform a transductive split with this command
```
python split_triplets.py indowiki_triplets.txt --setting transductive --valid 13245 --test 14818 --out-file-prefix indowiki_transductive
```
and an inductive split with this command
```
python split_triplets.py indowiki_triplets.txt --setting inductive --out-file-prefix indowiki_inductive
```
Note that results from the two commands above may not be reproducible

### (Optional) Test Inductivity

General informations of a data split (train/valid/test) can be obtained using the `analyze_dataset.py` program. Below is an example of obtaining the number of entities, relations, and triplets present in the indowiki_inductive_test text file:
```
python analyze_dataset.py indowiki_inductive_test.txt
```

If you are not sure whether two KE splits are really inductive (have disjoint sets of entities), you can perform an inductivity check:
```
python test_inductivity.py indowiki_inductive_train.txt indowiki_inductive_test.txt indowiki_inductive_test.txt
```