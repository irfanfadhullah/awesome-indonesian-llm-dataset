# 📚 Knowledge Graphs

This section covers knowledge graph datasets and resources for Indonesian, including entity-relation triplets and structured knowledge representations.

## IndoWiki: Indonesian Knowledge Graph

- **Description**: Knowledge graph from WikiData aligned with Indonesian Wikipedia
- **Size**: 533K+ entities, 939 relations, 2.6M+ triplets
- **Format**: Both transductive and inductive splits available
- **Links**: [GitHub Repository](https://github.com/IgoRamli/IndoWiki/) | [Google Drive](https://drive.google.com/drive/folders/1V79VrSJ_ljz652iETARjHoB_zEfEIxV1?usp=sharing)

**Data Structure:**
- Transductive setting: Shared entities/relations across splits
- Inductive setting: Disjoint entities between train/test
- Complete pipeline for data processing included

### Dataset Statistics

| Setting      | Split | #Entity | #Relation | #Triplet |
|--------------|-------|---------|-----------|----------|
| Transductive | Train | 533611  | 939       | 2629235  |
|              | Valid | 18745   | 386       | 13245    |
|              | Test  | 20774   | 409       | 14818    |
| Inductive    | Train | 513695  | 931       | 1968029  |
|              | Valid | 2890    | 209       | 13245    |
|              | Test  | 4639    | 238       | 14818    |

### Pipeline Overview

IndoWiki dataset is built from a series of data processing to transform WikiData and Wikipedia dumps into entity descriptions and triplets:

1. **Filtering Entities**: Filter WikiData entities that have entries in Indonesian Wikipedia
2. **Extracting Wikipedia Dumps**: Process raw Wikipedia XML format to plain text
3. **Extracting Titles**: Create mapping between entity IDs and Wikipedia titles
4. **Mapping Descriptions**: Create entity-description mappings
5. **Extracting Triplets**: Generate valid triplets from filtered entities
6. **Creating Splits**: Generate transductive and inductive data splits

### Usage Examples

```


# Filter entities with Indonesian Wikipedia entries

python filter_entities.py dump.json.gz --sitelink=idwiki

# Extract Wikipedia content

WikiExtractor.py --infn idwiki-latest-pages-articles.xml.bz2

# Create entity mappings

python map_entities.py wiki.txt titles.txt --out-file=indowiki_text.txt

# Extract triplets

python extract_triplets.py dump-idwiki.json.gz indowiki_text.txt --out-file=indowiki_triplets.txt

# Create splits

python split_triplets.py indowiki_triplets.txt --setting transductive --valid 13245 --test 14818

```

## Citations

```

@misc{indowiki2021,
title={IndoWiki: Indonesian Knowledge Graph Dataset},
author={Ramli, Igo and others},
year={2021},
url={https://github.com/IgoRamli/IndoWiki/}
}

```