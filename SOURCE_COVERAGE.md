# Repository source coverage

Generated from `catalog.json` and the fingerprint-matched `audit.json` snapshot.
These are **source records, not a count of unique datasets**: mirrors, configurations,
papers and tools are distinguished. A reachable index is not a successful sample test.

Total source records: **284**.

| Resource kind | Records |
| --- | ---: |
| dataset_source | 224 |
| model | 10 |
| paper | 13 |
| reference | 18 |
| tool | 19 |

## Data-source records

`registered` means a reviewed canonical loader. `file`/`hf_parquet` preserve raw
task fields; they need an explicit preparation recipe before training. `artifact`
downloads an opaque archive without extraction. Pending/none requires audit,
permission, credentials, provider action or a custom adapter. GitHub-discovered
file candidates are not asserted to be task-ready datasets. Unknown splits are
excluded from default training plans. An HTTP/DNS failure describes this audit,
not permanent unavailability. See each source's `instructions` and license evidence.

| Source ID | Adapter | Availability observed | Access | Declared license |
| --- | --- | --- | --- | --- |
| `clean-mc4-indonesian` | none | no_download_url | manual | unknown |
| [`corpora-uni-leipzig-de-en-68df48f2`](https://corpora.uni-leipzig.de/en?corpusId=ind_mixed_2013) | none | landing_page_reachable | unverified | unknown |
| [`corpora-wortschatz-leipzig-de-en-a2824add`](https://corpora.wortschatz-leipzig.de/en?corpusId=ind_news_2022) | none | landing_page_reachable | unverified | unknown |
| [`cs-taltech-ee-staff-tanel-alumae-data-voxlingua107-96fce01b`](https://cs.taltech.ee/staff/tanel.alumae/data/voxlingua107) | none | landing_page_reachable | unverified | unknown |
| [`cs-taltech-ee-staff-tanel-alumae-data-voxlingua107-id-zip-7aaddc3b`](https://cs.taltech.ee/staff/tanel.alumae/data/voxlingua107/id.zip) | artifact | reachable | public_unverified | unknown |
| [`cs-taltech-ee-staff-tanel-alumae-data-voxlingua107-jw-zip-a5353a60`](https://cs.taltech.ee/staff/tanel.alumae/data/voxlingua107/jw.zip) | artifact | reachable | public_unverified | unknown |
| [`cs-taltech-ee-staff-tanel-alumae-data-voxlingua107-su-zip-a35a97c0`](https://cs.taltech.ee/staff/tanel.alumae/data/voxlingua107/su.zip) | artifact | reachable | public_unverified | unknown |
| [`data-mendeley-com-datasets-2zpbjs22k3-1-36e2f205`](https://data.mendeley.com/datasets/2zpbjs22k3/1) | none | landing_page_reachable | manual | unknown |
| [`data-mendeley-com-datasets-8fw93k4rny-4-e3747d6b`](https://data.mendeley.com/datasets/8fw93k4rny/4) | none | landing_page_reachable | manual | unknown |
| [`data-mendeley-com-datasets-nvc3rsf53b-5-9dc16085`](https://data.mendeley.com/datasets/nvc3rsf53b/5) | none | landing_page_reachable | manual | unknown |
| [`data-statmt-org-cc-100-b2098ed4`](http://data.statmt.org/cc-100) | none | landing_page_reachable | unverified | unknown |
| [`docs-google-com-forms-d-1bfkimfszoswkcbua76yhqi9hizlrjyne-1g-r5unfww-edi-fed3a0fa`](https://docs.google.com/forms/d/1bFkimFsZoswKCbUa76yHqi9hizLrJYne-1G_r5unfww/edit) | none | landing_page_reachable | manual | unknown |
| [`drive-google-com-drive-folders-1v79vrsj-ljz652ietarjhob-zefeixv1-ea03c745`](https://drive.google.com/drive/folders/1V79VrSJ_ljz652iETARjHoB_zEfEIxV1?usp=sharing) | none | landing_page_reachable | manual | unknown |
| [`drive-google-com-file-d-1-stkadftgjltyshwqxhgo3ntjkvfxm-q-view-b4ebfa03`](https://drive.google.com/file/d/1-stKadfTgJLtYsHWqXhGO3nTjKVFxm_Q/view) | none | landing_page_reachable | manual | unknown |
| [`drive-google-com-file-d-1ogybpfxfav3tbwp1qcwt-cc9cvwsjaco-view-bbc8681e`](https://drive.google.com/file/d/1OgYbPfXFAv3TbwP1Qcwt_CC9cVWSJaco/view) | none | landing_page_reachable | manual | unknown |
| [`drive-google-com-file-d-1stcb5ndpq6vuolxr29dbvssservxld1d-view-afcde990`](https://drive.google.com/file/d/1sTCB5NDPq6vUOlxR29DbvSssErvXLD1d/view?usp=sharing) | none | not_found | manual | unknown |
| [`fineweb-edu-id`](https://huggingface.co/datasets/irfanfadhullah/FineWeb-Edu-25K) | registered | pinned_metadata_verified | public | unknown |
| [`github-com-abhimantramb-elang-blob-master-word2vec-utils-indonesian-regi-387d603a`](https://github.com/abhimantramb/elang/blob/master/word2vec/utils/indonesian-region.txt) | file_pending | not_found | unverified | unknown |
| [`github-com-abhimantramb-elang-tree-master-word2vec-utils-stopwords-list-72d2c8fc`](https://github.com/abhimantramb/elang/tree/master/word2vec/utils/stopwords-list) | github_pending | check_failed | unverified | unknown |
| [`github-com-agusmakmun-sentistrengthid-blob-master-id-dict-emoticon-txt-b6502aaa`](https://github.com/agusmakmun/SentiStrengthID/blob/master/id_dict/emoticon.txt) | file | reachable | public | unknown |
| [`github-com-agusmakmun-sentistrengthid-blob-master-id-dict-rootword-txt-a37c068b`](https://github.com/agusmakmun/SentiStrengthID/blob/master/id_dict/rootword.txt) | file | reachable | public | unknown |
| [`github-com-agusmakmun-sentistrengthid-blob-master-id-dict-sentimentword--dcb306fb`](https://github.com/agusmakmun/SentiStrengthID/blob/master/id_dict/sentimentword.txt) | file | reachable | public | unknown |
| [`github-com-agusmakmun-sentistrengthid-blob-master-id-dict-slangword-txt-a78e3e1b`](https://github.com/agusmakmun/SentiStrengthID/blob/master/id_dict/slangword.txt) | file | reachable | public | unknown |
| [`github-com-annisanurulazhar-absa-playground-6a14bac0`](https://github.com/annisanurulazhar/absa-playground) | file | native_file_index_verified | public | unknown |
| [`github-com-dziem-proner-labeled-text-e523a088`](https://github.com/dziem/proner-labeled-text) | file | native_file_index_verified | public | unknown |
| [`github-com-edwardsamuel-wilayah-administratif-indonesia-tree-master-csv-fd747a83`](https://github.com/edwardsamuel/Wilayah-Administratif-Indonesia/tree/master/csv) | file | native_file_index_verified | public | unknown |
| [`github-com-esdurmus-wikilingua-7fdcf5c6`](https://github.com/esdurmus/Wikilingua) | none | no_indexed_data_files | manual | unknown |
| [`github-com-fajri91-inset-1a716a1e`](https://github.com/fajri91/InSet) | file | native_file_index_verified | public | unknown |
| [`github-com-famrashel-idn-tagged-corpus-4d4b91b4`](https://github.com/famrashel/idn-tagged-corpus) | file | native_file_index_verified | public | unknown |
| [`github-com-famrashel-idn-treebank-35f05da5`](https://github.com/famrashel/idn-treebank) | none | no_indexed_data_files | manual | unknown |
| [`github-com-festvox-datasets-cmu-wilderness-26fa465d`](https://github.com/festvox/datasets-CMU_Wilderness) | file | native_file_index_verified | public | unknown |
| [`github-com-frankydotid-indonesian-speech-recognition-3a404a3d`](https://github.com/frankydotid/Indonesian-Speech-Recognition) | file | native_file_index_verified | public | unknown |
| [`github-com-google-research-datasets-tydiqa-bbaf7643`](https://github.com/google-research-datasets/tydiqa) | file | native_file_index_verified | public | unknown |
| [`github-com-grit-id-nergrit-corpus-03040546`](https://github.com/grit-id/nergrit-corpus) | github_pending | check_failed | unverified | unknown |
| [`github-com-haryoa-indo-collex-15f62dc3`](https://github.com/haryoa/indo-collex) | file | native_file_index_verified | public | unknown |
| [`github-com-haryoa-stif-indonesia-67f74ee6`](https://github.com/haryoa/stif-indonesia) | file | native_file_index_verified | public | unknown |
| [`github-com-ialfina-id-hatespeech-detection-1274b659`](https://github.com/ialfina/id-hatespeech-detection) | file | native_file_index_verified | public | unknown |
| [`github-com-ialfina-kethu-c94981ee`](https://github.com/ialfina/kethu) | file | native_file_index_verified | public | unknown |
| [`github-com-ialfina-ner-dataset-modified-dee-300ec7ad`](https://github.com/ialfina/ner-dataset-modified-dee) | file | native_file_index_verified | public | unknown |
| [`github-com-igoramli-indowiki-2e851fc3`](https://github.com/IgoRamli/IndoWiki) | none | no_indexed_data_files | manual | unknown |
| [`github-com-indobenchmark-indonlg-7e049059`](https://github.com/indobenchmark/indonlg) | file | native_file_index_verified | public | unknown |
| [`github-com-indobenchmark-indonlu-4e203553`](https://github.com/indobenchmark/indonlu) | file | native_file_index_verified | public | unknown |
| [`github-com-indolem-indobertweet-e6127a30`](https://github.com/indolem/IndoBERTweet) | file | native_file_index_verified | public | unknown |
| [`github-com-indolem-sum-liputan6-1eea3c27`](https://github.com/indolem/sum_liputan6) | none | no_indexed_data_files | permission_required | research-only |
| [`github-com-indonlp-nusa-writes-9afc4a23`](https://github.com/IndoNLP/nusa-writes) | file | native_file_index_verified | public | unknown |
| [`github-com-ir-nlp-csui-indonli-07e8f40a`](https://github.com/ir-nlp-csui/indonli) | file | native_file_index_verified | public | unknown |
| [`github-com-jolicode-emoji-search-blob-master-synonyms-cldr-emoji-annotat-3fbb8a95`](https://github.com/jolicode/emoji-search/blob/master/synonyms/cldr-emoji-annotation-synonyms-id.txt) | file_pending | http_error | unverified | unknown |
| [`github-com-jordhy97-final-project-9debcde3`](https://github.com/jordhy97/final_project) | file | native_file_index_verified | public | unknown |
| [`github-com-kata-ai-indosum-075b525e`](https://github.com/kata-ai/indosum) | none | no_indexed_data_files | manual | unknown |
| [`github-com-kata-ai-kawat-3b281d55`](https://github.com/kata-ai/kawat) | file | native_file_index_verified | public | unknown |
| [`github-com-kmkurn-id-nlp-resource-2af1032b`](https://github.com/kmkurn/id-nlp-resource) | none | no_indexed_data_files | manual | unknown |
| [`github-com-louisowen6-nlp-bahasa-resources-blob-master-combined-root-wor-ce107578`](https://github.com/louisowen6/NLP_bahasa_resources/blob/master/combined_root_words.txt) | file | reachable | public | unknown |
| [`github-com-louisowen6-nlp-bahasa-resources-blob-master-combined-slang-wo-f2245e1b`](https://github.com/louisowen6/NLP_bahasa_resources/blob/master/combined_slang_words.txt) | file | reachable | public | unknown |
| [`github-com-louisowen6-nlp-bahasa-resources-blob-master-combined-stop-wor-d2a14789`](https://github.com/louisowen6/NLP_bahasa_resources/blob/master/combined_stop_words.txt) | file | reachable | public | unknown |
| [`github-com-louisowen6-quora-paraphrasing-id-44094cf5`](https://github.com/louisowen6/quora_paraphrasing_id) | file | native_file_index_verified | public | unknown |
| [`github-com-masdevid-id-opinionwords-blob-master-negative-txt-fb2074a2`](https://github.com/masdevid/ID-OpinionWords/blob/master/negative.txt) | file | reachable | public | unknown |
| [`github-com-masdevid-id-opinionwords-blob-master-positive-txt-da1eb13c`](https://github.com/masdevid/ID-OpinionWords/blob/master/positive.txt) | file | reachable | public | unknown |
| [`github-com-nasalsabila-kamus-alay-40fc0e01`](https://github.com/nasalsabila/kamus-alay) | file | native_file_index_verified | public | unknown |
| [`github-com-okkyibrohim-id-abusive-language-detection-97221951`](https://github.com/okkyibrohim/id-abusive-language-detection) | file | native_file_index_verified | public | unknown |
| [`github-com-okkyibrohim-id-multi-label-hate-speech-and-abusive-language-d-1641a63c`](https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection/blob/master/new_kamusalay.csv) | file | reachable | public | unknown |
| [`github-com-okkyibrohim-id-multi-label-hate-speech-and-abusive-language-d-79a215f3`](https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection) | file | native_file_index_verified | public | unknown |
| [`github-com-panggi-pujangga-blob-master-resource-formalization-formalizat-1eff6800`](https://github.com/panggi/pujangga/blob/master/resource/formalization/formalizationDict.txt) | file | reachable | public | unknown |
| [`github-com-panggi-pujangga-blob-master-resource-netagger-contextualfeatu-57ea8b31`](https://github.com/panggi/pujangga/blob/master/resource/netagger/contextualfeature/ppre.txt) | file | reachable | public | unknown |
| [`github-com-panggi-pujangga-blob-master-resource-netagger-contextualfeatu-8a3ca3b0`](https://github.com/panggi/pujangga/blob/master/resource/netagger/contextualfeature/country.txt) | file | reachable | public | unknown |
| [`github-com-panggi-pujangga-blob-master-resource-netagger-contextualfeatu-b3fd7b11`](https://github.com/panggi/pujangga/blob/master/resource/netagger/contextualfeature/lpre.txt) | file | reachable | public | unknown |
| [`github-com-panggi-pujangga-blob-master-resource-reference-opre-txt-e9b0ef23`](https://github.com/panggi/pujangga/blob/master/resource/reference/opre.txt) | file | reachable | public | unknown |
| [`github-com-panggi-pujangga-blob-master-resource-sentencedetector-acronym-54c712f7`](https://github.com/panggi/pujangga/blob/master/resource/sentencedetector/acronym.txt) | file | reachable | public | unknown |
| [`github-com-pentagonal-indonesia-postal-code-tree-master-csv-e3b5366f`](https://github.com/pentagonal/Indonesia-Postal-Code/tree/master/Csv) | file | native_file_index_verified | public | unknown |
| [`github-com-prasastoadi-serangkai-blob-master-serangkai-kamus-data-kamus--8f01603a`](https://github.com/prasastoadi/serangkai/blob/master/serangkai/kamus/data/kamus-kata-dasar.csv) | file | reachable | public | unknown |
| [`github-com-ramaprakoso-analisis-sentimen-blob-master-kamus-acronym-txt-35d5bee9`](https://github.com/ramaprakoso/analisis-sentimen/blob/master/kamus/acronym.txt) | file | reachable | public | unknown |
| [`github-com-ramaprakoso-analisis-sentimen-blob-master-kamus-emoticon-txt-97238f27`](https://github.com/ramaprakoso/analisis-sentimen/blob/master/kamus/emoticon.txt) | file | reachable | public | unknown |
| [`github-com-ramaprakoso-analisis-sentimen-blob-master-kamus-kbba-txt-02a3ba48`](https://github.com/ramaprakoso/analisis-sentimen/blob/master/kamus/kbba.txt) | file | reachable | public | unknown |
| [`github-com-ramaprakoso-analisis-sentimen-blob-master-kamus-negatif-ta2-t-cb0edd90`](https://github.com/ramaprakoso/analisis-sentimen/blob/master/kamus/negatif_ta2.txt) | file | reachable | public | unknown |
| [`github-com-ramaprakoso-analisis-sentimen-blob-master-kamus-negative-add--fb596325`](https://github.com/ramaprakoso/analisis-sentimen/blob/master/kamus/negative_add.txt) | file | reachable | public | unknown |
| [`github-com-ramaprakoso-analisis-sentimen-blob-master-kamus-negative-keyw-073b8f11`](https://github.com/ramaprakoso/analisis-sentimen/blob/master/kamus/negative_keyword.txt) | file | reachable | public | unknown |
| [`github-com-ramaprakoso-analisis-sentimen-blob-master-kamus-positif-ta2-t-c19e0fcd`](https://github.com/ramaprakoso/analisis-sentimen/blob/master/kamus/positif_ta2.txt) | file | reachable | public | unknown |
| [`github-com-ramaprakoso-analisis-sentimen-blob-master-kamus-positive-add--e8342de9`](https://github.com/ramaprakoso/analisis-sentimen/blob/master/kamus/positive_add.txt) | file | reachable | public | unknown |
| [`github-com-ramaprakoso-analisis-sentimen-blob-master-kamus-positive-keyw-6eae157c`](https://github.com/ramaprakoso/analisis-sentimen/blob/master/kamus/positive_keyword.txt) | file | reachable | public | unknown |
| [`github-com-ramaprakoso-analisis-sentimen-blob-master-kamus-stopword-txt-41f8ebdf`](https://github.com/ramaprakoso/analisis-sentimen/blob/master/kamus/stopword.txt) | file | reachable | public | unknown |
| [`github-com-sastrawi-sastrawi-blob-master-data-kata-dasar-original-txt-c605ac65`](https://github.com/sastrawi/sastrawi/blob/master/data/kata-dasar.original.txt) | file | reachable | public | unknown |
| [`github-com-sastrawi-sastrawi-blob-master-data-kata-dasar-txt-aecd8386`](https://github.com/sastrawi/sastrawi/blob/master/data/kata-dasar.txt) | file | reachable | public | unknown |
| [`github-com-seuriously-genderprediction-blob-master-namatraining-txt-8f0e1554`](https://github.com/seuriously/genderprediction/blob/master/namatraining.txt) | file | reachable | public | unknown |
| [`github-com-universaldependencies-ud-indonesian-gsd-2b137fdc`](https://github.com/UniversalDependencies/UD_Indonesian-GSD) | file | native_file_index_verified | public | unknown |
| [`github-com-victoriasovereigne-tesaurus-2e9e9bd8`](https://github.com/victoriasovereigne/tesaurus) | file | native_file_index_verified | public | unknown |
| [`github-com-wikidepia-indonesia-dataset-tree-master-paraphrase-paws-9d129f3f`](https://github.com/Wikidepia/indonesia_dataset/tree/master/paraphrase/PAWS) | none | no_indexed_data_files | manual | unknown |
| [`github-com-wikidepia-indonesia-dataset-tree-master-question-answering-ma-9921aa96`](https://github.com/Wikidepia/indonesia_dataset/tree/master/question-answering/mathematics_dataset) | file | native_file_index_verified | public | unknown |
| [`github-com-wikidepia-indonesian-datasets-tree-master-dictionary-wordlist-a18d5adc`](https://github.com/Wikidepia/indonesian_datasets/tree/master/dictionary/wordlist/data/wordlist.txt) | file | reachable | public | unknown |
| [`github-com-wikidepia-indonesian-datasets-tree-master-paraphrase-paws-7e49f54d`](https://github.com/Wikidepia/indonesian_datasets/tree/master/paraphrase/paws) | file | native_file_index_verified | public | unknown |
| [`github-com-wikidepia-mathematics-dataset-id-2cc65389`](https://github.com/Wikidepia/mathematics_dataset_id) | none | no_indexed_data_files | manual | unknown |
| [`github-com-willyfh-msvd-indonesian-4d701381`](https://github.com/willyfh/msvd-indonesian) | file | native_file_index_verified | public | unknown |
| [`github-com-willyfh-msvd-indonesian-blob-main-data-msvd-indonesian-txt-d791fa44`](https://github.com/willyfh/msvd-indonesian/blob/main/data/MSVD-indonesian.txt) | file | reachable | public | unknown |
| [`github-com-yasirutomo-python-sentianalysis-id-blob-master-data-feature-l-b6d189f5`](https://github.com/yasirutomo/python-sentianalysis-id/blob/master/data/feature_list/stopwordsID.txt) | file | reachable | public | unknown |
| [`github-com-yohanesgultom-nlp-experiments-tree-master-data-ner-be05d23a`](https://github.com/yohanesgultom/nlp-experiments/tree/master/data/ner) | file | native_file_index_verified | public | unknown |
| [`hf-cis-lmu-m-lama`](https://huggingface.co/datasets/cis-lmu/m_lama) | none | metadata_available | manual | cc-by-nc-sa-4.0 |
| [`hf-common-voice`](https://huggingface.co/datasets/common_voice) | none | metadata_available | manual | cc0-1.0 |
| [`hf-covost2`](https://huggingface.co/datasets/covost2) | none | metadata_available | manual | cc-by-nc-4.0 |
| [`hf-csebuetnlp-xlsum`](https://huggingface.co/datasets/csebuetnlp/xlsum) | none | metadata_available | manual | cc-by-nc-sa-4.0 |
| [`hf-danielsyahputra-ktp-vlm-instruct-dataset`](https://huggingface.co/datasets/danielsyahputra/ktp-vlm-instruct-dataset) | hf_parquet | parquet_index_verified | public | unknown |
| [`hf-datasaur-mongabay-collection`](https://huggingface.co/datasets/Datasaur/Mongabay-collection) | hf_parquet | parquet_index_verified | public | unknown |
| [`hf-helsinki-nlp-tatoeba`](https://huggingface.co/datasets/Helsinki-NLP/tatoeba) | none | metadata_available | manual | cc-by-2.0 |
| [`hf-id-clickbait`](https://huggingface.co/datasets/id_clickbait) | none | metadata_available | manual | cc-by-4.0 |
| [`hf-id-newspapers-2018`](https://huggingface.co/datasets/id_newspapers_2018) | hf_parquet | native_file_index_verified | public | cc-by-4.0 |
| [`hf-id-panl-bppt`](https://huggingface.co/datasets/id_panl_bppt) | none | metadata_available | manual | unknown |
| [`hf-indolem-indocareer`](https://huggingface.co/datasets/indolem/IndoCareer) | hf_parquet | parquet_index_verified | public | cc-by-nc-4.0 |
| [`hf-indolem-indoculture`](https://huggingface.co/datasets/indolem/IndoCulture) | hf_parquet | parquet_index_verified | public | cc-by-nc-sa-4.0 |
| [`hf-indolem-indommlu`](https://huggingface.co/datasets/indolem/IndoMMLU) | file | native_file_index_verified | public | mit |
| [`hf-indonesian-nlp-wikipedia-id`](https://huggingface.co/datasets/indonesian-nlp/wikipedia-id) | hf_parquet | parquet_index_verified | public | unknown |
| [`hf-indonli`](https://huggingface.co/datasets/indonli) | none | metadata_available | manual | cc-by-sa-4.0 |
| [`hf-indonlp-cendol-collection-v1`](https://huggingface.co/datasets/indonlp/cendol_collection_v1) | hf_parquet | parquet_index_verified | public | apache-2.0 |
| [`hf-indonlp-cendol-collection-v2`](https://huggingface.co/datasets/indonlp/cendol_collection_v2) | hf_parquet | parquet_index_verified | public | apache-2.0 |
| [`hf-indonlp-nusaparagraph-emot`](https://huggingface.co/datasets/indonlp/nusaparagraph_emot) | none | metadata_available | manual | apache-2.0 |
| [`hf-indonlp-nusaparagraph-rhetoric`](https://huggingface.co/datasets/indonlp/nusaparagraph_rhetoric) | none | metadata_available | manual | apache-2.0 |
| [`hf-indonlp-nusaparagraph-topic`](https://huggingface.co/datasets/indonlp/nusaparagraph_topic) | none | metadata_available | manual | apache-2.0 |
| [`hf-indonlp-nusatranslation-emot`](https://huggingface.co/datasets/indonlp/nusatranslation_emot) | none | metadata_available | manual | apache-2.0 |
| [`hf-indonlp-nusatranslation-mt`](https://huggingface.co/datasets/indonlp/nusatranslation_mt) | none | metadata_available | manual | apache-2.0 |
| [`hf-indonlp-nusatranslation-senti`](https://huggingface.co/datasets/indonlp/nusatranslation_senti) | none | metadata_available | manual | apache-2.0 |
| [`hf-open-subtitles`](https://huggingface.co/datasets/open_subtitles) | none | metadata_available | manual | unknown |
| [`hf-opus100`](https://huggingface.co/datasets/opus100) | hf_parquet | native_file_index_verified | public | unknown |
| [`hf-oscar-corpus-oscar-2301`](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301) | none | metadata_available | gated | cc0-1.0 |
| [`hf-ravialdy-javanese-translated`](https://huggingface.co/datasets/ravialdy/javanese-translated) | hf_parquet | parquet_index_verified | public | unknown |
| [`hf-rifkiaputri-su-csqa`](https://huggingface.co/datasets/rifkiaputri/su-csqa) | hf_parquet | parquet_index_verified | public | cc-by-nc-sa-4.0 |
| [`hf-rifoag-javanese-sundanese-story-cloze`](https://huggingface.co/datasets/rifoag/javanese_sundanese_story_cloze) | hf_parquet | parquet_index_verified | public | unknown |
| [`hf-robinsyihab-qqpr-triplets-id`](https://huggingface.co/datasets/robinsyihab/QQPR-triplets-ID) | hf_parquet | parquet_index_verified | public | unknown |
| [`hf-sabilmakbar-indo-wiki`](https://huggingface.co/datasets/sabilmakbar/indo_wiki) | file | native_file_index_verified | public | cc-by-sa-3.0,gfdl |
| [`hf-seacrowd-casa`](https://huggingface.co/datasets/SEACrowd/casa) | none | metadata_available | manual | unknown |
| [`hf-seacrowd-emot`](https://huggingface.co/datasets/SEACrowd/emot) | none | metadata_available | manual | unknown |
| [`hf-seacrowd-facqa`](https://huggingface.co/datasets/SEACrowd/facqa) | none | metadata_available | manual | unknown |
| [`hf-seacrowd-hoasa`](https://huggingface.co/datasets/SEACrowd/hoasa) | none | metadata_available | manual | unknown |
| [`hf-seacrowd-smsa`](https://huggingface.co/datasets/SEACrowd/smsa) | none | metadata_available | manual | unknown |
| [`hf-seacrowd-tydiqa`](https://huggingface.co/datasets/SEACrowd/tydiqa) | none | metadata_available | manual | apache-2.0 |
| [`hf-senti-lex`](https://huggingface.co/datasets/senti_lex) | artifact | metadata_available | public | gpl-3.0 |
| [`hf-tapaco`](https://huggingface.co/datasets/tapaco) | hf_parquet | native_file_index_verified | public | cc-by-2.0 |
| [`hf-uonlp-culturax`](https://huggingface.co/datasets/uonlp/CulturaX) | none | metadata_available | gated | unknown |
| [`hf-wikidepia-indoparacrawl`](https://huggingface.co/datasets/Wikidepia/IndoParaCrawl) | hf_parquet | parquet_index_verified | public | unknown |
| [`id-wiktionary-org-wiki-lampiran-daftar-singkatan-dan-akronim-dalam-bahas-8cb105f5`](https://id.wiktionary.org/wiki/Lampiran:Daftar_singkatan_dan_akronim_dalam_bahasa_Indonesia#A) | none | landing_page_reachable | unverified | unknown |
| [`ilps-science-uva-nl-ilps-wp-content-uploads-sites-6-files-bahasaindonesi-803ed56f`](http://ilps.science.uva.nl/ilps/wp-content/uploads/sites/6/files/bahasaindonesia/tempo.zip) | artifact | check_failed | public_unverified | unknown |
| [`ilps-science-uva-nl-ilps-wp-content-uploads-sites-6-files-bahasaindonesi-efca86d9`](http://ilps.science.uva.nl/ilps/wp-content/uploads/sites/6/files/bahasaindonesia/kompas.zip) | artifact | check_failed | public_unverified | unknown |
| `indocloze` | none | no_download_url | manual | unknown |
| `indonesian-unsupervised-speech` | none | no_download_url | permission_required | unknown |
| [`indonlu-bapos`](https://github.com/IndoNLP/indonlu/tree/master/dataset/bapos_pos-idn) | file | reachable | public | unknown |
| [`indonlu-casa`](https://github.com/IndoNLP/indonlu/tree/ce728f6926a36174b9923dfe49d6a6839b6e9bb7/dataset/casa_absa-prosa) | file | reachable | public | unknown |
| [`indonlu-emot`](https://github.com/IndoNLP/indonlu/tree/ce728f6926a36174b9923dfe49d6a6839b6e9bb7/dataset/emot_emotion-twitter) | file | reachable | public | unknown |
| [`indonlu-facqa`](https://github.com/IndoNLP/indonlu/tree/master/dataset/facqa_qa-factoid-itb) | file | reachable | public | unknown |
| [`indonlu-hoasa`](https://github.com/IndoNLP/indonlu/tree/ce728f6926a36174b9923dfe49d6a6839b6e9bb7/dataset/hoasa_absa-airy) | file | reachable | public | unknown |
| [`indonlu-keps`](https://github.com/IndoNLP/indonlu/tree/master/dataset/keps_keyword-extraction-prosa) | file | reachable | public | unknown |
| [`indonlu-nergrit`](https://github.com/IndoNLP/indonlu/tree/master/dataset/nergrit_ner-grit) | file | reachable | public | unknown |
| [`indonlu-nerp`](https://github.com/IndoNLP/indonlu/tree/master/dataset/nerp_ner-prosa) | file | reachable | public | unknown |
| [`indonlu-posp`](https://github.com/IndoNLP/indonlu/tree/master/dataset/posp_pos-prosa) | file | reachable | public | unknown |
| [`indonlu-terma`](https://github.com/IndoNLP/indonlu/tree/master/dataset/terma_term-extraction-airy) | file | reachable | public | unknown |
| [`indonlu-wrete`](https://github.com/IndoNLP/indonlu/tree/master/dataset/wrete_entailment-ui) | file | reachable | public | unknown |
| [`lindat-mff-cuni-cz-repository-xmlui-handle-11858-00-097c-0000-0005-bf85--74c0bf35`](https://lindat.mff.cuni.cz/repository/xmlui/handle/11858/00-097C-0000-0005-BF85-F) | none | landing_page_reachable | manual | unknown |
| [`medium-com-puspitakaban-pos-tagging-bahasa-indonesia-dengan-flair-nlp-c1-2e5a88a5`](https://medium.com/@puspitakaban/pos-tagging-bahasa-indonesia-dengan-flair-nlp-c12e45542860) | none | access_denied | unverified | unknown |
| `minangnlp` | none | no_download_url | manual | unknown |
| [`nusax-ace`](https://github.com/IndoNLP/nusax/tree/75b0be5d982348332509f296e10d9bf9cb66d60f/datasets/sentiment/acehnese) | file | reachable | public | cc-by-sa-4.0 |
| [`nusax-ban`](https://github.com/IndoNLP/nusax/tree/75b0be5d982348332509f296e10d9bf9cb66d60f/datasets/sentiment/balinese) | file | reachable | public | cc-by-sa-4.0 |
| [`nusax-bbc`](https://github.com/IndoNLP/nusax/tree/75b0be5d982348332509f296e10d9bf9cb66d60f/datasets/sentiment/toba_batak) | file | reachable | public | cc-by-sa-4.0 |
| [`nusax-bjn`](https://github.com/IndoNLP/nusax/tree/75b0be5d982348332509f296e10d9bf9cb66d60f/datasets/sentiment/banjarese) | file | reachable | public | cc-by-sa-4.0 |
| [`nusax-bug`](https://github.com/IndoNLP/nusax/tree/75b0be5d982348332509f296e10d9bf9cb66d60f/datasets/sentiment/buginese) | file | reachable | public | cc-by-sa-4.0 |
| [`nusax-en`](https://github.com/IndoNLP/nusax/tree/75b0be5d982348332509f296e10d9bf9cb66d60f/datasets/sentiment/english) | file | reachable | public | cc-by-sa-4.0 |
| [`nusax-id`](https://github.com/IndoNLP/nusax) | registered | reachable | public | cc-by-sa-4.0 |
| [`nusax-jv`](https://github.com/IndoNLP/nusax) | registered | reachable | public | cc-by-sa-4.0 |
| [`nusax-lexicon-ace`](https://github.com/IndoNLP/nusax/tree/75b0be5d982348332509f296e10d9bf9cb66d60f/datasets/lexicon) | file | reachable | public | cc-by-sa-4.0 |
| [`nusax-lexicon-ban`](https://github.com/IndoNLP/nusax/tree/75b0be5d982348332509f296e10d9bf9cb66d60f/datasets/lexicon) | file | reachable | public | cc-by-sa-4.0 |
| [`nusax-lexicon-bbc`](https://github.com/IndoNLP/nusax/tree/75b0be5d982348332509f296e10d9bf9cb66d60f/datasets/lexicon) | file | reachable | public | cc-by-sa-4.0 |
| [`nusax-lexicon-bjn`](https://github.com/IndoNLP/nusax/tree/75b0be5d982348332509f296e10d9bf9cb66d60f/datasets/lexicon) | file | reachable | public | cc-by-sa-4.0 |
| [`nusax-lexicon-bug`](https://github.com/IndoNLP/nusax/tree/75b0be5d982348332509f296e10d9bf9cb66d60f/datasets/lexicon) | file | reachable | public | cc-by-sa-4.0 |
| [`nusax-lexicon-en`](https://github.com/IndoNLP/nusax/tree/75b0be5d982348332509f296e10d9bf9cb66d60f/datasets/lexicon) | file | reachable | public | cc-by-sa-4.0 |
| [`nusax-lexicon-jv`](https://github.com/IndoNLP/nusax/tree/75b0be5d982348332509f296e10d9bf9cb66d60f/datasets/lexicon) | file | reachable | public | cc-by-sa-4.0 |
| [`nusax-lexicon-mad`](https://github.com/IndoNLP/nusax/tree/75b0be5d982348332509f296e10d9bf9cb66d60f/datasets/lexicon) | file | reachable | public | cc-by-sa-4.0 |
| [`nusax-lexicon-min`](https://github.com/IndoNLP/nusax/tree/75b0be5d982348332509f296e10d9bf9cb66d60f/datasets/lexicon) | file | reachable | public | cc-by-sa-4.0 |
| [`nusax-lexicon-nij`](https://github.com/IndoNLP/nusax/tree/75b0be5d982348332509f296e10d9bf9cb66d60f/datasets/lexicon) | file | reachable | public | cc-by-sa-4.0 |
| [`nusax-lexicon-su`](https://github.com/IndoNLP/nusax/tree/75b0be5d982348332509f296e10d9bf9cb66d60f/datasets/lexicon) | file | reachable | public | cc-by-sa-4.0 |
| [`nusax-mad`](https://github.com/IndoNLP/nusax/tree/75b0be5d982348332509f296e10d9bf9cb66d60f/datasets/sentiment/madurese) | file | reachable | public | cc-by-sa-4.0 |
| [`nusax-min`](https://github.com/IndoNLP/nusax/tree/75b0be5d982348332509f296e10d9bf9cb66d60f/datasets/sentiment/minangkabau) | file | reachable | public | cc-by-sa-4.0 |
| [`nusax-mt`](https://github.com/IndoNLP/nusax/tree/main/datasets/mt) | file | reachable | public | cc-by-sa-4.0 |
| [`nusax-nij`](https://github.com/IndoNLP/nusax/tree/75b0be5d982348332509f296e10d9bf9cb66d60f/datasets/sentiment/ngaju) | file | reachable | public | cc-by-sa-4.0 |
| [`nusax-su`](https://github.com/IndoNLP/nusax) | registered | reachable | public | cc-by-sa-4.0 |
| [`openslr-org-35-62ac7cb1`](https://openslr.org/35) | none | landing_page_reachable | manual | unknown |
| [`openslr-org-36-c2ebc482`](https://openslr.org/36) | none | landing_page_reachable | manual | unknown |
| [`openwebtext-id`](https://huggingface.co/datasets/irfanfadhullah/OpenWebText-Indonesia-10k) | registered | pinned_metadata_verified | public | unknown |
| [`opus-nlpl-eu-bible-uedin-php-3c469bb6`](https://opus.nlpl.eu/bible-uedin.php) | none | not_found | unverified | unknown |
| [`opus-nlpl-eu-cb43cb13`](http://opus.nlpl.eu) | none | landing_page_reachable | unverified | unknown |
| [`opus-nlpl-eu-qed-en-es-v2-0a-qed-d20ceb2d`](https://opus.nlpl.eu/QED/en&es/v2.0a/QED) | none | not_found | unverified | unknown |
| [`opus-nlpl-eu-qed-php-458fa591`](https://opus.nlpl.eu/QED.php) | none | not_found | unverified | unknown |
| [`research-nii-ac-jp-src-en-register-html-ff0653bf`](http://research.nii.ac.jp/src/en/register.html) | none | landing_page_reachable | permission_required | research-only |
| [`research-nii-ac-jp-src-en-titml-idn-html-0b78b636`](http://research.nii.ac.jp/src/en/TITML-IDN.html) | none | landing_page_reachable | permission_required | research-only |
| [`simple-summaries-id`](https://huggingface.co/datasets/irfanfadhullah/indonesian-simple-summaries) | registered | pinned_metadata_verified | public | apache-2.0 |
| [`smsa`](https://github.com/IndoNLP/indonlu) | registered | reachable | public | unknown |
| [`sourceforge-net-p-wn-msa-tab-head-tree-trunk-bfebb30c`](https://sourceforge.net/p/wn-msa/tab/HEAD/tree/trunk) | none | landing_page_reachable | manual | unknown |
| [`stor-akmal-dev-cc12m-jsonl-zst-afb610dd`](https://stor.akmal.dev/cc12m.jsonl.zst) | file | check_failed | public_unverified | unknown |
| [`stor-akmal-dev-cc3m-train-jsonl-zst-5ec1f888`](https://stor.akmal.dev/cc3m-train.jsonl.zst) | file | check_failed | public_unverified | unknown |
| [`stor-akmal-dev-ccnews-id-tar-f3176761`](https://stor.akmal.dev/ccnews-id.tar) | artifact | check_failed | public_unverified | unknown |
| [`stor-akmal-dev-gigaword-tar-zst-2378f678`](https://stor.akmal.dev/gigaword.tar.zst) | artifact | check_failed | public_unverified | unknown |
| [`stor-akmal-dev-gtts-500k-zip-04a47ee1`](https://stor.akmal.dev/gtts-500k.zip) | artifact | check_failed | public_unverified | unknown |
| [`stor-akmal-dev-idmultinli-f7e78300`](https://stor.akmal.dev/idmultinli) | none | check_failed | unverified | unknown |
| [`stor-akmal-dev-idsnli-6606967d`](https://stor.akmal.dev/idsnli) | none | check_failed | unverified | unknown |
| [`stor-akmal-dev-jv-text-d6c1b671`](https://stor.akmal.dev/jv-text) | none | check_failed | unverified | unknown |
| [`stor-akmal-dev-kaskus-jsonl-zst-d8c56968`](https://stor.akmal.dev/kaskus.jsonl.zst) | file | check_failed | public_unverified | unknown |
| [`stor-akmal-dev-kaskus-webtext-tar-zst-c9486a63`](https://stor.akmal.dev/kaskus-webtext.tar.zst) | artifact | check_failed | public_unverified | unknown |
| [`stor-akmal-dev-parabank-v2-0-jsonl-zst-33aa7daa`](https://stor.akmal.dev/parabank-v2.0.jsonl.zst) | file | check_failed | public_unverified | unknown |
| [`stor-akmal-dev-paranmt-5m-jsonl-zst-7badd9ca`](https://stor.akmal.dev/paranmt-5m.jsonl.zst) | file | check_failed | public_unverified | unknown |
| [`stor-akmal-dev-reddit-tldr-jsonl-zst-db426fe8`](https://stor.akmal.dev/reddit-tldr.jsonl.zst) | file | check_failed | public_unverified | unknown |
| [`stor-akmal-dev-squad-18768da1`](https://stor.akmal.dev/squad) | none | check_failed | unverified | unknown |
| [`stor-akmal-dev-twitter-dump-30f9aaf2`](https://stor.akmal.dev/twitter-dump) | none | check_failed | unverified | unknown |
| [`stor-akmal-dev-wikihow-json-zst-4d7addf4`](https://stor.akmal.dev/wikihow.json.zst) | file | check_failed | public_unverified | unknown |
| [`stor-akmal-dev-wikipedia-links-tar-zst-c7315ef9`](https://stor.akmal.dev/wikipedia-links.tar.zst) | artifact | check_failed | public_unverified | unknown |
| [`stor-akmal-dev-yfcc100-jsonl-zst-6e368503`](https://stor.akmal.dev/yfcc100.jsonl.zst) | file | check_failed | public_unverified | unknown |
| [`storage-depia-wiki-europarl-jsonl-zst-a608d01f`](https://storage.depia.wiki/europarl.jsonl.zst) | file | check_failed | public_unverified | unknown |
| [`storage-depia-wiki-europat-jsonl-zst-8a98edff`](https://storage.depia.wiki/europat.jsonl.zst) | file | check_failed | public_unverified | unknown |
| [`storage-depia-wiki-sbert-paraphrase-951c43dd`](https://storage.depia.wiki/sbert-paraphrase) | none | check_failed | unverified | unknown |
| [`storage-googleapis-com-babert-pretraining-indonlu-finals-dataset-preproc-922a65b8`](https://storage.googleapis.com/babert-pretraining/IndoNLU_finals/dataset/preprocessed/dataset_all_uncased_blankline.txt.xz) | file | access_denied | public_unverified | unknown |
| [`traces1-inria-fr-oscar-012ac415`](https://traces1.inria.fr/oscar/#corpus) | none | access_denied | unverified | unknown |
| [`twitter-com-pelangipuisi-1d4b8a2b`](https://twitter.com/PelangiPuisi) | none | landing_page_reachable | unverified | unknown |
| [`wit3-fbk-eu-mt-php-d6ae1108`](https://wit3.fbk.eu/mt.php?release=2017-01-more) | none | landing_page_reachable | unverified | unknown |
| [`www-cs-utexas-edu-users-ml-clamp-videodescription-youtubeclips-tar-76c25985`](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/YouTubeClips.tar) | artifact | reachable | public_unverified | unknown |
| [`www-dropbox-com-s-12h3qqog6q4bjve-panx-dataset-tar-98ee9a3b`](https://www.dropbox.com/s/12h3qqog6q4bjve/panx_dataset.tar?dl=1) | artifact | reachable | public_unverified | unknown |
| [`www-kaggle-com-datasets-andikawilliam-clickid-56e212fb`](https://www.kaggle.com/datasets/andikawilliam/clickid) | none | not_found | credentials_required | unknown |
| [`www-kaggle-com-datasets-hakikidamana-javanese-corpus-data-cf60dec8`](https://www.kaggle.com/datasets/hakikidamana/javanese-corpus/data) | none | not_found | credentials_required | unknown |
| [`www-kaggle-com-datasets-hengkymulyono-indonesian-speech-with-accents-5-e-0aa926f5`](https://www.kaggle.com/datasets/hengkymulyono/indonesian-speech-with-accents-5-ethnic-groups) | none | not_found | credentials_required | unknown |
| [`www-kaggle-com-datasets-rtatman-sentiment-lexicons-for-81-languages-10748fb4`](https://www.kaggle.com/datasets/rtatman/sentiment-lexicons-for-81-languages) | none | not_found | credentials_required | unknown |
| [`www-researchgate-net-publication-338409000-dataset-indonesia-untuk-anali-de410700`](https://www.researchgate.net/publication/338409000_Dataset_Indonesia_untuk_Analisis_Sentimen) | none | access_denied | unverified | unknown |
| [`www2-nict-go-jp-astrec-att-member-mutiyama-alt-35939d79`](http://www2.nict.go.jp/astrec-att/member/mutiyama/ALT) | none | landing_page_reachable | unverified | unknown |
| [`zenodo-org-records-15494944-6394b306`](https://zenodo.org/records/15494944) | none | landing_page_reachable | manual | unknown |
| [`zenodo-org-records-841984-838904f2`](https://zenodo.org/records/841984) | none | landing_page_reachable | manual | unknown |

Regenerate with `python scripts/report_coverage.py` after refreshing the audit.
