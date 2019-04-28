## Doc2query: Document Expansion by Query Prediction

This repository contains the code to reproduce our entry to the [MSMARCO passage
ranking task](http://www.msmarco.org/leaders.aspx), which was placed first on April 8th, 2019.

MSMARCO Passage Re-Ranking Leaderboard (Apr 8th 2019) | Eval MRR@10  | Dev MRR@10
------------------------------------- | :------: | :------:
1st Place - BERTter Indexing (this code) | **36.8** | **37.5**
2nd Place - SAN + BERT base              | 35.9     | 37.0
3rd Place - BERT + Small Training        | 35.9     | 36.5

The paper describing our implementation is [here](https://arxiv.org/pdf/1904.08375.pdf).


### Installation

We first need to install [OpenNMT](http://opennmt.net/) so we can train a model
to predict queries from documents. We clone from 0.8.2 because that was the 
version we trained our models. However, feel free to use a newer version, but we
cannot guarantee that the commands below will work.

```
git clone --branch 0.8.2 https://github.com/OpenNMT/OpenNMT-py.git
cd OpenNMT-py
pip install -r requirements.txt
cd ..
```

We also need to install [Anserini](https://github.com/castorini/Anserini), so we can index and 
retrieve the expanded documents.

```
sudo apt-get install maven
git clone https://github.com/castorini/Anserini.git
cd Anserini
mvn clean package appassembler:assemble
tar xvfz eval/trec_eval.9.0.4.tar.gz -C eval/ && cd eval/trec_eval.9.0.4 && make
cd ../ndeval && make
cd ../../../
```

### Data Preprocessing

First, we need to download and extract the MS MARCO dataset:

```
DATA_DIR=./msmarco_data
mkdir ${DATA_DIR}

wget https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz -P ${DATA_DIR}

tar -xvf ${DATA_DIR}/collectionandqueries.tar.gz -C ${DATA_DIR}
```

To confirm, `collectionandqueries.tar.gz` should have MD5 checksum of `fed5aa512935c7b62787cb68ac9597d6`.

The scripts below convert the data to a format that can be consumed by
OpenNMT training and inference scripts:
```
python ./convert_msmarco_to_opennmt.py \
    --collection_path=${DATA_DIR}/collection.tsv \
    --train_queries=${DATA_DIR}/queries.train.tsv \
    --train_qrels=${DATA_DIR}/qrels.train.tsv \
    --dev_queries=${DATA_DIR}/queries.dev.tsv \
    --dev_qrels=${DATA_DIR}/qrels.dev.small.tsv \
    --output_folder=${DATA_DIR}/opennmt_format
```

The output files and their number of lines should be:
```
$ wc -l ./msmarco_data/opennmt_format/*
  8841823 ./msmarco_data/opennmt_format/src-collection.txt
     7437 ./msmarco_data/opennmt_format/src-dev.txt
   532751 ./msmarco_data/opennmt_format/src-train.txt
     7437 ./msmarco_data/opennmt_format/tgt-dev.txt
   532751 ./msmarco_data/opennmt_format/tgt-train.txt
```

The last step is to preprocess train and dev files with the following command:
```
python ./OpenNMT-py/preprocess.py \
  -train_src ${DATA_DIR}/opennmt_format/src-train.txt \
  -train_tgt ${DATA_DIR}/opennmt_format/tgt-train.txt \
  -valid_src ${DATA_DIR}/opennmt_format/src-dev.txt \
  -valid_tgt ${DATA_DIR}/opennmt_format/tgt-dev.txt \
  -save_data ${DATA_DIR}/opennmt_format/preprocessed \
  -src_seq_length 10000 \
  -tgt_seq_length 10000 \
  -src_seq_length_trunc 400 \
  -tgt_seq_length_trunc 100 \
  -dynamic_dict \
  -share_vocab \
  -src_vocab_size 32000 \
  -tgt_vocab_size 32000 \
  -shard_size 100000
```

### Training doc2query (i.e. a transformer model)

```
python -u /scratch/rfn216/doc2query/git/dl4ir-doc2query/OpenNMT-py/train.py  \
        -data ${DATADIR}/preprocessed \
        -save_model ${RUNDIR}/model \
        -layers 6 \
        -rnn_size 512 \
        -word_vec_size 512 \
        -transformer_ff 2048 \
        -heads 8 \
        -encoder_type transformer \
        -decoder_type transformer \
        -position_encoding \
        -train_steps ${TRAIN_STEPS} \
        -max_generator_batches 2 \
        -dropout 0.1 \
        -batch_size 4096 \
        -batch_type tokens \
        -normalization tokens \
        -accum_count 2 \
        -optim adam \
        -adam_beta2 0.998 \
        -decay_method noam \
        -warmup_steps 8000 \
        -learning_rate 2.0 \
        -max_grad_norm 0.0 \
        -param_init 0.0 \
        -param_init_glorot \
        -label_smoothing 0.1 \
        -valid_steps 5000 \
        -save_checkpoint_steps 5000 \
        -world_size 4 \
        -share_embeddings \
        -gpu_ranks 0 1 2 3
```

The command above starts training a transformer model using four GPUs (you can 
train with one GPU by setting `gpu_ranks 0` and `world_size 1`). It should
take approximately 3-6 hours to reach iteration 10,000, which contains the 
lowest perplexity on the dev set (~15.2).

We can now evaluate in BLEU points the performance of our model:

```
python ./OpenNMT-py/translate.py \
                     -gpu 0 \
                     -model ${DATA_DIR}/doc2query_step_10000.pt \
                     -src ${DATA_DIR}/opennmt_format/src-dev.txt \
                     -tgt ${DATA_DIR}/opennmt_format/tgt-dev.txt \
                     -output ${DATA_DIR}/opennmt_format/pred-dev.txt \
                     -replace_unk \
                     -verbose \
                     -report_time \
                     -beam_size 1

perl ./OpenNMT-py/tools/multi-bleu.perl \
    ${DATA_DIR}/opennmt_format/tgt-dev.txt < ${DATA_DIR}/opennmt_format/pred-dev.txt
```

The output should be similar to this:

```
BLEU = 8.82, 35.0/14.3/5.7/2.5 (BP=0.957, ratio=0.958, hyp_len=34050, ref_len=35553)
```

In case you don't want to train a doc2query model yourself, you can 
[download our trained model here](https://drive.google.com/open?id=1tHL3ZcXBSqBcavpusQB0otxdOYDNW_rV).

### Predicting Queries

We use our best model checkpoint (iteration 10,000) to predict 5 queries for
each document in the collection:
```
python ./OpenNMT-py/translate.py \
  -gpu 0 \
  -model ${DATA_DIR}/doc2query_step_10000.pt \
  -src ${DATADIR}/opennmt_format/src-collection.txt \
  -output ${DATADIR}/opennmt_format/pred-collection_beam5.txt \
  -batch_size 32 \
  -beam_size 5 \
  --n_best 5 \
  -replace_unk \
  -report_time
```

The step above takes many hours. Alternatively, you can split 
`src-collection.txt` into multiple files, process them in parallel and merge the
output. For example:

```
# Split in 9 files, each with a 1M docs.
split -l 1000000 --numeric-suffixes ${DATA_DIR}/opennmt_format/src-collection.txt ${DATA_DIR}/opennmt_format/src-collection.txt

...
# Execute 9 translate.py in parallel, one for each split file.
...

# Merge the predictions into a single file.
cat ${DATA_DIR}/opennmt_format/pred-collection_beam5.txt?? > ${DATA_DIR}/opennmt_format/pred-collection_beam5.txt
```

In any case, you can [download the predicted queries here](https://drive.google.com/open?id=1WGeeMEI6Ol5ECP_XW0G35O3shPrJD6vJ).


### Expanding docs
Next, we need to merge the original documents with the predicted queries into
Anserini's jsonl files (which have one json object per line):

```
python ./convert_collection_to_jsonl.py \
    --collection_path=${DATA_DIR}/collection.tsv \
    --predictions=${DATA_DIR}/opennmt_format/pred-collection_beam5.txt \
    --beam_size=5 \
    --output_folder=${DATA_DIR}/collection_jsonl
```

The above script should generate 9 jsonl files in ${DATA_DIR}/collection_jsonl, each with 1M lines/docs (except for the last one, which should have 841,823 lines).

We can now index these docs as a `JsonCollection` using Anserini:

```
sh ./Anserini/target/appassembler/bin/IndexCollection -collection JsonCollection \
 -generator LuceneDocumentGenerator -threads 9 -input ${DATA_DIR}/collection_jsonl \
 -index ${DATA_DIR}/lucene-index-msmarco -optimize
```

The output message should be something like this:

```
2019-04-26 07:49:14,549 INFO  [main] index.IndexCollection (IndexCollection.java:647) - Total 8,841,823 documents indexed in 00:06:02
```

Your speed may vary... with a modern desktop machine with an SSD, indexing takes around a minute.

## Retrieving and Evaluating the Dev set

Since queries of the dev set are too many (+100k), it would take a long time to retrieve all of them. To speed this up, we use only the queries that are in the qrels file: 

```
python ./Anserini/src/main/python/msmarco/filter_queries.py --qrels=${DATA_DIR}/qrels.dev.small.tsv \
 --queries=${DATA_DIR}/queries.dev.tsv --output_queries=${DATA_DIR}/queries.dev.small.tsv
```

The output queries file should contain 6980 lines.
```
$ wc -l ${DATA_DIR}/queries.dev.small.tsv
6980 /scratch/rfn216/msmarco_data//queries.dev.small.tsv
```

We can now retrieve this smaller set of queries.

```
cd Anserini
python ./src/main/python/msmarco/retrieve.py --index ${DATA_DIR}/lucene-index-msmarco \
 --qid_queries ${DATA_DIR}/queries.dev.small.tsv --output ${DATA_DIR}/run.dev.small.tsv --hits 1000
cd ..
```

Retrieval speed will vary by machine:
On a modern desktop with an SSD, we can get ~0.04 seconds per query (taking about five minutes).
On a slower machine with mechanical disks, the entire process might take as long as a couple of hours.
The option `-hits` specifies the of documents per query to be retrieved.
Thus, the output file should have approximately 6980 * 1000 = 6.9M lines. 


Finally, we can evaluate the retrieved documents using this the official MS MARCO evaluation script: 

```
python ./src/main/python/msmarco/msmarco_eval.py ${DATA_DIR}/qrels.dev.small.tsv ${DATA_DIR}/run.dev.small.tsv
```

And the output should be like this:

```
#####################
MRR @10: 0.22155540774093688
QueriesRanked: 6980
#####################
```

Note that these results are 0.6 higher than the ones in the paper. This is due 
to better BM25 tuning (b1=0.8, k=0.6).

In case you want to compare your retrieved docs against ours, you can 
[download our retrieved docs here](https://drive.google.com/file/d/1uW2JF5aXDTjlKUnMQttXrCPo5pqjEphk/view?usp=sharing).

## Reranking with BERT

Most of the gains come from re-ranking with BERT the passages retrieved with
BM25 + Doc2query. To implement BERT re-ranker, we follow the same procedure 
described in the
[BERT for Passage Re-ranking repository](https://github.com/nyu-dl/dl4marco-bert).

We first need to convert dev queries and retrieved docs into the TFRecord format
that will be consumed by BERT:

```
python convert_msmarco_to_tfrecord.py \
  --output_folder=${DATA_DIR}/bert_tfrecord \
  --collection_path=${DATA_DIR}/collection.tsv \
  --vocab=vocab.txt \
  --queries=${DATA_DIR}/queries.dev.small.tsv \
  --run=${DATA_DIR}/run.dev.small.tsv \
  --qrels=${DATA_DIR}/qrels.dev.small.tsv
```

This script above produces the files `dataset.tf` and `query_doc_ids.txt`, and 
they should be moved to a folder in the [Google Cloud Storage](https://cloud.google.com/storage/). For you convenience, you can [download these files here](https://drive.google.com/file/d/1oh2UwqDuXhqGutIqRWxglBSQQ9neb4b8/view?usp=sharing).

We are now ready to use our [Google's Colab to re-rank with BERT](https://colab.research.google.com/drive/1NXJZ5TaBj_i_g_0KxzJ9ZMsjn310h2YQ).

Because we did not see any difference from training BERT with the expanded vs 
original docs, we simple re-rank dev queries using the 
[same checkpoint](https://drive.google.com/open?id=1crlASTMlsihALlkabAQP6JTYIZwC1Wm8)
from the [BERT for Passage Re-ranking repository](https://github.com/nyu-dl/dl4marco-bert), 
that is, no training is required in this step.

The Colab is configured to use TPUs and it should take 5-10 hours to re-rank all
6980 dev set queries. If you use a GPU, expect this step to be 10x longer. 

After it finishes, we can download the run file `msmarco_predictions_dev.tsv` 
(which is in the Google Storage folder you specified in OUTPUT_DIR) and evaluate it:

```
python ./src/main/python/msmarco/msmarco_eval.py ${DATA_DIR}/qrels.dev.small.tsv ${DATA_DIR}/msmarco_predictions_dev.tsv
```

The output should be like this:
```
#####################
MRR @10: 0.3763750170555333
QueriesRanked: 6980
#####################
```

Note that this MRR@10 is slightly higher than our leadearboard entry, 
probably because the better tuned BM25.

You can [download our run file here](https://drive.google.com/file/d/1H5mNO6z1ZR47pGkEFDF5Dcbmfd7le3Cc/view?usp=sharing).


#### How do I cite this work?
```
@article{nogueira2019document,
  title={Document Expansion by Query Prediction},
  author={Nogueira, Rodrigo and Yang, Wei and Lin, Jimmy and Cho, Kyunghyun},
  journal={arXiv preprint arXiv:1904.08375},
  year={2019}
}
