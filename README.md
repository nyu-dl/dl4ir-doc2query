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
to predict queries from documents.
```
git clone https://github.com/OpenNMT/OpenNMT-py
cd OpenNMT-py
pip install -r requirements.txt
cd ..
```

And [Anserini](https://github.com/castorini/Anserini), so we can index and 
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

The last step is to preprecess train and dev files with the following command:
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
        -world_size 2 \
        -share_embeddings \
        -gpu_ranks 0 1
```

The command above starts training a transformer model using two GPUs (you can 
train with one GPU by setting `gpu_ranks 0` and `world_size 1`). It should
take approximately 2-4 hours to reach iteration 15,000, which contains the 
lowest perplexity on the dev set (~15.7).

We can now evaluate in BLEU points the performance of our model:

```
python ./OpenNMT-py/translate.py \
                     -gpu 0 \
                     -model ${DATA_DIR}/doc2query_step_15000.pt \
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
BLEU = 7.78, 32.6/13.1/4.9/2.5 (BP=0.915, ratio=0.918, hyp_len=40528, ref_len=44138)
```

In case you don't want to train a doc2query model yourself, you can [download our trained model here](https://drive.google.com/open?id=1ieQ4-d2zvxAF7iYqC_9sFYQCea1xPINU).

### Predicting Queries

We use our best model checkpoint (iteration 15,000) to predict 5 queries for
each document in the collection:
```
python ./OpenNMT-py/translate.py \
  -gpu 0 \
  -model ${DATA_DIR}/doc2query_step_15000.pt \
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

In any case, you can [download the predicted queries here](https://drive.google.com/file/d/1DyAkwwHUIE7Yk_svtTUZBlAtlPDA7u7B/view?usp=sharing).


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

Since queries of the set are too many (+100k), it would take a long time to retrieve all of them. To speed this up, we use only the queries that are in the qrels file: 

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
MRR @10: 0.21579006913175935
QueriesRanked: 6980
```

In case you want to compare your retrieved docs against ours, you can [download our retrieved docs here](https://drive.google.com/open?id=11dHqA0VBk6oHTW6HDHWR9qtBBQqTfHSI).

### Reranking with BERT

Coming soon.


#### How do I cite this work?
```
@article{nogueira2019document,
  title={Document Expansion by Query Prediction},
  author={Nogueira, Rodrigo and Yang, Wei and Lin, Jimmy and Cho, Kyunghyun},
  journal={arXiv preprint arXiv:1904.08375},
  year={2019}
}
