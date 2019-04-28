"""Converts MS MARCO queries and corpus into TFRecord that will be consumed by BERT.
The main necessary inputs are:
- Passage Collection (tsv file) 
- Pairs of Query-Relevant Passage (called qrels in TREC's nomenclature)
- Pairs of Query-Candidate Passage (called run in TREC's nomenclature)
The outputs is a TFRecord file and a text file with query id and doc id mappings.
"""
import collections
import os
import tensorflow as tf
import time
import tokenization


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "output_folder", None, "Folder where the TFRecord files will be writen.")

flags.DEFINE_string(
    "vocab", None, "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "collection_path", None,
    "Path to the tsv file containing the MS MARCO documents.")

flags.DEFINE_string(
    "queries", None, "Path to the <query id; query text> pairs.")

flags.DEFINE_string(
    "run", None, "Path to the query id / candidate doc ids pairs.")

flags.DEFINE_string(
    "qrels", None, "Path to the query id / relevant doc ids pairs.")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum query sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated.")

flags.DEFINE_integer(
    "num_docs", 1000, "The number of docs per query.")


def convert_dataset(data, collection, tokenizer):

  ids_file = open(FLAGS.output_folder + '/query_doc_ids.txt', 'w')
  output_path = FLAGS.output_folder + '/dataset.tf'

  start_time = time.time()

  random_title = list(collection.keys())[0]

  with tf.python_io.TFRecordWriter(output_path) as writer:
    for i, query_id in enumerate(data):
      query, qrels, doc_titles = data[query_id]
      
      query = tokenization.convert_to_unicode(query)
      query_ids = tokenization.convert_to_bert_input(
          text=query, 
          max_seq_length=FLAGS.max_query_length,
          tokenizer=tokenizer, 
          add_cls=True)

      query_ids_tf = tf.train.Feature(
          int64_list=tf.train.Int64List(value=query_ids))

      doc_titles = doc_titles[:FLAGS.num_docs]

      # Add fake docs so we always have max_docs per query.
      doc_titles += max(0, FLAGS.num_docs - len(doc_titles)) * [random_title]

      labels = [
          1 if doc_title in qrels else 0 
          for doc_title in doc_titles
      ]

      if i % 1000 == 0:
        print('query: {}; len qrels: {}'.format(query, len(qrels)))
        print('sum labels: {}'.format(sum(labels)))
        for j, (label, doc_title) in enumerate(zip(labels, doc_titles)):
          print('doc {}, label {}, title: {}\n{}\n'.format(
              j, label, doc_title, collection[doc_title]))
        print()

      doc_token_ids = [
          tokenization.convert_to_bert_input(
              text=tokenization.convert_to_unicode(collection[doc_title]),
              max_seq_length=FLAGS.max_seq_length - len(query_ids),
              tokenizer=tokenizer,
              add_cls=False)
          for doc_title in doc_titles
      ]

      for doc_token_id, label, doc_title in zip(
          doc_token_ids, labels, doc_titles):

        ids_file.write('{}\t{}\n'.format(query_id, doc_title))

        doc_ids_tf = tf.train.Feature(
            int64_list=tf.train.Int64List(value=doc_token_id))

        labels_tf = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[label]))

        len_gt_titles_tf = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[len(qrels)]))

        features = tf.train.Features(feature={
            'query_ids': query_ids_tf,
            'doc_ids': doc_ids_tf,
            'label': labels_tf,
            'len_gt_titles': len_gt_titles_tf,
        })
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())

      if i % 1000 == 0:
        print('wrote {} of {} queries'.format(i, len(data)))
        time_passed = time.time() - start_time
        est_hours = (len(data) - i) * time_passed / (max(1.0, i) * 3600)
        print('estimated total hours to save: {}'.format(est_hours))

  ids_file.close()


def load_qrels(path):
  """Loads qrels into a dict of key: query_id, value: list of relevant doc ids."""
  qrels = collections.defaultdict(set)
  with open(path) as f:
    for i, line in enumerate(f):
      query_id, _, doc_id, relevance = line.rstrip().split('\t')
      if int(relevance) >= 1:
        qrels[query_id].add(doc_id)
      if i % 1000 == 0:
        print('Loading qrels {}'.format(i))
  return qrels


def load_queries(path):
  """Loads queries into a dict of key: query_id, value: query text."""
  queries = {}
  with open(path) as f:
    for i, line in enumerate(f):
      query_id, query = line.rstrip().split('\t')
      queries[query_id] = query
      if i % 1000 == 0:
        print('Loading queries {}'.format(i))
  return queries


def load_run(path):
  """Loads run into a dict of key: query_id, value: list of candidate doc ids."""

  # We want to preserve the order of runs so we can pair the run file with the
  # TFRecord file.
  run = collections.OrderedDict()
  with open(path) as f:
    for i, line in enumerate(f):
      query_id, doc_title, rank = line.split('\t')
      if query_id not in run:
        run[query_id] = []
      run[query_id].append((doc_title, int(rank)))
      if i % 1000000 == 0:
        print('Loading run {}'.format(i))
  # Sort candidate docs by rank.
  sorted_run = collections.OrderedDict()
  for query_id, doc_titles_ranks in run.items():
    sorted(doc_titles_ranks, key=lambda x: x[1])
    doc_titles = [doc_titles for doc_titles, _ in doc_titles_ranks]
    sorted_run[query_id] = doc_titles

  return sorted_run


def merge(qrels, run, queries):
  """Merge qrels and runs into a single dict of key: query, 
  value: tuple(relevant_doc_ids, candidate_doc_ids)"""
  data = collections.OrderedDict()
  for query_id, candidate_doc_ids in run.items():
    query = queries[query_id]
    relevant_doc_ids = set()
    if qrels:
      relevant_doc_ids = qrels[query_id]
    data[query_id] = (query, relevant_doc_ids, candidate_doc_ids)
  return data


def load_collection(path):
  """Loads tsv collection into a dict of key: doc id, value: doc text."""
  collection = {}
  with open(path) as f:
    for i, line in enumerate(f):
      doc_id, doc_text = line.rstrip().split('\t')
      collection[doc_id] = doc_text.replace('\n', ' ')
      if i % 1000000 == 0:
        print('Loading collection, doc {}'.format(i))

  return collection


def main(_):
  print('Loading Tokenizer...')
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab, do_lower_case=True)

  if not os.path.exists(FLAGS.output_folder):
    os.mkdir(FLAGS.output_folder)

  qrels = None
  if FLAGS.qrels:
    qrels = load_qrels(path=FLAGS.qrels)

  queries = load_queries(path=FLAGS.queries)
  run = load_run(path=FLAGS.run)
  data = merge(qrels=qrels, run=run, queries=queries)

  print('Loading Collection...')
  collection = load_collection(FLAGS.collection_path)

  print('Converting to TFRecord...')
  convert_dataset(data=data, collection=collection, tokenizer=tokenizer)

  print('Done!')  


if __name__ == '__main__':
  flags.mark_flag_as_required('output_folder')
  flags.mark_flag_as_required('collection_path')
  flags.mark_flag_as_required('vocab')
  flags.mark_flag_as_required('queries')
  flags.mark_flag_as_required('run')
  flags.mark_flag_as_required('output_folder')
  tf.app.run(main)
