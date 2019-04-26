'''Converts MS MARCO doc-query pairs to the OpenNMT format.'''
import collections
import os
from absl import app
from absl import flags


FLAGS = flags.FLAGS

flags.DEFINE_string('collection_path', None, 'tsv collection file.')
flags.DEFINE_string('train_queries', None, 'training set queries.')
flags.DEFINE_string('train_qrels', None, 'training set qrels.')
flags.DEFINE_string('dev_queries', None, 'dev set queries.')
flags.DEFINE_string('dev_qrels', None, 'dev set qrels.')
flags.DEFINE_string('output_folder', None, 'Folder to write the OpenNMT files.')


def convert_collection(collection):
  """Convert documents into source sentences."""
  print('Converting collection...')
  src_file = open(os.path.join(FLAGS.output_folder, 'src-collection.txt'), 'w')
  for i, (doc_id, doc_text) in enumerate(collection.items()):
    src_file.write(doc_text + '\n')
    if i % 1000000 == 0:
      print('Converting collection, doc {} of {}'.format(i, len(collection)))

  src_file.close()


def convert_dataset(collection, queries, qrels, set_name):
  """Convert queries and relevant documents into source and target sentences."""
  print('Converting {} set...'.format(set_name))

  src_file = open(
      os.path.join(FLAGS.output_folder, 'src-{}.txt'.format(set_name)), 'w')
  tgt_file = open(
      os.path.join(FLAGS.output_folder, 'tgt-{}.txt'.format(set_name)), 'w')

  for i, (query_id, relevant_doc_ids) in enumerate(qrels.items()):
    query = queries[query_id]
    for doc_id in relevant_doc_ids:
      doc = collection[doc_id]
      src_file.write(doc + '\n')
      tgt_file.write(query + '\n')

      if i % 100000 == 0:
        print('Converting {} set, query {} of {}'.format(
            set_name, i, len(qrels)))

  src_file.close()
  tgt_file.close()


def load_qrels(path):
  """Loads qrels into a dict of key: query id, value: set of relevant doc ids."""
  qrels = collections.defaultdict(set)
  with open(path) as f:
    for i, line in enumerate(f):
      query_id, _, doc_id, relevance = line.rstrip().split('\t')
      if int(relevance) >= 1:
        qrels[query_id].add(doc_id)
      if i % 100000 == 0:
        print('Loading qrels {}'.format(i))
  return qrels


def load_queries(path):
  """Loads queries into a dict of key: query id, value: query text."""
  queries = {}
  with open(path) as f:
    for i, line in enumerate(f):
      query_id, query = line.rstrip().split('\t')
      queries[query_id] = query
      if i % 100000 == 0:
        print('Loading queries {}'.format(i))
  return queries


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
  if not os.path.exists(FLAGS.output_folder):
    os.makedirs(FLAGS.output_folder)

  collection = load_collection(path=FLAGS.collection_path)
  convert_collection(collection)

  for set_name, queries_path, qrels_path in zip(
      ['train', 'dev'],
      [FLAGS.train_queries, FLAGS.dev_queries], 
      [FLAGS.train_qrels, FLAGS.dev_qrels]):

    queries = load_queries(path=queries_path)
    qrels = load_qrels(path=qrels_path)

    convert_dataset(
        collection=collection, queries=queries, qrels=qrels, set_name=set_name)

  print('Done!')


if __name__ == '__main__':
  flags.mark_flag_as_required('collection_path')
  flags.mark_flag_as_required('train_queries')
  flags.mark_flag_as_required('train_qrels')
  flags.mark_flag_as_required('dev_queries')
  flags.mark_flag_as_required('dev_qrels')
  flags.mark_flag_as_required('output_folder')
  app.run(main)
