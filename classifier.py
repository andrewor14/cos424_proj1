#!/usr/bin/env python

from preprocessSentences import clean_word, clean_words, parse_example

import argparse
import numpy as np
import sys

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-t", "--train", help="training data file", required=True)
  parser.add_argument("-b", "--bow", help="bag of words file", required=True)
  parser.add_argument("-v", "--vocab", help="vocab file", required=True)
  parser.add_argument("-e", "--test", help="test data file", required=True)
  parser.add_argument("-m", "--model", help="model", required=True)
  args = parser.parse_args()

  # Build vocabs list from train dataset
  vocabs = []
  with open(args.vocab, "r") as f:
    for line in f.readlines():
      vocabs += [clean_word(line.decode("utf-8-sig"))]

  # Do the classifying
  if args.model == 'bnb':
    bernoulli_naive_bayes(args.train, args.bow, vocabs, args.test)
  elif args.model == 'mnb':
    multinomial_naive_bayes(args.train, args.bow, vocabs, args.test)
  elif args.model == 'skbnb':
    sk_bernoulli_naive_bayes(args.train, args.bow, vocabs, args.test)
  else:
    raise Exception("Unknown model '%s' % args.model")

def sk_bernoulli_naive_bayes(train_file, bow_file, vocabs, test_file):
  train_labels = []
  bow_data = []
  with open(train_file, "r") as f:
    for line in f.readlines():
      train_labels += [int(line.split()[-1])]
  with open(bow_file, "r") as f:
    for line in f.readlines():
      bow_data += [[int(v) for v in line.split(",")]]
  train_labels = np.array(train_labels)
  bow_data = np.array(bow_data)
  from sklearn.naive_bayes import BernoulliNB
  clf = BernoulliNB()
  clf.fit(bow_data, train_labels)
  # Test it!
  vocab_index = {}
  log_threshold = 10
  num_test_examples = 0
  num_test_correct = 0
  for i, word in enumerate(vocabs):
    vocab_index[word] = i
  with open(test_file, "r") as f:
    for i, line in enumerate(f.readlines()):
      expected_label = int(line.split()[-1])
      # Turn sentences into bag of word vectors
      word_vector = [0] * len(vocabs)
      words = clean_words(parse_example(line).split())
      for word in words:
        index = vocab_index.get(word)
        if index > 0:
          word_vector[index] += 1
      word_vector = np.array(word_vector)
      # Do the prediction
      predicted_label = clf.predict([word_vector])[0]
      if predicted_label == expected_label:
        num_test_correct += 1
      num_test_examples += 1
      if i < log_threshold:
        print_classify_example(line, words, predicted_label, expected_label)
  print_result(num_test_correct, num_test_examples)

def bernoulli_naive_bayes(train_file, bow_file, vocabs, test_file):
  manual_naive_bayes(train_file, bow_file, vocabs, test_file, bernoulli=True)

def multinomial_naive_bayes(train_file, bow_file, vocabs, test_file):
  manual_naive_bayes(train_file, bow_file, vocabs, test_file, bernoulli=False)

def manual_naive_bayes(train_file, bow_file, vocabs, test_file, bernoulli):
  # Calculate prior
  train_labels = []
  with open(train_file, "r") as f:
    for line in f.readlines():
      train_labels += [int(line.split()[-1])]
  num_train_examples = len(train_labels)
  num_train_positive = sum([l for l in train_labels if l == 1])
  num_train_negative = num_train_examples - num_train_positive
  positive_prior = float(num_train_positive) / num_train_examples
  negative_prior = float(num_train_negative) / num_train_examples
  print "Num train examples = %s, positive prior = %s, negative prior = %s" %\
    (num_train_examples, positive_prior, negative_prior)

  # Build up document frequency
  #print "Building up document frequency by word..."
  #total_documents_by_word = {}
  #with open(bow_file, "r") as f:
  #  for i, line in enumerate(f.readlines()):
  #    label = train_labels[i]
  #    vector = [int(v) for v in line.split(",")]
  #    assert len(vector) == len(vocabs),\
  #      "expected %d words in line %d, got %d".format(len(vocabs), i + 1, len(vector))
  #    for j, value in enumerate(vector):
  #      if value > 0:
  #        word = vocabs[j]
  #        total_documents_by_word[word] = (total_documents_by_word.get(word) or 0) + 1

  # Build up word counts by label
  num_features = len(vocabs)
  pos_counts_by_word = {}
  neg_counts_by_word = {}
  with open(bow_file, "r") as f:
    for i, line in enumerate(f.readlines()):
      vector = [int(v) for v in line.split(",")]
      label = train_labels[i]
      counts_by_word = pos_counts_by_word if label == 1 else neg_counts_by_word
      for j, value in enumerate(vector):
        word = vocabs[j]
        if value > 0 and word in vocabs:
          if word not in counts_by_word:
            counts_by_word[word] = 0
          if bernoulli:
            counts_by_word[word] += 1
          else:
            #tf = value
            #idf = float(num_train_examples) / total_documents_by_word[word]
            #counts_by_word[word] += tf * idf
            counts_by_word[word] += value
  print "Using %s features: %s..." % (num_features, ", ".join(list(vocabs)[:10]))
  print "%s features are used in positive reviews: %s..." %\
    (len(pos_counts_by_word), ", ".join(pos_counts_by_word.keys()[:10]))
  print "%s features are used in negative reviews: %s..." %\
    (len(neg_counts_by_word), ", ".join(neg_counts_by_word.keys()[:10]))

  # Helper method to compute likelihood of a word given a label
  def compute_likelihood(word, label):
    counts_by_word = pos_counts_by_word if label == 1 else neg_counts_by_word
    num_train_examples = num_train_positive if label == 1 else num_train_negative
    word_count = counts_by_word[word] if word in counts_by_word else 0
    return float(word_count + 1) / (num_train_examples + (2 if bernoulli else num_features))

  # Do some classifying!
  num_test_correct = 0
  num_test_examples = 0
  log_threshold = 10
  with open(test_file, "r") as f:
    for i, line in enumerate(f.readlines()):
      words = clean_words(parse_example(line).split())
      expected_label = int(line.split()[-1])
      pos_probability = positive_prior
      neg_probability = negative_prior
      for w in words:
        pos_probability *= compute_likelihood(w, 1)
        neg_probability *= compute_likelihood(w, 0)
      predicted_label = 1 if pos_probability >= neg_probability else 0
      if predicted_label == expected_label:
        num_test_correct += 1
      num_test_examples += 1
      if i < log_threshold:
        prob_string = "  positive probability = %s, negative probability = %s" % (pos_probability, neg_probability)
        print_classify_example(line, words, predicted_label, expected_label, prob_string)
  print_result(num_test_correct, num_test_examples)

def print_classify_example(line, words, predicted_label, expected_label, extra=""):
  print "------------------------------------------------------------------------"
  print "Classifying line:"
  print "  %s" % line.strip()
  print "  words = [%s]" % (", ".join(words))
  if extra:
    print extra
  print "  predicted label = %s, expected label = %s" % (predicted_label, expected_label)

def print_result(num_test_correct, num_test_examples):
  print "\n================================"
  print "You guessed %s/%s correct." % (num_test_correct, num_test_examples)
  print "================================\n"

if __name__ == "__main__":
  main()

