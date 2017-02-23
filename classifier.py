#!/usr/bin/env python

from preprocessSentences import clean_word, clean_words, parse_example

import argparse
import sys

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-t", "--train", help="training data file", required=True)
  parser.add_argument("-b", "--bow", help="bag of words file", required=True)
  parser.add_argument("-v", "--vocab", help="vocab file", required=True)
  parser.add_argument("-e", "--test", help="test data file", required=True)
  args = parser.parse_args()

  # Calculate prior
  train_labels = []
  with open(args.train, "r") as train_data:
    for line in train_data.readlines():
      train_labels += [int(line.split()[-1])]
  num_train_examples = len(train_labels)
  num_train_positive = sum([l for l in train_labels if l == 1])
  num_train_negative = num_train_examples - num_train_positive
  positive_prior = float(num_train_positive) / num_train_examples
  negative_prior = float(num_train_negative) / num_train_examples
  print "Num train examples = %s, positive prior = %s, negative prior = %s" %\
    (num_train_examples, positive_prior, negative_prior)

  # Build vocab dictionary
  vocabs = []
  with open(args.vocab, "r") as vocab_data:
    for line in vocab_data.readlines():
      vocabs += [clean_word(line.decode("utf-8-sig"))]

  # Build up word counts by label
  num_features = len(vocabs)
  pos_examples_by_word = {}
  neg_examples_by_word = {}
  with open(args.bow, "r") as bow_data:
    for i, line in enumerate(bow_data.readlines()):
      line_num = i + 1
      vector = [int(v) for v in line.split(",")]
      label = train_labels[i]
      assert len(vector) == num_features,\
        "expected %d features in line %d, got %d".format(num_features, line_num, len(vector))
      examples_by_word = pos_examples_by_word if label == 1 else neg_examples_by_word
      for j, value in enumerate(vector):
        if value > 0:
          if vocabs[j] not in examples_by_word:
            examples_by_word[vocabs[j]] = 0
          examples_by_word[vocabs[j]] += value
  print "Using %s features: %s..." % (num_features, ", ".join(vocabs[:15]))
  print "%s features are used in positive reviews: %s..." %\
    (len(pos_examples_by_word), ", ".join(pos_examples_by_word.keys()[:10]))
  print "%s features are used in negative reviews: %s..." %\
    (len(neg_examples_by_word), ", ".join(neg_examples_by_word.keys()[:10]))

  # Helper method to compute likelihood of a word given a label
  def compute_likelihood(word, label):
    examples_by_word = pos_examples_by_word if label == 1 else neg_examples_by_word
    num_train_examples = num_train_positive if label == 1 else num_train_negative
    word_count = examples_by_word[word] if word in examples_by_word else 0
    return float(word_count + 1) / (num_train_examples + num_features)

  # Do some classifying!
  num_test_correct = 0
  num_test_examples = 0
  log_threshold = 10
  with open(args.test, "r") as test_data:
    for i, line in enumerate(test_data.readlines()):
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
        print "------------------------------------------------------------------------"
        print "Classifying line:"
        print "  %s" % line.strip()
        print "  words = [%s]" % (", ".join(words))
        print "  label = %s" % label
        print "  positive probability = %s, negative probability = %s" % (pos_probability, neg_probability)
        print "  predicted label = %s, expected label = %s" % (predicted_label, expected_label)

  print "\n================================"
  print "You guessed %s/%s correct." % (num_test_correct, num_test_examples)
  print "================================\n"

if __name__ == "__main__":
  main()

