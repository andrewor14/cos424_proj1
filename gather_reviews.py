#!/usr/bin/env python

import os
import sys
from os.path import isfile, join

if len(sys.argv) != 2:
  print "Usage: gather_reviews.py <path>"
  sys.exit(1)

path = sys.argv[1]
reviews_for_training = "reviews.train"
reviews_for_testing = "reviews.test"

def parse_reviews(review_file):
  '''
  Return a list of reviews in the format "<ID> <review text> <rating>"
  '''
  print "  ... parsing reviews in %s" % review_file
  all_reviews = []
  with open(review_file, "r") as f:
    lines = f.readlines()
    i = 0
    unique_id = review_text = rating = None
    while i < len(lines):
      line = lines[i].strip()
      if line == "<unique_id>": unique_id = lines[i+1].strip()
      if line == "<review_text>": review_text = lines[i+1].strip()
      if line == "<rating>": rating = lines[i+1].strip()
      if line == "</review>":
        all_reviews += ["%s\t%s\t%s" % (unique_id, review_text, rating)]
        unique_id = review_text = rating = None
      i += 1
  return all_reviews

with open(reviews_for_training, "w") as train_writer, open(reviews_for_testing, "w") as test_writer:
  i = 0
  dirs = [f for f in os.listdir(path) if not isfile(join(path, f))]
  for d in dirs:
    print "Looking at dir %s" % d
    review_files = [f for f in os.listdir(join(path, d)) if f == "positive.review" or f == "negative.review"]
    assert len(review_files) == 2 # one positive, one negative
    for review_file in review_files:
      for review in parse_reviews(join(path, d, review_file)):
        writer = test_writer if i % 5 == 0 else train_writer
        writer.write("%s\n" % review)
        i += 1

