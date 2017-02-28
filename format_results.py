#!/usr/bin/env python

import os
from os.path import isfile, join
import sys

if len(sys.argv) != 2:
  print "Usage: format_results.py <path>"
  sys.exit(1)

path = sys.argv[1]
files = [join(path, f) for f in os.listdir(path) if isfile(join(path, f)) and f.endswith(".log")]

model_to_scores = {}

for f in files:
  with open(f, "r") as reader:
    model = None
    for line in reader.readlines():
      if line.startswith("Running model"):
        model = line.split()[2].replace("'", "")
        if model not in model_to_scores:
          model_to_scores[model] = []
      elif line.startswith("You guessed"):
        accuracy = float(line.split()[4].replace("%", "")) / 100
        model_to_scores[model] += [('accuracy', accuracy)]
      elif "Precision" in line:
        precision = float(line.strip().split()[2])
        model_to_scores[model] += [('precision', precision)]
      elif "Recall" in line:
        recall = float(line.strip().split()[2])
        model_to_scores[model] += [('recall', recall)]
      elif "F1" in line:
        f1 = float(line.strip().split()[2])
        model_to_scores[model] += [('f1', f1)]
      elif "AUC" in line:
        auc = float(line.strip().split()[2])
        model_to_scores[model] += [('auc', auc)]

for model, scores in model_to_scores.items():
  new_scores = {}
  for (s, v) in scores:
    new_scores[s] = (new_scores.get(s) or []) + [v]
  for s in new_scores.keys():
    new_scores[s] = float(sum(new_scores[s])) / len(new_scores[s])
  #print "%s: %s" % (model, new_scores)
  print "%s: %.2f & %.2f & %.2f & %.2f & %.2f" %\
    (model, new_scores['accuracy'], new_scores['precision'], new_scores['recall'], new_scores['f1'], new_scores['auc'])

