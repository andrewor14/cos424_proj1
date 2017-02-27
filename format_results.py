#!/usr/bin/env python

import os
from os.path import isfile, join
import sys

if len(sys.argv) != 2:
  print "Usage: format_results.py <path>"
  sys.exit(1)

path = sys.argv[1]
files = [join(path, f) for f in os.listdir(path) if isfile(join(path, f)) and f.endswith(".log")]

model_to_accuracy = {}
model_to_auc = {}

for f in files:
  with open(f, "r") as reader:
    interesting_lines = [line for line in reader.readlines() if\
      line.startswith("Running model") or line.startswith("You guessed") or "AUC" in line]
    for i in range(len(interesting_lines)):
      if interesting_lines[i].startswith("Running model"):
        model = interesting_lines[i].split()[2].replace("'", "")
        accuracy = float(interesting_lines[i+1].split()[4].replace("%", ""))
        auc = float(interesting_lines[i+2].strip().split()[2])
        model_to_accuracy[model] = (model_to_accuracy.get(model) or []) + [accuracy]
        model_to_auc[model] = (model_to_auc.get(model) or []) + [auc]

#print model_to_accuracy
#print model_to_auc

for m in model_to_accuracy.keys():
  model_to_accuracy[m] = sum(model_to_accuracy[m]) / len(model_to_accuracy[m])
for m in model_to_auc.keys():
  model_to_auc[m] = sum(model_to_auc[m]) / len(model_to_auc[m])

print "Accuracy: %s" % model_to_accuracy
print "AUC: %s" % model_to_auc

