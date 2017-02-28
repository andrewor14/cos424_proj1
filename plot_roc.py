#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) != 2:
  print "Usage: plot_roc.py <log_file>"
  sys.exit(1)

log_file = sys.argv[1]
roc_data = [] # (model, fpr, tpr, auc)

def parse_list(line):
  return [float(x) for x in line[line.find("["):line.find("]")].strip("[]").split(", ")]

with open(log_file, "r") as f:
  model = fpr = tpr = auc = None
  for line in f.readlines():
    if line.startswith("Running model"):
      if model is not None:
        roc_data += [(model, fpr, tpr, auc)]
      model = line.split()[2].replace("'", "") 
      fpr = tpr = auc = None
    elif "False positive rate" in line:
      fpr = np.array(parse_list(line))
    elif "True positive rate" in line:
      tpr = np.array(parse_list(line))
    elif "AUC" in line:
      auc = float(line.strip().split()[2])
  roc_data += [(model, fpr, tpr, auc)]

# Data is a list of (name, fpr, tpr, auc) tuples
plt.title('Receiver operating characteristic')
colors = ['g', 'k', 'r', 'm']
for (i, (model, fpr, tpr, auc)) in enumerate(roc_data):
  plt.plot(fpr, tpr, color=colors[i], linewidth=2, label='%s; AUC = %0.2f'% (model, auc))
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.show()

