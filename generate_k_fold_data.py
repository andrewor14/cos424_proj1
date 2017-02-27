#!/bin/usr/env python

import sys
import math

if len(sys.argv) != 3:
  print "Usage: generate_k_fold_data.py <k> <data_file>"
  sys.exit(1)

k = int(sys.argv[1])
data_file = sys.argv[2]
num_examples = 0

with open(data_file, "r") as f:
  num_examples = len(f.readlines())

num_examples_per_partition = math.ceil(float(num_examples) / k)

for i in range(1, k+1):
  train_file = "%s%s.train" % (data_file, i)
  test_file = "%s%s.test" % (data_file, i)
  with open(data_file, "r") as reader,\
      open(train_file, "w") as train_writer,\
      open(test_file, "w") as test_writer:
    line_count = 0
    partition_number = 0
    for line in reader.readlines():
      if line_count % num_examples_per_partition == 0:
        partition_number += 1
      writer = test_writer if partition_number == i else train_writer
      writer.write(line)
      line_count += 1

