#!/usr/bin/env python

import sys

if len(sys.argv) != 2:
  print "Usage: format_twitter.py <inputfile>"
  sys.exit(2)

inputfile = sys.argv[1]
trainfile = inputfile + ".train"
testfile = inputfile + ".test"

with open(trainfile, "w") as train_writer:
  with open(testfile, "w") as test_writer:
    with open(inputfile, "r") as reader:
      for i, line in enumerate(reader.readlines()):
        if i == 0:
          continue
        def write_with(writer):
          split = line.split(",")
          number = split[0]
          label = split[1]
          text = split[3].strip()
          new_line = "%s %s        %s\n" % (number, text, label)
          writer.write(new_line)
        if i % 50 == 0:
          write_with(train_writer)
        if i % 250 == 1:
          write_with(test_writer)

