#!/bin/bash

k=$1
prefix=$2
shift
shift
model=$@

for i in $(seq 1 $k); do
  train_file="$prefix""$i"".train"
  test_file="$prefix""$i"".test"
  log_file="$prefix""$i"".log"
  echo "Generating bag of words on fold $i."
  echo "Train file is $train_file, test file is $test_file. Writing to $log_file..."
  python preprocessSentences.py -p . -t "$train_file" > "$log_file" 2>&1
  for m in $model; do
    echo "Running model '$m'..."
    python classifier.py -t "$train_file" -b out_bag_of_words_5.csv -v out_vocab_5.txt -e "$test_file" -m "$m" >> "$log_file" 2>&1
  done
done

