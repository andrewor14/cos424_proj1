#!/bin/bash

k=$1
prefix=$2
model=$3

for i in $(seq 1 $k); do
  train_file="$prefix""$i"".train"
  test_file="$prefix""$i"".test"
  log_file="$prefix""$i"".$model"".log"
  echo "Running on fold $i. Train file is $train_file, test file is $test_file. Writing to $log_file..."
  python preprocessSentences.py -p . -t "$train_file" > "$log_file" 2>&1
  python classifier.py -t "$train_file" -b out_bag_of_words_5.csv -v out_vocab_5.txt -e "$test_file" -m "$model" >> "$log_file" 2>&1
done

