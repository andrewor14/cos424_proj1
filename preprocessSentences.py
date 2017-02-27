import nltk, re, pprint
from nltk import word_tokenize
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, isdir, join
import numpy
import re
import sys
import getopt
import codecs
import time
import os
import csv

chars = ['{','}','#','%','&','\(','\)','\[','\]','<','>',',', '!', '.', ';', 
'?', '*', '\\', '\/', '~', '_','|','=','+','^',':','\"','\'','@','-']

#def stem(word):
#   regexp = r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$'
#   stem, suffix = re.findall(regexp, word)[0]
#   return stem

def unique(a):
   """ return the list with duplicate elements removed """
   return list(set(a))

def intersect(a, b):
   """ return the intersection of two lists """
   return list(set(a) & set(b))

def union(a, b):
   """ return the union of two lists """
   return list(set(a) | set(b))

def get_files(mypath):
   return [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]

def get_dirs(mypath):
   return [ f for f in listdir(mypath) if isdir(join(mypath,f)) ]

# Reading a bag of words file back into python. The number and order
# of sentences should be the same as in the *samples_class* file.
def read_bagofwords_dat(myfile):
  bagofwords = numpy.genfromtxt('myfile.csv',delimiter=',')
  return bagofwords

def parse_example(line):
  raw = line.decode('latin1')
  raw = ' '.join(raw.rsplit()[1:-1])
  # remove noisy characters; tokenize
  raw = re.sub('[%s]' % ''.join(chars), ' ', raw)
  return raw

porter = nltk.PorterStemmer() # also lancaster stemmer
wnl = nltk.WordNetLemmatizer()
stop_words = stopwords.words("english")

def clean_word(word):
  word = word.lower().strip()
  word = wnl.lemmatize(word)
  try:
    word = porter.stem(word)
  except Exception as e:
    print "STEMMING WORD FAILED ON '%s'" % word
  return word

def clean_words(tokens):
  return [clean_word(w) for w in tokens if w.lower() not in stop_words and w.isalpha()]

def add_bigrams(tokens):
  bigrams = ["%s %s" % (tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)]
  return tokens + bigrams

def tokenize_corpus(path, train=True):
  classes = []
  samples = []
  docs = []
  if train == True:
    words = {}
  f = open(path, 'r')
  lines = f.readlines()

  for line in lines:
    classes.append(line.rsplit()[-1])
    samples.append(line.rsplit()[0])
    raw = parse_example(line)
    tokens = clean_words(word_tokenize(raw))
    tokens = add_bigrams(tokens)
    if train == True:
     for t in tokens: 
         try:
             words[t] = words[t]+1
         except:
             words[t] = 1
    docs.append(tokens)

  if train == True:
     return(docs, classes, samples, words)
  else:
     return(docs, classes, samples)

def wordcount_filter(words, num=5):
   keepset = []
   for k in words.keys():
       if(words[k] > num):
           keepset.append(k)
   print "Vocab length:", len(keepset)
   return(sorted(set(keepset)))


def find_wordcounts(docs, vocab):
   bagofwords = numpy.zeros(shape=(len(docs),len(vocab)), dtype=numpy.uint8)
   vocabIndex={}
   for i in range(len(vocab)):
      vocabIndex[vocab[i]]=i

   for i in range(len(docs)):
       doc = docs[i]

       for t in doc:
          index_t=vocabIndex.get(t)
          if index_t>=0:
             bagofwords[i,index_t]=bagofwords[i,index_t]+1

   print "Finished find_wordcounts for:", len(docs), "docs"
   return(bagofwords)


def main(argv):
  
  start_time = time.time()

  path = ''
  outputf = 'out'
  vocabf = ''
  trainfile = 'train.txt'

  try:
    opts, args = getopt.getopt(argv,"p:t:o:v:",["path=","ofile=","vocabfile="])
  except getopt.GetoptError:
    print 'Usage: \n python preprocessSentences.py -p <path> -t <trainfile> -o <outputfile> -v <vocabulary>'
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
      print 'Usage: \n python preprocessSentences.py -p <path> -t <trainfile> -o <outputfile> -v <vocabulary>'
      sys.exit()
    elif opt in ("-t", "--train"):
      trainfile = arg
    elif opt in ("-p", "--path"):
      path = arg
    elif opt in ("-o", "--ofile"):
      outputf = arg
    elif opt in ("-v", "--vocabfile"):
      vocabf = arg

  traintxt = path+"/"+trainfile
  print 'Path:', path
  print 'Training data:', traintxt

  # Build up things for feature selection
  count_by_word = {}
  total_documents_by_word = {}
  pos_documents_by_word = {}
  neg_documents_by_word = {}
  all_document_words = [] # each element is a list of words in a particular document
  word_count_threshold = 5
  with open(traintxt, "r") as train_data:
    for i, line in enumerate(train_data.readlines()):
      split = line.strip().split()
      tokens = clean_words(split[1:-1])
      tokens = add_bigrams(tokens)
      all_document_words += [tokens]
      label = int(float(split[-1]))
      for token in tokens:
        count_by_word[token] = (count_by_word.get(token) or 0) + 1
      for token in list(set(tokens)):
        docs_by_word = pos_documents_by_word if label == 1 else neg_documents_by_word
        docs_by_word[token] = (docs_by_word.get(token) or 0) + 1
        total_documents_by_word[token] = (total_documents_by_word.get(token) or 0) + 1
  total_documents = len(all_document_words)

  print "Done building things up man. Num documents in train set: %s." % total_documents
  print "Number of features before any feature selection: %s" % len(count_by_word)

  def computeCPD(word):
    pos_freq = pos_documents_by_word[word] if word in pos_documents_by_word else 0
    neg_freq = neg_documents_by_word[word] if word in neg_documents_by_word else 0
    return float(abs(pos_freq - neg_freq)) / (pos_freq + neg_freq)

  # Filter out some features, first by total count and then by CPD score
  for word, count in count_by_word.items():
    if count < word_count_threshold:
      del count_by_word[word]
  print "Number of features after filtering out words by count threshold: %s" % len(count_by_word)
  vocabs = sorted(count_by_word.keys(), key=computeCPD, reverse=True)
  max_num_features = min(int(len(vocabs) * 0.9), 3000)
  vocabs = vocabs[:max_num_features]

  print "Done doing the feature selection thing man. Num vocabs: %s." % len(vocabs)

  # Write vocab file
  vocab_file_name = outputf+"_vocab_"+str(word_count_threshold)+".txt"
  with codecs.open(path+"/"+vocab_file_name, "w","utf-8-sig") as f:
    f.write("\n".join(vocabs))

  # Write bag of words file
  bow_file_name = path+"/"+outputf+"_bag_of_words_"+str(word_count_threshold)+".csv"
  bow_data = numpy.zeros(shape=(total_documents, len(vocabs)), dtype=numpy.uint8)
  vocab_index = {}
  for i, vocab in enumerate(vocabs):
    vocab_index[vocab] = i
  for i, tokens in enumerate(all_document_words):
    for token in tokens:
      index = vocab_index.get(token)
      if index > 0:
        bow_data[i, index] = bow_data[i, index] + 1
  with open(bow_file_name, "wb") as f:
    writer = csv.writer(f)
    writer.writerows(bow_data)

  ## Tokenize training data (if training vocab doesn't already exist):
  #if (not vocabf):
  #  word_count_threshold = 5
  #  (docs, classes, samples, words) = tokenize_corpus(traintxt, train=True)
  #  vocab = wordcount_filter(words, num=word_count_threshold)
  #  # Write new vocab file
  #  vocabf = outputf+"_vocab_"+str(word_count_threshold)+".txt"
  #  outfile = codecs.open(path+"/"+vocabf, 'w',"utf-8-sig")
  #  outfile.write("\n".join(vocab))
  #  outfile.close()
  #else:
  #  word_count_threshold = 0
  #  (docs, classes, samples) = tokenize_corpus(traintxt, train=False)
  #  vocabfile = open(path+"/"+vocabf, 'r')
  #  vocab = [line.rstrip('\n') for line in vocabfile]
  #  vocabfile.close()

  #print 'Vocabulary file:', path+"/"+vocabf

  ## Get bag of words:
  #bow = find_wordcounts(docs, vocab)
  ## Check: sum over docs to check if any zero word counts
  #print "Doc with smallest number of words in vocab has:", min(numpy.sum(bow, axis=1))

  ## Write bow file
  #with open(path+"/"+outputf+"_bag_of_words_"+str(word_count_threshold)+".csv", "wb") as f:
  #  writer = csv.writer(f)
  #  writer.writerows(bow)

  ## Write classes
  #outfile= open(path+"/"+outputf+"_classes_"+str(word_count_threshold)+".txt", 'w')
  #outfile.write("\n".join(classes))
  #outfile.close()

  # Write samples
  #outfile= open(path+"/"+outputf+"_samples_class_"+str(word_count_threshold)+".txt", 'w')
  #outfile.write("\n".join(samples))
  #outfile.close()

  print 'Output files:', path+"/"+outputf+"*"

  # Runtime
  print 'Runtime:', str(time.time() - start_time)

if __name__ == "__main__":
  main(sys.argv[1:])

 
