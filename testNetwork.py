import itertools
import numpy as np
import re
import Stemmer
import itertools
import sys
import os
import time
import csv
import nltk
from datetime import datetime
from utils import *
from RNN import RNNNumpy

vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"
stemmer = Stemmer.Stemmer('english')

def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        print "epoch number %d" % epoch
        for i in range(len(y_train)):
            print "begin calculate loss"
            loss = model.calculate_loss(X_train, y_train, learning_rate)
            print "finish los calculation"
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            save_model_parameters_theano("./data/rnn-theano-%s.npz" % (time), model)
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5 
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()  
    

def generate_sentence(model):
    # We start the sentence with the start token
    new_sentence = [word_to_index[sentence_start_token]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_to_index[sentence_end_token]:
        next_word_probs = model.forward_propagation(new_sentence)
        #print next_word_probs
        #print "---------------------------"
        #print next_word_probs[-1][0]
        #sampled_word = word_to_index[unknown_token]
        # We don't want to sample unknown words
        #while sampled_word == word_to_index[unknown_token]:
          #  samples = np.random.multinomial(1, next_word_probs[-1])
         #   sampled_word = np.argmax(samples)
        #new_sentence.append(sampled_word)
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    print sentence_str
    return sentence_str


print "Reading CSV file..."
with open('./data/reddit-comments-2015-08.csv', 'rb') as f:
    reader = csv.reader(f, skipinitialspace=True)
    reader.next()
    # Split full comments into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    # Append SENTENCE_START and SENTENCE_END
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
print "Parsed %d sentences." % (len(sentences))

tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
 
# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print "Found %d unique words tokens." % len(word_freq.items())

# Count the word frequencies
vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

#print "test"
model = RNNNumpy(vocabulary_size)
#load_model_parameters_theano('./data/trained-model-theano.npz', model)
#sentense = 'hi how are you'
#sentensCode = [word_to_index[word] for word in sentense.split(' ')]
#next_word_probs = model.forward_propagation(sentensCode)
#print next_word_probs
#print "length is %d" % len(next_word_probs)

#variants = [np.array(arr).argmax() for arr in next_word_probs[0]] 
#print variants
#genSentense = []
#for var in variants:
#    genSentense.append(index_to_word[var])

#print genSentense
    
 
print "Using vocabulary size %d." % vocabulary_size
print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])


 
# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
 
print "\nExample sentence: '%s'" % sentences[0]
print "\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0]

# Create the training data
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

print 'Example X_train '
print X_train[1:10]
print 'Example Y_train '
print Y_train[1:10]


#t1 = time.time()
#model.numpy_sdg_step(X_train[10], y_train[10],0.005) 
#t2 = time.time()
#print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)


np.random.seed(10)
# Train on a small subset of the data to see what happens
losses = train_with_sgd(model, X_train[:50000], y_train[:50000], nepoch=1, evaluate_loss_after=1)
losses = train_with_sgd(model, X_train[10000:60000], y_train[10000:60000], nepoch=1, evaluate_loss_after=1)
losses = train_with_sgd(model, X_train[20000:70000], y_train[20000:70000], nepoch=1, evaluate_loss_after=1)

#load_model_parameters_theano('./data/trained-model-theano.npz', model)


#num_sentences = 2
#senten_min_length = 3
 
#for i in range(num_sentences):
#   sent = []
    # We want long sentences, not sentences with one or two words
#    while len(sent) < senten_min_length:
#        sent = generate_sentence(model)
#        print sent
#    print " ".join(sent)










