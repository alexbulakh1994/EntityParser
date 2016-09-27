import csv
import random
import numpy as np
import itertools
import nltk
import pickle
import gensim
import sys
import os
import time
import logging
from gensim.models import word2vec
from datetime import datetime
from utils import *
from RNN import RNNNumpy

#C:\Users\Alex\Downloads/pod_gtin.csv
#'C:\NLM/company.csv'
# marked words

keyWordList = ['PRODUCT_NAME', 'PRODUCT_NAMEs', 'COMPANY_NAME', 'COMPANY_NAMEs', 'COUNTRY', 'NATIONALITY', 'YEAR']
Y_train = []
sentenseList = []
globalComments = []

RULES = ['COMPANY_NAME is an NATIONALITY brand of PRODUCT_NAME produced from YEAR to YEAR in Springfields, Massachusetts COUNTRY',
        'COMPANY_NAME produced PRODUCT_NAME, but the name was changed to the COMPANY_NAME in YEAR',
        'UNKNOWN_TOCKEN COMPANY_NAME is a COUNTRY PRODUCT_NAME  manufacturer with headquarters UNKNOWN_TOCKEN in Solna',
        'COMPANY_NAME products are PRODUCT_NAMEs for consumer packaging and graphical applications.',
        'It is the world\'s largest producer of  PRODUCT_NAME UNKNOWN_TOCKEN',
        'It develops and markets PRODUCT_NAMEs for customers in metallurgical industries of using',
        'The company\'s portfolio of products include PRODUCT_NAMEs surface coating and welding',
        'Early production consisted of PRODUCT_NAME UNKNOWN_TOCKEN and for nowadays it is very nice',
        'The production is based on hard technique and includes PRODUCT_NAME',
        'NATIONALITY company, formed in YEAR and specialised in processing PRODUCT_NAME',
        'In out COUNTRY this company producing preserved PRODUCT_NAME in all over the world',
        'COMPANY_NAME is a PRODUCT_NAME brand including PRODUCT_NAME and have high quality',
        'Our company is the largest producer of  PRODUCT_NAME in COUNTRY a lot of time',
        'Our company is manufacturer of PRODUCT_NAME in COUNTRY for nowadays',
        'The main products are in the category of PRODUCT_NAME',
        'COMPANY_NAME  is best known for their PRODUCT_NAME, sold in COUNTRY',
        'Make original focus on PRODUCT_NAME for exracting investor for those industry',
        'COMPANY_NAME  makes PRODUCT_NAME for COMPANY_NAMEs in Europe, USA and other conuntry of the world',
        'UNKNOWN_TOCKEN products include PRODUCT_NAME which were invented in the early YEAR',
        'COMPANY_NAME  world\'s largest suppliers of PRODUCT_NAME',
        'Company started making PRODUCT_NAMEs in YEAR UNKNOWN_TOCKEN',
        'COMPANY_NAME started making PRODUCT_NAME a lot of time ago',
        'UNKNOWN_TOCKEN were company to mass-produce  PRODUCT_NAME  as well as the first to build a PRODUCT_NAMEs',
        'COMPANY_NAME were company to mass-produce  PRODUCT_NAMEs and famous it in all of the world',
        'Company is best known for the PRODUCT_NAME and their quality', 
        'They mostly made portable PRODUCT_NAME but later they create no portable products',
        'COMPANY_NAME mostly made portable PRODUCT_NAME UNKNOWN_TOCKEN',
        'The company produces a range of PRODUCT_NAME all over the world and have a large budget',
        'Originally a manufacturer of PRODUCT_NAME make this company famous',
        'Silva Sweden AB is an world famous company, most known for their PRODUCT_NAMEs',
        'Company based in our city that makes PRODUCT_NAME',
        'The company also makes PRODUCT_NAME UNKNOWN_TOCKEN',
        'The company develops, produces and markets PRODUCT_NAME for UNKNOWN_TOCKEN',
        'Its core activity is the production, distribution and sale of PRODUCT_NAME' 
]

unknown_token = 'UNKNOWN_TOCKEN'

def companyLoad(file_destination):
    with open(file_destination,'r') as dest_f:
        data_iter = csv.reader(dest_f,delimiter = ',', 
                           quotechar = ',')
        data = []
        for row in data_iter:
            data.append(row) 
    result =  [x for x in data if x != []]
    strResult = [' '.join(x) for x in result]
    return np.asarray(strResult)    

def productLoad(file_destination, column):
    with open(file_destination,'r') as dest_f:
        data_iter = csv.reader(dest_f,delimiter = ';', 
                           quotechar = '"')
        data = []
        for row in data_iter:
            data.append(row[column]) 
    result =  [x for x in data if x != '' and len(x) < 30]
    return np.asarray(result) 


def txtFileLoad(file_locality):
    col1 = np.genfromtxt(file_locality, usecols=(0),delimiter=',',dtype=None)
    return col1    
    

def parse():
    index = 0
    global sentenseList
    for item in productList:
        rule = random.choice(RULES)
        sentenseList.append(parseRule(rule.split(' '),item))
        if index % 10000 == 0:
            print 'current index is: ' + str(index)
            print sentenseList[index]
        index = index + 1
         
    print len(sentenseList)
    return sentenseList
        
def keyWords(x, item):
    if x == 'PRODUCT_NAME':
        return matchOneProductTemplate(item)
    elif x == 'PRODUCT_NAMEs':    
        return matchManyProductTemplates(item)
    elif x == 'COMPANY_NAME':    
        return matchCompany(1)
    elif x == 'COMPANY_NAMEs':
        return matchCompany(-1)
    elif x == 'COUNTRY':    
        return matchCountry()
    elif x == 'NATIONALITY':    
        return matchNation()
    elif x == 'YEAR':    
        return getYear()

def matchOneProductTemplate(item):
    res = np.array([item])
    encodeProdCompFunc(item, 1, 2, 3)
    return res

def encodeProdCompFunc(item, codeW1, codeW2, codeW3):
    global Y_train
    arrLen = len(item.split(' '))
    for i in range(arrLen):
        if i == 0:
            Y_train[len(Y_train) - 1].append(codeW1)
        elif i == arrLen - 1:
            Y_train[len(Y_train) - 1].append(codeW3)
        else:
            Y_train[len(Y_train) - 1].append(codeW2)


def matchManyProductTemplates(item):
    countGoods = random.randint(0, 5) 
    goods = np.random.choice(productList, countGoods)
    for g in goods:
        encodeProdCompFunc(g, 1, 2, 3)
    np.append(goods, item)
    return goods

def matchCompany(value):
    if value > 0:
        company = np.random.choice(companyList, 1)
    else:
        numbers = random.randint(2, 3) 
        company = np.random.choice(companyList, numbers)
    for item in company:
        encodeProdCompFunc(item, 4, 5, 6)    
    return company

def matchCountry():
    country = np.random.choice(countryList, 1)
    for item in country:
        encodeProdCompFunc(item, 7, 7, 7)
    return country
    
def matchNation():
    nation = np.random.choice(nationList, 1)
    for item in nation:
        encodeProdCompFunc(item, 8, 8, 8)
    return nation


def parseRule(RULE_ARRAY, item):
    sentense = []
    global Y_train
    Y_train.append([])
    for word in RULE_ARRAY:
        if word in keyWordList:
            res = keyWords(word, item)
            sentense.append(' '.join(res))
        else:
            sentense.append(word)
            Y_train[len(Y_train) - 1].append(0)
    return ' '.join(sentense)    

def getYear():
    year = np.array([str(random.randint(1990, 2015))])
    for item in year:
        encodeProdCompFunc(item, 9, 9, 9)
    return year
            

def deleteStopWords(vocab):
    clearRuleSet = []
    for rule in vocab:
        newRule = []
        for word in rule.split(' '):
            if word not in stop_words:
                newRule.append(word)
        clearRuleSet.append(' '.join(newRule))
    return clearRuleSet

def loadMoreWords():
    print "Reading CSV file..."
    with open('C:\NLM\data/reddit-comments-2015-08.csv', 'rb') as f:
        reader = csv.reader(f, skipinitialspace=True)
        reader.next()
        # Split full comments into sentences
        sentences1 = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
        print 'parse second entity file'
        #sentences2 = itertools.chain(unicode(x, errors='ignore') for x in sentenseList[1:50000])
 
        sentenceTest1 = ["%s" % (x) for x in sentences1]
        #sentenceTest2 = ["%s" % (x) for x in sentences2]
        sentences = sentenceTest1 + [unicode(x, errors='ignore') for x in sentenseList]
    print "Parsed %d sentences." % (len(sentences))
    #globalComments = deleteStopWords(sentences)
    return sentences

def createDictionary(sentences, vocabulary_size):
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
 
    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print "Found %d unique words tokens." % len(word_freq.items())

    # Count the word frequencies
    vocab = word_freq.most_common(vocabulary_size-1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)]) 
    print "Using vocabulary size %d." % vocabulary_size
    print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])
    saveDictionary(word_to_index, 'word_to_index.pickle')
    saveDictionary(index_to_word, 'index_to_word.pickle')
    return [index_to_word, word_to_index]

def wordVectorized():
    model =  word2vec.Word2Vec.load_word2vec_format('text2.model.bin', binary=True)
    word_to_vector = dict([(i,model[w]) if w in model else (i, model[unknown_token]) for i,w in enumerate(index_to_word)])
    return word_to_vector


def saveTestData(listData):
    with open('C:\NLM\test/test.txt', 'wb') as f:
        pickle.dump(listData, f)
        
    
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

def createDataSetGensim(dataset):
    result = []
    random.shuffle(dataset)
    for x in dataset:
        result.append(x.split(' '))
    return result

def createX_train(tokenized_sentences):
    X_train = np.asarray([[word_to_index[w] if w in word_to_index else word_to_index[unknown_token] for w in sent.split(' ')] for sent in tokenized_sentences])
    return X_train     

def train_with_sgd(model, X_train, y_train, word_to_vector, learning_rate=0.005, nepoch=1, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        #print "before training we get: "
        #loss_test = model.calculate_loss(X_train, y_train, word_to_vector, learning_rate, False)
        #time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        #print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss_test)

        print "epoch number %d" % epoch
        print "begin calculate loss"
        loss = model.calculate_loss(X_train, y_train, word_to_vector, learning_rate, True)
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

def printResult(Sentense, probabilityMatrix):
    index = 0
    print 'sentense length is : %d probMatrixLen is : %d ' % (len(Sentense), len(probabilityMatrix))
    for vect in probabilityMatrix:
        maxProbability = np.max(vect)
        maxIndexProbability = np.argmax(vect)
        print "Current word is ( %s ) with probability %f and categoty number : %d" % (Sentense[index], maxProbability, maxIndexProbability)
        index = index + 1

def saveDictionary(wordDict, fileName):
    with open(fileName, 'wb') as handle:
        pickle.dump(wordDict, handle)

def loadDict(fileName):
    with open(fileName, 'rb') as handle:
        res = pickle.load(handle)
    return res 



testText = txtFileLoad('C:\NLM/testText.txt')

#----------------------------------------------------
nationList = txtFileLoad('C:\NLM/nation.txt')
countryList = txtFileLoad('C:\NLM/country.txt') 
companyList = companyLoad('C:\NLM/company.csv')    
productList = productLoad('C:\Users\Alex\Downloads/pod_gtin.csv', 1)

#------------------------------------------    
parse()


#------------------------------------------------
#mainList = loadMoreWords()
#------------------------------------------------

#index_to_word, word_to_index = createDictionary(mainList, 50000)
index_to_word = loadDict('index_to_word.pickle')
word_to_index = loadDict('word_to_index.pickle')

print 'create word to vector'
word_to_vector = wordVectorized()


#X_train = createX_train(sentenseList)
#deleted_items = []
#for i in np.arange(len(X_train)):
#    if len(X_train[i]) != len(Y_train[i]):
#        deleted_items.append(i)
        
#print 'length of deleted element is ' + str(len(deleted_items))

#X_train = np.delete(X_train, deleted_items)
#Y_train = np.delete(Y_train, deleted_items)


#------------------- for testing ----------------
#X_trainWords = [index_to_word[w] for w in X_train[135]]

X_test = createX_train(testText)
print X_test
#---------------------------------


print 'neural netword train start'
model = RNNNumpy(200, 10)
#------------------ for testing -------------------------
load_model_parameters_theano('./data/rnn-theano-2016-09-27-17-46-56.npz', model)


X_trainWords = [sent.split(' ') for sent in testText]
index = 0
for item in X_test:
    vectProbability = model.predict(item, word_to_vector)
    #X_trainWords = [index_to_word[w] for w in item]
    printResult(X_trainWords[index], vectProbability)
    index = index + 1
#-----------------------------------------------------

#np.random.seed(10)
# Train on a small subset of the data to see what happens
#losses = train_with_sgd(model, X_train[:100000], Y_train[:100000], word_to_vector, nepoch=3, evaluate_loss_after=1)

#saveTestData(mainList)

#print 'begin work with gensim'
#sentences = createDataSetGensim(mainList)
#sentences = MySentences(r'C:\NLM\test') # a memory-friendly iterator
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#model = gensim.models.Word2Vec(sentences)
#model = word2vec.Word2Vec(sentences, size=200)
#model.save_word2vec_format('text2.model.bin', binary=True)

    



