import numpy as np
import time
import random
from datetime import datetime

def softmax(x):
    sf = np.exp(x)
    sf = sf/np.sum(sf, axis=0)
    return sf

def findMostFeature(word, word_to_vector):
    return np.argmax(np.asarray(word_to_vector[word]))

def createY_patternProbability(y, dimension):
    result = np.zeros((len(y), dimension))
    
    for i in np.arange(len(y)):
        y_prob_pattern = np.zeros(dimension)
        y_prob_pattern[y[i]] = 1
        result[i] = y_prob_pattern
    return result


class RNNNumpy:
     
    def __init__(self, word_dim, word_dim_out, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.word_dim_out = word_dim_out
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim_out, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

       
    def forward_propagation(self, x, word_to_vector):
        # The total number of time steps
        T = len(x)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        # The outputs at each time step. Again, we save them for later.
        o = np.zeros((T, self.word_dim_out))
        # For each time step...
        for t in np.arange(T):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            s[t] = np.tanh(self.U.dot(word_to_vector[x[t]]) + self.W.dot(s[t-1]))
            #if (random.randint(0,1001) % 1000) == 0:
                #print 'V matrix is :'
                #print self.V[:5]
                #print 's vector is :'
                #print s
            o[t] = softmax(self.V.dot(s[t]))
        return [o, s]
        RNNNumpy.forward_propagation = forward_propagation

    def predict(self, x, word_to_vector):
        # Perform forward propagation and return index of the highest score
        o, s = self.forward_propagation(x,word_to_vector)
        #print o
        return o #np.argmax(o, axis=1)
        RNNNumpy.predict = predict

    def calculate_total_loss(self, x, y, learning_rate, word_to_vector, isTrain):
        L = 0
        # For each sentence...
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i], word_to_vector)
            #corectPrediction = np.amax(o, axis=1)
            y_pattern = createY_patternProbability(y[i], self.word_dim_out)
            
            L += -1 * np.sum(y_pattern[i].dot(np.log(o[i] + 0.0001)) for i in np.arange(len(o)))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if (i % 1000) == 0:
                print "%s current L is %f train number = %d " % (time, L, i)
            if isTrain:    
                self.numpy_sdg_step(x[i], y[i], learning_rate, o, s, word_to_vector)
            
        return L
        RNNNumpy.calculate_total_loss = calculate_total_loss

    
    def calculate_loss(self, x, y, word_to_vector, learning_rate, isTrain):
        # Divide the total loss by the number of training examples
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x,y, learning_rate, word_to_vector, isTrain)/N
        RNNNumpy.calculate_loss = calculate_loss

    def numpy_sdg_step(self, x, y, learning_rate, o, s, word_to_vector):
        # Calculate the gradients
        dLdU, dLdV, dLdW = self.bptt(x, y, o, s, learning_rate, word_to_vector)
        # Change parameters according to gradients and learning rate
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW
        
    def bptt(self, x, y, o, s, learning_rate, word_to_vector):
        T = len(y)
        # Perform forward propagation
        # o, s = self.forward_propagation(x)
        # We accumulate the gradients in these variables
        delta_array = np.zeros((T, 100))
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.
        # For each output backwards...
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            # Initial delta calculation
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                #print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                dLdW += np.outer(delta_t, s[bptt_step-1])
                #index = random.randint(0,199)
                index = findMostFeature(x[bptt_step], word_to_vector)
                dLdU[:, index] += delta_t
                #new_dLdU = self.findBestdLdWIndex(dLdU, dLdW, dLdV, delta_t, x,y, learning_rate, word_to_vector)
                #dLdU = new_dLdU
                
                #dLdU.dot(word_to_vector[x[btt_step]]) += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
                
        
        return [dLdU, dLdV, dLdW]
        RNNNumpy.bptt = bptt
        RNNNumpy.numpy_sdg_step = numpy_sdg_step


   

    
    
        

   
        
