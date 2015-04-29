from theano import tensor as T, shared, function, scan #, sparse as S #, compile as theano_compile
tdot = T.tensordot
from theano.tensor.nnet import softmax, sigmoid
from numpy import float64, array, zeros_like
from math import ceil

bankDir = '../data/stanfordSentimentTreebank/'

# Load dictionary from strings to IDs; and from IDs to sentiments,
# Set up IDs for lexical items and find vocabulary size

phraseid = {}
sentiment = {}
embeddingid = {}
vocab = 0

with open(bankDir+'dictionary.txt','r') as f:
    for line in f:
        key, value = line.rstrip().split('|')
        pid = int(value)
        phraseid[key] = pid
        if not ' ' in key:
            embeddingid[pid] = vocab
            vocab += 1

with open(bankDir+'sentiment_labels.txt','r') as f:
    f.readline()
    for line in f:
        key, value = line.rstrip().split('|')
        sentiment[int(key)] = float64(value)

def getid(phrase):
    return phraseid[phrase]

def getsent(pid):
    return sentiment[pid]

def getembid(pid):
    return embeddingid[pid]


# Function to update weights, using AdaGrad and momentum, and L1 regularisation

def update_function(parameters, learningRate, adaDecayCoeff, momDecayCoeff, regularisation):
    """
    parameters - sequence of Theano shared variables
    learningRate - float
    adaDecayCoeff - float
    momDecayCoeff - float
    regularisation - sequence of floats
    Returns a function that will take the gradients,
    and update the parameters accordingly
    """
    N = len(parameters)
    assert len(regularisation) == N
    
    gradient = [T.TensorVariable(p.type, name=p.name+'Grad') for p in parameters] #[:3]]
    #gradient.append(S.csr_matrix(parameters[3].name+'Grad', 'float64'))
    zero = [T.zeros_like(p) for p in parameters]
    squareSum = [shared(zeros_like(p.get_value()), name=p.name+'SqSum') for p in parameters]
    stepSize  = [shared(zeros_like(p.get_value()), name=p.name+'Step')  for p in parameters]
    
    rate = shared(learningRate, name='rate')
    adaDecay = shared(adaDecayCoeff, name='adaDecay')
    momDecay = shared(momDecayCoeff, name='momDecay')
    reg = shared(array(regularisation), name='reg')
    
    update_sum = function(gradient, updates=
                          list((squareSum[i],
                                adaDecay*squareSum[i] + gradient[i]**2)
                               for i in range(N)), #-1))
                          #  + [(squareSum[3],
                          #      adaDecay*squareSum[3] + S.sqr(gradient[3]))],
                          allow_input_downcast=True)
    
    update_step= function(gradient, updates=
                          list((stepSize[i],
                                momDecay*stepSize[i] + T.switch(T.eq(squareSum[i],0),
                                                                zero[i],
                                                                rate/T.sqrt(squareSum[i])*gradient[i]))
                               for i in range(N)), #-1))
                          #  + [(stepSize[3],
                          #      momDecay*stepSize[3] + S.mul(gradient[3], rate/T.sqrt(squareSum[3])))],
                          allow_input_downcast=True)
    
    update_wei = function([], updates=
                          list((parameters[i],
                                parameters[i] - stepSize[i])
                               for i in range(N)),
                          allow_input_downcast=True)
    
    regularise = function([], updates=
                          list((parameters[i],
                                T.switch(T.lt(abs(parameters[i]),reg[i]),
                                         zero[i],
                                         parameters[i] - reg[i]*T.sgn(parameters[i])))
                               for i in range(N)),
                          allow_input_downcast=True)
    
    def update(*grads):
        update_sum(*grads)
        update_step(*grads)
        update_wei()
        regularise()
    
    return update #, squareSum, stepSize


# Function to calculate the gradient

def gradient_function(wQuad, wLin, wSent, granular=True):
    """
    parameters:
    wQuad - 3rd order tensor for combining vectors
    wLin  - 2nd order tensor for combining vectors
    wSent - vector(s) for predicting sentiment
    """
    
    def merge(left, right, cur, prev):
        """
        Given: a set of embeddings (prev),
         the index of next node (cur),
         and the indices of the child nodes (left, right),
        calculate the embedding for the next node,
        and output the set of embeddings with this one updated (new),
        along with the index for the next node (cur+1).
        """
        first = prev[left]
        second= prev[right]
        cat = T.concatenate((first,second))
        out = tdot(tdot(wQuad,first,((2),(0))),second,((1),(0))) + tdot(wLin,cat,((1),(0))) #+ bias
        rect = out * (out >= 0)
        new = T.set_subtensor(prev[cur], rect, inplace=False)
        return [cur+1, new]
    
    if granular:
        def cost(vector, gold):
            """
            Given an output vector, and the gold standard class,
            calculate (one minus) the softmax probability of predicting the gold 
            """
            prob = softmax(tdot(vector,wSent,(0,1)))  # softmax returns a matrix
            return 1-prob[0,gold]
    else:
        def cost(vector, gold):
            """
            Given an output vector, and the gold standard annotation,
            calculate the square loss of predicting the gold from the vector
            """
            pred = sigmoid(tdot(vector,wSent,(0,0)) ) #+ bSent)
            return (pred-gold)**2
    
    sentembed = T.dmatrix('sentembed')
    leftindices = T.bvector('leftindices')
    rightindices= T.bvector('rightindices')
    if granular:
        senti = T.bvector('senti')
    else:
        senti = T.dvector('senti')
    n,m = T.shape(sentembed)
    padded = T.concatenate((sentembed,T.zeros((n-1,m),'float64')))
    
    output, _ = scan(merge,
                       sequences=[leftindices,
                                  rightindices],
                       outputs_info=[n,
                                     padded])
    vec = output[-1][-1]  # Take just the set of embeddings from the last calculation
    
    loss, _ = scan(cost,
                     sequences=[vec,
                                senti],
                     outputs_info=None)
    total = T.sum(loss)
    
    grads = T.grad(total, [wQuad,wLin,wSent,sentembed])
    find_grad = function([sentembed,leftindices,rightindices,senti], grads, allow_input_downcast=True)
    find_error= function([sentembed,leftindices,rightindices,senti], total, allow_input_downcast=True)
    
    return find_grad, find_error


# Load sentences

def get_data(granularity=5):
    """
    Returns three lists of datapoints in the form:
    [embedding ids, left children, right children, sentiment scores]
    The lists are: train, test, dev
    If granularity is nonzero, splits the sentiment score into as many bins
    """
    
    train = []
    test = []
    dev = []
    
    with open(bankDir+'SOStr.txt','r') as ftext, \
         open(bankDir+'STree.txt','r') as ftree, \
         open(bankDir+'datasetSplit.txt','r') as fsplit:
        fsplit.readline()
        for line in ftext:
            # Find: tokens, phrase ids, embedding ids, child nodes
            tokens = line.rstrip().split('|')
            N = len(tokens)
            ids = list(map(getid, tokens))
            embids = tuple(map(getembid, ids))
            tree = [int(x)-1 for x in next(ftree).rstrip().split('|')] # -1 for 0-indexing
            reverse = tree[::-1]
            leftchildren = []
            rightchildren = []
            for x in range(N,2*N-1):
                leftchildren.append(tree.index(x))
                rightchildren.append(2*N-2-reverse.index(x))
            # Construct constituent phrases to find their ids
            for i in range(N,2*N-1):
                nonterm = [i]
                terminals = []
                while nonterm:
                    chosen = nonterm.pop() - N
                    left = leftchildren[chosen]
                    right = rightchildren[chosen]
                    if left < N : terminals.append(left)
                    else: nonterm.append(left)
                    if right < N: terminals.append(right)
                    else: nonterm.append(right)
                text = ' '.join(tokens[min(terminals):max(terminals)+1])
                ids.append(getid(text))
            scores = list(map(getsent,ids))
            if granularity:
                scores = [max(ceil(x*granularity)-1,0) for x in scores]  # e.g. [0,0.2],(0.2,0.4],(0.4,0.6],etc.
            # Record data
            datapoint = [embids,
                         array(leftchildren, 'int8'),
                         array(rightchildren, 'int8'),
                         array(scores, 'int8')]
            section = fsplit.readline().strip().split(',')[-1]
            if section == '1':
                train.append(datapoint)
            elif section == '2':
                test.append(datapoint)
            else:
                dev.append(datapoint)
    
    return [train, test, dev]
