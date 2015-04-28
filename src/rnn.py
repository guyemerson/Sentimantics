from theano import tensor as T, shared, function, scan #, compile as theano_compile
tdot = T.tensordot
from numpy import longfloat, array, zeros_like

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
        phraseid[key] = int(value)

with open(bankDir+'sentiment_labels.txt','r') as f:
    f.readline()
    for line in f:
        key, value = line.rstrip().split('|')
        sentiment[int(key)] = longfloat(value)

for p,i in phraseid.items():
    if not ' ' in p:
        embeddingid[i] = vocab
        vocab += 1

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
    
    gradient = [T.TensorVariable(p.type, name=p.name+'Grad') for p in parameters]
    zero = [T.zeros_like(p) for p in parameters]
    squareSum = [shared(zeros_like(p.get_value()), name=p.name+'SqSum') for p in parameters]
    stepSize  = [shared(zeros_like(p.get_value()), name=p.name+'Step')  for p in parameters]
    
    rate = shared(learningRate, name='rate')
    adaDecay = shared(adaDecayCoeff, name='adaDecay')
    momDecay = shared(momDecayCoeff, name='momDecay')
    reg = shared(array(regularisation), name='reg')
    
    update_sum = function(gradient, updates=
                          tuple((squareSum[i],
                                 adaDecay*squareSum[i] + gradient[i]**2)
                                for i in range(N)),
                          allow_input_downcast=True)
    
    update_step= function(gradient, updates=
                          tuple((stepSize[i],
                                 momDecay*stepSize[i] + T.switch(T.eq(squareSum[i],0),
                                                                 rate/T.sqrt(squareSum[i])*gradient[i],
                                                                 zero[i]))
                                for i in range(N)),
                          allow_input_downcast=True)
    
    update_wei = function(gradient, updates=
                          tuple((parameters[i],
                                 parameters[i] - stepSize[i])
                                for i in range(N)),
                          allow_input_downcast=True)
    
    regularise = function([], updates=
                          tuple((parameters[i],
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
    
    return update


# Function to calculate the gradient

def gradient_function(wQuad, wLin, wSent):
    """
    parameters:
    wQuad - 3rd order tensor for combining vectors
    wLin  - 2nd order tensor for combining vectors
    wSent - vector for predicting sentiment
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
    
    def cost(vector, gold):
        """
        Given an output vector, and the gold standard annotation,
        calculate the square loss of predicting the gold from the vector
        """
        pred = T.nnet.sigmoid(tdot(vector,wSent,(0,0)) ) #+ bSent)
        return (pred-gold)**2
    
    sentembed = T.dmatrix('sentembed')
    leftindices = T.bvector('leftindices')
    rightindices= T.bvector('rightindices')
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
    
    return find_grad


# Load sentences

def get_data():
    """
    Returns a list of datapoints in the form:
    [embedding ids, left children, right children, sentiment scores]
    """
    
    data = []
    
    with open(bankDir+'SOStr.txt','r') as ftext, \
         open(bankDir+'STree.txt','r') as ftree:
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
            # Record all data
            data.append([array(list(map(getsent,ids))),
                         array(leftchildren),
                         array(rightchildren),
                         embids])
    
    return data
