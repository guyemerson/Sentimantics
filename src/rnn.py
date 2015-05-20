from theano import tensor as T, shared, function, scan, compile as theano_compile #, sparse as S 
tdot = T.tensordot
theano_compile.mode.Mode(linker='cvm', optimizer='fast_run')
from theano.tensor.nnet import softmax, sigmoid
from numpy import float64, array, zeros_like, eye
from math import ceil, floor
import pickle
from graph import Graph, Node  # @UnusedImport

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

predid = {}
linkid = {}
pred_vocab = 0
linknum = 1 # the first type of link will connect a fleshed out node with the bare node
maxlink = 0
with open('../data/sentibank123_graph.pk','rb') as f:
    try:
        while True:
            graph = pickle.load(f)
            if graph:
                for node in graph:
                    base = node.lemma #.split('_')[0]
                    if not base in predid:
                        predid[base] = pred_vocab
                        pred_vocab += 1
                    for label, _ in sorted(node.outgoing, key=lambda x:x[0]):
                        if not label in linkid:
                            linkid[label] = linknum
                            linknum += 1
                    maxlink = max(maxlink, len(node.outgoing))
                    #if len(node.outgoing) == 9:
                    #    print(graph)
    except EOFError:
        pass

def getpredid(lemma):
    return predid[lemma]
def getlinkid(label):
    return linkid[label]

# Function to update weights, using AdaGrad and momentum, and L1 regularisation

def update_function(parameters, learningRate, adaDecayCoeff, momDecayCoeff, reg_one, reg_two):
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
    if reg_one: assert len(reg_one.get_value()) == N
    if reg_two: assert len(reg_two.get_value()) == N
    
    gradient = [T.TensorVariable(p.type, name=p.name+'Grad') for p in parameters] #[:3]]
    #gradient.append(S.csr_matrix(parameters[3].name+'Grad', 'float64'))
    zero = [T.zeros_like(p) for p in parameters]
    squareSum = [shared(zeros_like(p.get_value()), name=p.name+'SqSum') for p in parameters]
    stepSize  = [shared(zeros_like(p.get_value()), name=p.name+'Step')  for p in parameters]
    
    rate = shared(learningRate, name='rate')
    adaDecay = shared(adaDecayCoeff, name='adaDecay')
    momDecay = shared(momDecayCoeff, name='momDecay')    
    
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
    
    if reg_one:
        regular_l1 = function([], updates=
                              list((parameters[i],
                                    T.switch(T.lt(abs(parameters[i]),reg_one[i]),
                                             zero[i],
                                             parameters[i] - reg_one[i]*T.sgn(parameters[i])))
                                   for i in range(N)),
                              allow_input_downcast=True)
    
    if reg_two:
        reg_two.set_value(array([1-x for x in reg_two.get_value()]))  # Convert to decay version
        regular_l2 = function([], updates=
                              list((parameters[i],
                                    reg_two[i]*parameters[i])
                                   for i in range(N)),
                              allow_input_downcast=True)
    
    def update(*grads):
        update_sum(*grads)
        update_step(*grads)
        update_wei()
        if reg_one: regular_l1()
        if reg_two: regular_l2()
    
    # If regularisation is part of the gradient, we still need to set weights to 0 appropriately for L1, i.e.:
    # don't allow weights to change sign in one step
    # if the weight is zero, the step size must be more than the adagrad-reduced (but momentum-increased?) L1 regularisation
    
    return update, squareSum, stepSize


def classify_and_cost_fns(wSent,granular,neighbour):
    """
    returns functions to classify vectors and find the error
    """
    if granular:
        def classify(vector):
            """
            Given an output vector, predict the class
            """
            prob = softmax(tdot(vector,wSent,(0,1)))[0]
            return T.argmax(prob)
        if not neighbour:
            def cost(vector, gold):
                """
                Given an output vector, and the gold standard class,
                calculate (one minus) the softmax probability of predicting the gold 
                """
                prob = softmax(tdot(vector,wSent,(0,1)))[0]  # softmax returns a matrix
                return 1 - prob[gold]
        else:
            partial = shared(eye(granular) + eye(granular,k=1)*neighbour + eye(granular,k=-1)*neighbour, 'partial', 'float64')
            def cost(vector, gold):
                """
                As above, but award some points for predicting the neighbouring class
                """
                prob = softmax(tdot(vector,wSent,(0,1)))[0]  # softmax returns a matrix
                return 1 - tdot(prob, partial[gold], (0,0))
    else:
        def classify(vector):
            """
            Given an output vector, predict the sentiment
            """
            return sigmoid(tdot(vector,wSent,(0,0)) )
        def cost(vector, gold):
            """
            Given an output vector, and the gold standard annotation,
            calculate the square loss of predicting the gold from the vector
            """
            pred = sigmoid(tdot(vector,wSent,(0,0))) #+ bSent)
            return (pred-gold)**2
    
    return classify, cost


# Function to calculate the gradient

def gradient_function(wQuad, wLin, wSent, granular=False, neighbour=False):
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
        out = tdot(tdot(wQuad,first,(2,0)),second,(1,0)) + tdot(wLin,cat,(1,0)) #+ bias
        rect = out * (out >= 0)
        new = T.set_subtensor(prev[cur], rect)
        return [cur+1, new]
    
    classify, cost = classify_and_cost_fns(wSent, granular, neighbour)
    
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
    
    classification, _ = scan(classify,
                             sequences=[vec],
                             outputs_info=None)
    
    grads = T.grad(total, [wQuad,wLin,wSent,sentembed])
    find_grad = function([sentembed,leftindices,rightindices,senti], grads, allow_input_downcast=True)
    find_error= function([sentembed,leftindices,rightindices,senti], total, allow_input_downcast=True)
    predict   = function([sentembed,leftindices,rightindices], classification, allow_input_downcast=True)
    
    return find_grad, find_error, predict


def gradient_dmrs(wQuad, wLin, wSent, max_children, granular=False, neighbour=False, labelled=False):
    """
    parameters:
    wQuad - 3rd order tensor for combining vectors
    wLin = [wHead, wDep] - two 2nd order tensors for combining vectors (applied to head and dependent, respectively)
    wSent - vector(s) for predicting sentiment
    """
    if labelled:
        raise NotImplementedError
    dim = wQuad.get_value(borrow=True).shape[0]
    
    def merge(children, cur, prev):
        """
        Given: a set of embeddings (prev),
         the index of next node (cur),
         and the indices of the child nodes (children), with the first item being the head
        calculate the embedding for the next node,
        and output the set of embeddings with this one updated (new),
        along with the index for the next node (cur+1).
        """
        v0 = prev[children[0]]
        v = T.zeros((max_children-1, dim), 'float64')
        for i in range(1,max_children):
            v = T.set_subtensor(v[i-1], prev[children[i]])  # Should be the zero vector if index >= cur
        mod = tdot(wQuad,v0,(1,0))
        out = tdot(wLin[0],v0,(1,0)) \
              + T.sum(tdot(wLin[1],v,(1,1)) + tdot(mod,v,(1,1)), 1)
        rect = out * (out >= 0)
        new = T.set_subtensor(prev[cur], rect)
        return [cur+1, new]
    
    
    classify, cost = classify_and_cost_fns(wSent, granular, neighbour)
    
    sentembed = T.dmatrix('sentembed')
    children = T.bmatrix('children')
    if granular:
        senti = T.bvector('senti')
    else:
        senti = T.dvector('senti')
    n,m = T.shape(sentembed)
    j,_ = T.shape(children)
    padded = T.concatenate((sentembed,T.zeros((j,m),'float64')))
    
    output, _ = scan(merge,
                     sequences=[children],
                     outputs_info=[n,
                                   padded])
    vec = output[-1][-1]  # Take just the set of embeddings from the last calculation
    
    loss, _ = scan(cost,
                   sequences=[vec,
                              senti],
                   outputs_info=None)
    total = T.sum(loss)
    
    classification, _ = scan(classify,
                             sequences=[vec],
                             outputs_info=None)
    
    grads = T.grad(total, [wQuad,wLin,wSent,sentembed])
    find_grad = function([sentembed,children,senti], grads, allow_input_downcast=True)
    find_error= function([sentembed,children,senti], total, allow_input_downcast=True)
    predict   = function([sentembed,children], classification, allow_input_downcast=True)
    
    # For one predicate:
    predembed = T.dvector('predembed')
    if granular:
        predsenti = T.bscalar('predsenti')
    else:
        predsenti = T.dscalar('predsenti')
    direct_class= classify(predembed)
    direct_cost = cost(predembed, predsenti)
    pred_grad = T.grad(direct_cost, [wSent,predembed])
    find_pred_grad = function([predembed,predsenti], pred_grad, allow_input_downcast=True)
    find_pred_error= function([predembed,predsenti], direct_cost, allow_input_downcast=True)
    pred_predict   = function([predembed], direct_class, allow_input_downcast=True)
    
    def find_grad_safe(emb, chi, sen):
        if chi.shape[0] > 0:
            return find_grad(emb, chi, sen)
        else:
            zero_grads = [zeros_like(wQuad.get_value(borrow=True)),
                          zeros_like(wLin.get_value(borrow=True))]
            wSent_grad, predembed_grad = find_pred_grad(emb[0], sen[0])
            return zero_grads + [wSent_grad, predembed_grad.reshape(1,-1)]
    
    def find_error_safe(emb, chi, sen):
        if chi.shape[0] > 0:
            return find_error(emb, chi, sen)
        else:
            return find_pred_error(emb[0], sen[0])
    
    def predict_safe(emb, chi):
        if chi.shape[0] > 0:
            return predict(emb, chi)
        else:
            return pred_predict(emb[0]).reshape(1,-1)
    
    return find_grad_safe, find_error_safe, predict_safe


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
                scores = array([max(ceil(x*granularity)-1,0) for x in scores], 'int8')  # e.g. [0,0.2],(0.2,0.4],(0.4,0.6],etc.
            else:
                scores = array(scores, 'float64')
            # Record data
            datapoint = [embids,
                         array(leftchildren, 'int8'),
                         array(rightchildren, 'int8'),
                         scores]
            section = fsplit.readline().strip().split(',')[-1]
            if section == '1':
                train.append(datapoint)
            elif section == '2':
                test.append(datapoint)
            else:
                dev.append(datapoint)
    
    return [train, test, dev]


def min_ignore(a,b):
    """Return the minimum of a and b, ignoring -1"""
    if a == -1:
        return b
    elif b == -1:
        return a
    else:
        return min(a,b)
def max_ignore(a,b):
    """Return the maximum of a and b, ignoring -1"""
    if a == -1:
        return b
    elif b == -1:
        return a
    else:
        return max(a,b)

def get_dmrs_data(granularity=5, backoff=False):
    """
    Returns three lists of datapoints in the form:
    [embedding ids, child_ids, child_labels, sentiment scores]
    The lists are: train, test, dev
    If granularity is nonzero, splits the sentiment score into as many bins
    """
    
    train, test, dev = [], [], []
    if backoff:
        orig_train, orig_test, orig_dev = get_data(granularity=granularity)
    
    with open(bankDir+'SOStr.txt','r') as ftext, \
         open(bankDir+'STree.txt','r') as ftree, \
         open(bankDir+'datasetSplit.txt','r') as fsplit, \
         open('../data/sentibank123_align.txt','r') as falign, \
         open('../data/sentibank123_graph.pk','rb') as fgraph:
        fsplit.readline()
        for line in ftext:
            # Load each data point
            tree_line = ftree.readline()
            split_line = fsplit.readline()
            section = split_line.strip().split(',')[-1]
            align_line = falign.readline()
            graph = pickle.load(fgraph)
            if graph:
                # Look up IDs for lemmas
                lemmas = []
                nodeid_to_orig = {}
                for i, node in enumerate(graph):
                    lemmas.append(node.lemma)
                    nodeid_to_orig[node.nodeid] = i
                predids = tuple(map(getpredid, lemmas))
                # Unpack alignment
                spans = [None] * len(lemmas)
                nodeid_to_span = {}
                for i, x in enumerate(align_line.split()):
                    nodeid, start, end = map(int,x.split(':'))
                    if nodeid in graph:
                        nodeid_to_span[nodeid] = (start, end)
                        spans[nodeid_to_orig[nodeid]] = (start, end)
                if 0 in graph:
                    start = min(x[0] for x in spans if x)
                    end   = max(x[1] for x in spans if x)
                    nodeid_to_span[0] = (start, end)
                    spans[nodeid_to_orig[0]] = (start, end)
                # Construct constituent phrases to find their spans and sentiments
                tokens = line.rstrip().split('|')
                N = len(tokens)
                tok_ids = list(map(getid, tokens))
                tok_spans = [(i,i) for i in range(N)]
                tree = [int(x)-1 for x in tree_line.rstrip().split('|')] # -1 for 0-indexing
                reverse = tree[::-1]
                leftchildren = []
                rightchildren = []
                for x in range(N,2*N-1):
                    leftchildren.append(tree.index(x))
                    rightchildren.append(2*N-2-reverse.index(x))
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
                    first = min(terminals)
                    last = max(terminals)
                    text = ' '.join(tokens[first:last+1])
                    tok_spans.append((first,last))
                    tok_ids.append(getid(text))
                tok_scores = list(map(getsent,tok_ids))
                if granularity:
                    tok_scores = [max(ceil(x*granularity)-1,0) for x in tok_scores]  # e.g. [0,0.2],(0.2,0.4],(0.4,0.6],etc.
                span_to_score = {tok_spans[n]:score for n,score in enumerate(tok_scores)}
                if granularity:
                    span_to_score[-1,-1] = floor(granularity/2)
                else:
                    span_to_score[-1,-1] = 0.5
                # Get spans for all subgraphs, and link information for nonterminals
                nodeid_to_full_span = {}
                nodeid_to_index = {}
                child_indices = []
                child_labels = []
                for i, node in enumerate(graph.bottom_up()):
                    nid = node.nodeid
                    start, end = nodeid_to_span[nid]
                    if node.outgoing:
                        child_indices.append([nodeid_to_orig[nid]])
                        child_labels.append([0])
                        for label, child in sorted(node.outgoing, key=lambda x:x[1].nodeid):
                            child_indices[-1].append(nodeid_to_index[child.nodeid])
                            child_labels[-1].append(getlinkid(label))
                            new_start, new_end = nodeid_to_full_span[child.nodeid]
                            start = min_ignore(start, new_start)
                            end = max_ignore(end, new_end)
                        # Add padding for Theano
                        while len(child_labels[-1]) <= maxlink:
                            child_indices[-1].append(len(spans))
                            child_labels[-1].append(linknum)
                        nodeid_to_full_span[nid] = (start,end)
                        nodeid_to_index[nid] = len(spans)
                        spans.append((start,end))
                    else:
                        nodeid_to_full_span[nid] = (start,end)
                        nodeid_to_index[nid] = nodeid_to_orig[nid]
                scores = []
                for start, end in spans:
                    if (start, end) in span_to_score:
                        scores.append(span_to_score[start,end])
                    else:
                        found = False
                        k = 0
                        while not found:
                            k += 1
                            # Try larger
                            for i in range(k+1):
                                new_start = start - k + i
                                new_end = end + i
                                if (new_start, new_end) in span_to_score:
                                    found = True
                                    break
                            # Try smaller
                            if not found and k <= (end-start):
                                for i in range(k+1):
                                    new_start = start + i
                                    new_end = end - k + i
                                    if (new_start, new_end) in span_to_score:
                                        found = True
                                        break
                        scores.append(span_to_score[new_start,new_end])
                # Record data
                datapoint = [array(predids),
                             array(child_indices),
                             array(child_labels),
                             array(scores)]
                if backoff:
                    datapoint = (True, datapoint)
                
                if section == '1':
                    train.append(datapoint)
                elif section == '2':
                    test.append(datapoint)
                else:
                    dev.append(datapoint)
            elif backoff:
                if section == '1':
                    train.append((False, orig_train[len(train)]))
                elif section == '2':
                    test.append((False, orig_test[len(test)]))
                else:
                    dev.append((False, orig_dev[len(dev)]))
    return [train, test, dev]