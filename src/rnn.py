from theano import tensor as T, shared, function, scan #, compile as theano_compile
tdot = T.tensordot
from numpy import longfloat, array, zeros_like, zeros
from numpy.random import randn, shuffle
#from scipy.sparse import csr_matrix
import pickle

bankDir = '../data/stanfordSentimentTreebank/'

# Load dictionary from strings to IDs; and from IDs to sentiments

print("Loading dictionaries...")

phraseid = {}
sentiment = {}

with open(bankDir+'dictionary.txt','r') as f:
    for line in f:
        key, value = line.rstrip().split('|')
        phraseid[key] = int(value)

with open(bankDir+'sentiment_labels.txt','r') as f:
    f.readline()
    for line in f:
        key, value = line.rstrip().split('|')
        sentiment[int(key)] = longfloat(value)

def getid(phrase):
    return phraseid[phrase]

def getsent(pid):
    return sentiment[pid]

# Set up IDs for lexical items and find vocabulary size

embeddingid = {}
vocab = 0
for p,i in phraseid.items():
    if not ' ' in p:
        embeddingid[i] = vocab
        vocab += 1

def getembid(token):
    return embeddingid[token]


# Set up weights and embeddings

print("Initialising weights...")

dim = 30

norm = 1/dim

wQuad = shared(norm*randn(dim,dim,dim), name='wQuad')
wLin  = shared(norm*randn(dim,2*dim), name='wLin')
bias  = shared(zeros(dim), name='bias')
embed = shared(norm*randn(vocab,dim), name='embed')
wSent = shared(norm*randn(dim), name='wSent')
bSent = shared(0., name='bSent')

wQuadSq = shared(zeros_like(wQuad.get_value()), name='wQuadSq')
wLinSq  = shared(zeros_like(wLin.get_value()), name='wLinSq')
biasSq  = shared(zeros_like(bias.get_value()), name='biasSq')
embedSq = shared(zeros_like(embed.get_value()), name='embedSq')
wSentSq = shared(zeros_like(wSent.get_value()), name='wSentSq')
bSentSq = shared(zeros_like(bSent.get_value()), name='bSentSq')

wQuadDel = shared(zeros_like(wQuad.get_value()), name='wQuadDel')
wLinDel  = shared(zeros_like(wLin.get_value()), name='wLinDel')
biasDel  = shared(zeros_like(bias.get_value()), name='biasDel')
embedDel = shared(zeros_like(embed.get_value()), name='embedDel')
wSentDel = shared(zeros_like(wSent.get_value()), name='wSentDel')
bSentDel = shared(zeros_like(bSent.get_value()), name='bSentDel')

#normalise = function([], updates={embed: embed / T.sqrt((embed**2).sum(1).dimshuffle(0,'x'))})
#normalise()


print("Setting up Theano functions...")

# Functions to update weights, using AdaGrad and momentum, and L1 regularisation

# Hyperparameters
rate = shared(1., name='rate')
adaDecay = shared(0.9, name='adaDecay')
momDecay = shared(0.5, name='momDecay')
reg = shared(array([0.1,0.1,0.1,0.1,0.1,0.1]), name='reg')

dwq = T.dtensor3()
dwl = T.dmatrix()
db  = T.dvector()
de  = T.dmatrix()
dws = T.dvector()
dbs = T.dscalar()

update_sum = function([dwq,dwl,db,de,dws,dbs],updates=
                  ((wQuadSq, adaDecay*wQuadSq + dwq**2),
                   (wLinSq,  adaDecay*wLinSq  + dwl**2),
                   (biasSq,  adaDecay*biasSq  + db**2),
                   (embedSq, adaDecay*embedSq + de**2),
                   (wSentSq, adaDecay*wSentSq + dws**2),
                   (bSentSq, adaDecay*bSentSq + dbs**2)),
                  allow_input_downcast=True) #Is downcast needed here?

update_step = function([dwq,dwl,db,de,dws,dbs],updates=
                  ((wQuadDel, momDecay*wQuadDel + T.switch(T.gt(T.abs_(wQuadSq),0),rate/T.sqrt(wQuadSq)*dwq,T.zeros_like(wQuadSq))),
                   (wLinDel,  momDecay*wLinDel  + T.switch(T.gt(T.abs_(wLinSq) ,0),rate/T.sqrt(wLinSq) *dwl,T.zeros_like(wLinSq))),
                   (biasDel,  momDecay*biasDel  + T.switch(T.gt(T.abs_(biasSq) ,0),rate/T.sqrt(biasSq) *db, T.zeros_like(biasSq))),
                   (embedDel, momDecay*embedDel + T.switch(T.gt(T.abs_(embedSq),0),rate/T.sqrt(embedSq)*de, T.zeros_like(embedSq))),
                   (wSentDel, momDecay*wSentDel + T.switch(T.gt(T.abs_(wSentSq),0),rate/T.sqrt(wSentSq)*dws,T.zeros_like(wSentSq))),
                   (bSentDel, momDecay*bSentDel + T.switch(T.gt(T.abs_(bSentSq),0),rate/T.sqrt(bSentSq)*dbs,T.zeros_like(bSentSq)))),
                  allow_input_downcast=True) #Is downcast needed here?

update_weights = function([],updates=
                  ((wQuad, wQuad - wQuadDel),
                   (wLin,  wLin  - wLinDel),
                   (bias,  bias  - biasDel),
                   (embed, embed - embedDel),
                   (wSent, wSent - wSentDel),
                   (bSent, bSent - bSentDel)),
                  allow_input_downcast=True) #Not quite sure why downcast needed here

regularise = function([],updates=
                      ((wQuad, T.switch(T.lt(T.abs_(wQuad),reg[0]),T.zeros_like(wQuad),wQuad - reg[0]*T.sgn(wQuad))),
                       (wLin,  T.switch(T.lt(T.abs_(wLin), reg[1]),T.zeros_like(wLin), wLin  - reg[1]*T.sgn(wLin))),
                       (bias,  T.switch(T.lt(T.abs_(bias), reg[2]),T.zeros_like(bias), bias  - reg[2]*T.sgn(bias))),
                       (embed, T.switch(T.lt(T.abs_(embed),reg[3]),T.zeros_like(embed),embed - reg[3]*T.sgn(embed))),
                       (wSent, T.switch(T.lt(T.abs_(wSent),reg[4]),T.zeros_like(wSent),wSent - reg[4]*T.sgn(wSent))),
                       (bSent, T.switch(T.lt(T.abs_(bSent),reg[5]),T.zeros_like(bSent),bSent - reg[5]*T.sgn(bSent)))),
                      allow_input_downcast=True)

def update(Dwq,Dwl,Db,De,Dws,Dbs):
    update_sum(Dwq,Dwl,Db,De,Dws,Dbs)
    update_step(Dwq,Dwl,Db,De,Dws,Dbs)
    update_weights()
    regularise()

def update_nobias(Dwq,Dwl,Db,De,Dws,Dbs):
    update_sum(Dwq,Dwl,zeros_like(Db),De,Dws,zeros_like(Dbs))
    update_step(Dwq,Dwl,zeros_like(Db),De,Dws,zeros_like(Dbs))
    update_weights()
    regularise()


# Functions to calculate the RNN operations

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
    out = tdot(tdot(wQuad,first,((2),(0))),second,((1),(0))) + tdot(wLin,cat,((1),(0))) + bias
    rect = out * (out >= 0)
    new = T.set_subtensor(prev[cur], rect, inplace=False)
    return [cur+1, new]

def cost(vector, gold):
    """
    Given an output vector, and the gold standard annotation,
    calculate the square loss of predicting the gold from the vector
    """
    pred = T.nnet.sigmoid(tdot(vector,wSent,(0,0)) + bSent)
    return (pred-gold)**2


sentembed = T.dmatrix('sentembed')
leftindices = T.bvector('leftindices')
rightindices= T.bvector('rightindices')
senti = T.dvector('senti')
n,m = T.shape(sentembed)
padded = T.concatenate((sentembed,T.zeros((n-1,m),'float64')))

output, up1 = scan(merge,
                   sequences=[leftindices,
                              rightindices],
                   outputs_info=[n,
                                 padded])
vec = output[-1][-1]  # Take just the set of embeddings from the last calculation
loss, up2 = scan(cost,
                 sequences=[vec,
                            senti],
                 outputs_info=None)
total = T.sum(loss)
gwq,gwl,gb,sge,gws,gbs = T.grad(total, [wQuad,wLin,bias,sentembed,wSent,bSent])
find_grad = function([sentembed,leftindices,rightindices,senti],[gwq,gwl,gb,sge,gws,gbs],allow_input_downcast=True)

"""
# Testing...
print('On to the test...')
# make things simple for now...
find_output = function([sentembed,leftindices,rightindices,senti],[vec,loss],allow_input_downcast=True)

wQuad.set_value(array([[[1.]]]))
wLin.set_value(array([[1.,1.]]))
bias.set_value(array([0.]))
embed.set_value(array([[1.],[0],[0.5]]))
wSent.set_value(array([1.]))
bSent.set_value(0.)

test_embedding = embed.get_value()
test_left = array([1,0])
test_right = array([2,3])
test_signal = array([0.,0,1,1,1])

stuff = find_grad(test_embedding, test_left, test_right, test_signal)
other = find_output(test_embedding, test_left, test_right, test_signal)
for x in stuff:
    print(x)
for x in other:
    print(x)
update_nobias(*stuff)
for x in (wQuad,wLin,bias,embed,wSent,bSent):
    print(x.get_value())
"""

# Load sentences

print('Loading sentences')

data_scores = []
data_left = []
data_right = []
data_embids = []    

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
        data_scores.append(array(list(map(getsent,ids))))
        data_left.append(array(leftchildren))
        data_right.append(array(rightchildren))
        data_embids.append(embids)

# Now we just need to take batches and do the descent...
# Also take the train-dev-test split, and see how performance changes...

#N_folds = 

datapoints = list(range(len(data_scores)))  # 11855
#shuffle(datapoints)

def descend():
    embeddings = embed.get_value()
    
    total_grad = [zeros_like(wQuad.get_value()),
                  zeros_like(wLin.get_value()),
                  zeros_like(bias.get_value()),
                  zeros_like(embed.get_value()),
                  zeros_like(wSent.get_value()),
                  zeros_like(bSent.get_value())]
    
    for i in datapoints:
        scores = data_scores[i]
        left = data_left[i]
        right = data_right[i]
        sentembed = array([embeddings[j] for j in data_embids[i]])
        grad = find_grad(sentembed, left, right, scores)
        for x in (0,1,2,4,5):
            total_grad[x] += grad[x]
        for n, j in enumerate(data_embids[i]):
            total_grad[3][j] += grad[3][n]
    
    update_nobias(*total_grad)

if __name__ == '__main__':
    filename = '../data/model/first.pk'
    for i in range(1000):
        descend()
        