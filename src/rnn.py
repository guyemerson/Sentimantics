from theano import tensor as T, shared, function
tdot = T.tensordot
from numpy import zeros, longfloat, array, zeros_like
from numpy.random import randn

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


# Set up weights and embeddings

print("Initialising weights...")

dim = 1 #30

norm = 1/dim

wQuad = shared(norm*randn(dim,dim,dim), name='wQuad')
wLin  = shared(norm*randn(dim,2*dim), name='wLin')
bias  = shared(norm*randn(dim), name='bias')
embed = shared(randn(vocab,dim), name='embed')
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

normalise = function([], updates={embed: embed / T.sqrt((embed**2).sum(1).dimshuffle(0,'x'))})
normalise()


print("Setting up Theano functions...")

# Functions to update weights, using AdaGrad and momentum
# (Add in regularisation...)

# Hyperparameters
rate = shared(1., name='rate')
adaDecay = shared(0.9, name='adaDecay')
momDecay = shared(0.5, name='momDecay')

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
                  ((wQuadDel, momDecay*wQuadDel + (T.abs_(dwq)>0)*rate*T.sqrt(wQuadSq)*dwq),
                   (wLinDel,  momDecay*wLinDel  + (T.abs_(dwl)>0)*rate*T.sqrt(wLinSq) *dwl),
                   (biasDel,  momDecay*biasDel  + (T.abs_(db) >0)*rate*T.sqrt(biasSq) *db),
                   (embedDel, momDecay*embedDel + (T.abs_(de) >0)*rate*T.sqrt(embedSq)*de),
                   (wSentDel, momDecay*wSentDel + (T.abs_(dws)>0)*rate*T.sqrt(wSentSq)*dws),
                   (bSentDel, momDecay*bSentDel + (T.abs_(dbs)>0)*rate*T.sqrt(bSentSq)*dbs)),
                  allow_input_downcast=True) #Is downcast needed here?

update_weights = function([],updates=
                  ((wQuad, wQuad - wQuadDel),
                   (wLin,  wLin  - wLinDel),
                   (bias,  bias  - biasDel),
                   (embed, embed - embedDel),
                   (wSent, wSent - wSentDel),
                   (bSent, bSent - bSentDel)),
                  allow_input_downcast=True) #Not quite sure why downcast needed here

def update(dwq,dwl,db,de,dws,dbs):
    update_sum(dwq,dwl,db,de,dws,dbs)
    update_step(dwq,dwl,db,de,dws,dbs)
    update_weights()

# Functions to calculate the RNN operations

left  = T.dvector('left')
right = T.dvector('right')
cat = T.concatenate([left,right])
linear = tdot(tdot(wQuad,left,((2),(0))),right,((1),(0))) + tdot(wLin,cat,((1),(0))) + bias
output = linear * (linear >= 0)
combine = function([left,right], output)

vec = T.dvector('vec')
gold = T.dscalar('gold')
pred = T.nnet.sigmoid( tdot(wSent,vec,((0),(0))) + bSent )
sqdiff = (pred-gold)**2
predict = function([vec],pred)

gEmb, gwSent, gbSent = T.grad(sqdiff, [vec,wSent,bSent])
diffPredict = function([vec,gold], [gEmb, gwSent, gbSent])


# Testing...
print('On to the test...')

wSent.set_value(array([1])) # make things simple for now...
for _ in range(100):
    vectors = embed.get_value()
    gradEmb = zeros((vocab,dim))
    gradWeiSent = zeros(dim)
    gradBiaSent = longfloat()
    for pid, eid in embeddingid.items():
        ge, gws, gbs = diffPredict(vectors[eid],sentiment[pid])
        gradEmb[eid] += ge
        #gradWeiSent += gws
        #gradBiaSent += gbs #leave out the bias...
    update(zeros((dim,dim,dim)),zeros((dim,2*dim)),zeros(dim),gradEmb,gradWeiSent,gradBiaSent)
    for x in ['thoughtful','entertaining','awful','dreadful']:
        print(x, predict(vectors[embeddingid[phraseid[x]]]))


# Load sentences

with open(bankDir+'SOStr.txt','r') as ftext, \
     open(bankDir+'STree.txt','r') as ftree:
    for line in ftext:
        tokens = line.rstrip().split('|')
        ids = list(map(getid, tokens))
        tree = [int(x) for x in next(ftree).rstrip().split('|')]
        reverse = tree[::-1]
        children = {x:(tree.index(x)+1, len(tree)-reverse.index(x)) for x in range(len(tokens)+1,2*len(tokens))}
        for i in range(len(ids)+1,len(tree)+1):
            nonterm = [i]
            terminals = []
            while nonterm:
                left, right = children[nonterm.pop()]
                if left < len(tokens)+1 : terminals.append(left)
                else: nonterm.append(left)
                if right < len(tokens)+1: terminals.append(right)
                else: nonterm.append(right)
            text = ' '.join(tokens[min(terminals)-1:max(terminals)])
            ids.append(phraseid[text])
        scores = list(map(getsent,ids))