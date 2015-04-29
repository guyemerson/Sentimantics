import pickle, sys
from argparse import ArgumentParser
from theano import tensor as T, shared #, function, scan #, compile as theano_compile
tdot = T.tensordot
from numpy import float64, array, zeros_like
from numpy.random import randn, shuffle
#from scipy import sparse
from rnn import vocab, update_function, gradient_function, get_data

_print = print
def print(*obj, **kws):
    _print(*obj, **kws)
    sys.stdout.flush()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--dim', type=int, default=16)
    parser.add_argument('--rate', type=float64, default=4.)
    parser.add_argument('--ada', type=float64, default=0.5)
    parser.add_argument('--mom', type=float64, default=0.5)
    parser.add_argument('--reg', type=float64, nargs=4, default=[2.,4,8,1])
    parser.add_argument('--batch', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--init', type=float64, default=0.1)
    
    arg = parser.parse_args()
    
    arg.filename = '../data/model/{}.pk'.format(arg.filename)
    
    dim = arg.dim
    norm = arg.init/dim
    
    wQuad = shared(norm*randn(dim,dim,dim), name='wQuad')
    wLin  = shared(norm*randn(dim,2*dim), name='wLin')
    wSent = shared(norm*randn(dim), name='wSent')
    embed = shared(norm*randn(vocab,dim), name='embed')
    
    params = [wQuad, wLin, wSent, embed]
    
    train, test, dev = get_data()
    n_items = len(train)
    
    if arg.batch == 0:
        arg.reg = [x/n_items for x in arg.reg]
        arg.mom = 1 - (1-arg.mom)/n_items
        arg.ada = 1 - (1-arg.ada)/n_items
        arg.rate /= n_items
    
    update = update_function(params,
                             learningRate = arg.rate,
                             adaDecayCoeff = arg.ada,
                             momDecayCoeff = arg.mom,
                             regularisation = arg.reg)
    gradient = gradient_function(wQuad, wLin, wSent)
    
    def save():
        with open(arg.filename, 'wb') as f:
            pickle.dump([x.get_value() for x in params], f)
    
    print('Beginning training')
    
    for i in range(arg.epoch):
        shuffle(train)
        v = 1
        for x in train:
            embeddings = embed.get_value(borrow=True)  # This might slow down a GPU?
            embids, left, right, scores = x
            sentembed = array([embeddings[j] for j in embids])
            grad = gradient(sentembed, left, right, scores)
            embgrad = zeros_like(embeddings) # sparse.lil_matrix(embeddings.shape)
            for n, j in enumerate(embids):
                embgrad[j] += grad[3][n]
            grad[3] = embgrad #.tocsr()
            update(*grad)
            if v==100: v=1; print(wSent.get_value(borrow=True))
            else: v+=1
        print(i+1)
        save()
    
    
    
"""For batch-learning...
def descend():
    embeddings = embed.get_value()
    
    total_grad = [zeros_like(wQuad.get_value()),
                  zeros_like(wLin.get_value()),
                  zeros_like(wSent.get_value()),
                  zeros_like(embed.get_value())]
    
    for x in data:
        embids, left, right, scores = x
        sentembed = array([embeddings[j] for j in embids])
        grad = gradient(sentembed, left, right, scores)
        for i in range(3):
            total_grad[x] += grad[x]
        for n, j in enumerate(embids):
            total_grad[3][j] += grad[3][n]
    
    update(*total_grad)"""
"""Interactive...
pos = ['good','great','interesting','wonderful','terrific','stunning','funny','fantastic']
neg = ['bad','awful','terrible','boring','dull','uninteresting','lifeless','poor']
with open('../data/model/y.pk', 'rb') as f:
    stuff = pickle.load(f)
def word2sent(text):
    return stuff[2].dot(stuff[3][getembid(getid(text))])
minitest()
def minitest():
    for w in pos: print(word2sent(w))
    print('-')
    for w in neg: print(word2sent(w))
minitest()
embeddings=embed.get_value()
m_embids = (19753, 16077)
madeup = [[embeddings[j] for j in m_embids], (0,), (1,), (1,0.6,1)]
madeup = [array(x) for x in madeup]
m_grad = gradient(*madeup)
m_embgrad = zeros_like(embeddings)
for n, j in enumerate(m_embids):
    m_embgrad[j] += m_grad[3][n]
m_grad[3] = m_embgrad
update(*m_grad)"""