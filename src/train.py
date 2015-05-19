import pickle, sys
from argparse import ArgumentParser
from theano import tensor as T, shared, compile as theano_compile #, function, scan
tdot = T.tensordot
theano_compile.mode.Mode(linker='cvm', optimizer='fast_run')
from numpy import float64, array, zeros_like, sign
from numpy.random import randn, shuffle
#from scipy import sparse
from math import sqrt
from rnn import vocab, update_function, gradient_function, get_data
from rnn import pred_vocab, maxlink, gradient_dmrs, get_dmrs_data

_print = print
def print(*obj, **kws):  # @ReservedAssignment
    _print(*obj, **kws)
    sys.stdout.flush()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--dir', default='../data/model/{}.pk')
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('-dmrs', action='store_const', const=True, default=False)
    parser.add_argument('-lab', action='store_const', const=True, default=False)
    parser.add_argument('--dim', type=int, default=25)
    parser.add_argument('--rate', type=float64, default=1.)
    parser.add_argument('--ada', type=float64, default=0.5)
    parser.add_argument('--mom', type=float64, default=0.1)
    parser.add_argument('--l1', type=float64, nargs=4, default=[0.2,0.5,1.,0.05])
    parser.add_argument('--l2', type=float64, nargs=4, default=[1.,1,2,1])
    parser.add_argument('-adareg', action='store_const', const=True, default=False)
    parser.add_argument('--batch', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--init', type=float64, default=0.001)
    parser.add_argument('--gran', type=int, default=5)
    parser.add_argument('--neigh', type=float64, default=0.5)
    
    arg = parser.parse_args()
    
    arg.savefile = arg.dir.format(arg.filename)
    arg.loadfile = arg.dir.format(arg.load)
    
    dim = arg.dim
    norm = arg.init/dim
    
    if arg.lab:
        raise NotImplementedError
    
    wQuad = shared(norm*randn(dim,dim,dim), name='wQuad')
    if arg.gran:
        wSent = shared(norm*randn(arg.gran,dim), name='wSent')
    else:
        wSent = shared(norm*randn(dim), name='wSent')
    if arg.dmrs:
        wLin  = shared(norm*randn(2,dim,dim), name='wLin')
        embed = shared(norm*randn(pred_vocab,dim), name='embed')
        train, test, dev = get_dmrs_data()
    else:
        wLin  = shared(norm*randn(dim,2*dim), name='wLin')
        embed = shared(norm*randn(vocab,dim), name='embed')
        train, test, dev = get_data()
    
    params = [wQuad, wLin, wSent, embed]
    
    n_items = len(train)
    
    if arg.batch == 0:
        arg.l1 = [x/n_items for x in arg.l1]
        arg.l2 = [x/n_items for x in arg.l2]
        arg.mom **= 1/n_items
        arg.ada **= 1/n_items
        arg.rate /= n_items
    else:
        raise NotImplementedError
    
    reg_one = shared(array(arg.l1), name='reg1')
    reg_two = shared(array(arg.l2), name='reg2')
    
    reg1 = reg_one.get_value(borrow=True)
    reg2 = reg_two.get_value(borrow=True)
    
    if arg.adareg:
        update = update_function(params, arg.rate, arg.ada, arg.mom, False, False)
    else: 
        update = update_function(params, arg.rate, arg.ada, arg.mom, reg_one, reg_two)
    if arg.dmrs:
        gradient, error, classes, pred_grad, pred_err, pred_class = gradient_dmrs(wQuad, wLin, wSent, maxlink+1, arg.gran, arg.neigh)
    else:
        gradient, error, classes = gradient_function(wQuad, wLin, wSent, arg.gran, arg.neigh)
    
    def add_reg(grads):
        for n,x in enumerate(grads):
            mat = params[n].get_value(borrow=True)
            x += reg2[n]*mat
            x += reg1[n]*sign(mat)
    
    def save():
        with open(arg.savefile, 'wb') as f:
            pickle.dump([x.get_value() for x in params], f)
    
    def load(filename):
        with open(filename, 'rb') as f:
            values = pickle.load(f)
            for n,x in enumerate(values):
                params[n].set_value(x)
    
    if arg.load:
        load(arg.loadfile)
    
    def evaluate(data, soft=True):
        if not arg.gran: soft=True
        loss = 0
        embeddings = embed.get_value(borrow=True)
        for x in data:
            if arg.dmrs:
                embids, children, _, scores = x
            else:
                embids, left, right, scores = x
            sentembed = array([embeddings[j] for j in embids])
            if soft:
                if arg.dmrs:
                    if children.shape[0] > 0:
                        loss += error(sentembed, children, scores)
                    else:
                        loss += pred_err(sentembed[0], scores[0])
                else:
                    loss += error(sentembed, left, right, scores)
            else:
                if arg.dmrs:
                    if children.shape[0] > 0:
                        iterator = enumerate(classes(sentembed, children))
                    else:
                        iterator = [(0, pred_class(sentembed[0])[()] )]
                else:
                    iterator = enumerate(classes(sentembed,left,right))
                for n,x in iterator:
                    if x != scores[n]:
                        loss += 1
        N = sum(len(x[-1]) for x in data)
        if arg.gran:
            return 1 - loss/N
        else:
            return 1 - sqrt(loss/N) 
    
    print('Beginning training')
    
    for i in range(arg.epoch):
        shuffle(train)
        v = 1
        for x in train:
            embeddings = embed.get_value(borrow=True)  # This might slow down a GPU?
            if arg.dmrs:
                embids, children, _, scores = x
                sentembed = array([embeddings[j] for j in embids])
                if children.shape[0] > 0:
                    grad = gradient(sentembed, children, scores)
                else:
                    grad = [zeros_like(wQuad.get_value(borrow=True)),
                            zeros_like(wLin.get_value(borrow=True))]
                    grad.extend(pred_grad(sentembed[0], scores[0]))
            else:
                embids, left, right, scores = x
                sentembed = array([embeddings[j] for j in embids])
                grad = gradient(sentembed, left, right, scores)
            embgrad = zeros_like(embeddings) # sparse.lil_matrix(embeddings.shape)
            for n, j in enumerate(embids):
                embgrad[j] += grad[3][n]
            grad[3] = embgrad #.tocsr()
            if arg.adareg: add_reg(grad)
            update(*grad)
            if v==100: v=1; print(wSent.get_value(borrow=True))
            else: v+=1
        print("\nEpoch {} complete!\nPerformance on devset:\n\nsoft {}\nhard {}\n\n".format(i+1, evaluate(dev), evaluate(dev, soft=False)))
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
pos = ['good','great']#,'interesting','wonderful','terrific','stunning','funny','fantastic']
neg = ['bad','terrible']#,'boring','dull','uninteresting','awful','lifeless','poor']
v = T.dvector()
from theano import function
import numpy
from rnn import getid, getembid
softmax = function([v],T.nnet.softmax(v))
arg.dir='../data/model/{}.pk'
print(*(numpy.max(x.get_value()) for x in params), sep='\n')
len([x for x in embed.get_value() if numpy.all(x==0)])/len(embed.get_value())
len([x for x in embed.get_value() if numpy.all(x==0)])
print(*(len([y for y in x.get_value().flat if y==0])/len(x.get_value().flatten()) for x in params), sep='\n')
def word2sent(text):
    return numpy.round(softmax(numpy.tensordot(wSent.get_value(True),embed.get_value(True)[getembid(getid(text))],(1,0))), 3)
def minitest():
    for w in pos: print(word2sent(w))
    print('-')
    for w in neg: print(word2sent(w))
def loadfile(name):
    load(arg.dir.format(name))
def testfile(name, data=dev, soft=True):
    loadfile(name)
    minitest()
    print(evaluate(data, soft=soft))
testfile('grain')
embeddings=embed.get_value()
m_embids = (19753, 16077)
madeup = [[embeddings[j] for j in m_embids], (0,), (1,), (1,0.6,1)]
madeup = [array(x) for x in madeup]
m_grad = gradient(*madeup)
m_embgrad = zeros_like(embeddings)
for n, j in enumerate(m_embids):
    m_embgrad[j] += m_grad[3][n]
m_grad[3] = m_embgrad
update(*m_grad)
def embof(text):
    return getembid(getid(text))
def sentof(tokens, leftch, rightch):
    sentembed = array([embed.get_value(borrow=True)[i] for i in map(embof, tokens)])
    return classes(sentembed, leftch, rightch)
"""