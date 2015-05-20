import pickle, sys
from argparse import ArgumentParser
from theano import tensor as T, shared, compile as theano_compile
tdot = T.tensordot
theano_compile.mode.Mode(linker='cvm', optimizer='fast_run')
from numpy import float64, array, zeros_like, sign
from numpy.random import randn, shuffle
from math import sqrt
from rnn import vocab, update_function, gradient_function, get_data
from rnn import pred_vocab, maxlink, gradient_dmrs, get_dmrs_data

_print = print
def print(*obj, **kws):  # @ReservedAssignment
    _print(*obj, **kws)
    sys.stdout.flush()

def initialise(gran, norm, dim, dmrs):
    wQuad = shared(norm*randn(dim,dim,dim), name='wQuad')
    if gran:
        wSent = shared(norm*randn(gran,dim), name='wSent')
    else:
        wSent = shared(norm*randn(dim), name='wSent')
    if dmrs:
        wLin  = shared(norm*randn(2,dim,dim), name='wLin')
        embed = shared(norm*randn(pred_vocab,dim), name='embed')
    else:
        wLin  = shared(norm*randn(dim,2*dim), name='wLin')
        embed = shared(norm*randn(vocab,dim), name='embed')
    
    return [wQuad, wLin, wSent, embed]

def evaluate(embeddings, loss_fn, data, backoff_emb=None, backoff_loss=None, sum_all=True, RMSq=False):
    loss = 0
    if backoff_emb:
        for first, x in data:
            embids, rest = x[0], x[1:]
            if first:
                sentembed = array([embeddings[j] for j in embids])
                loss += loss_fn(sentembed, *rest)
            else:
                sentembed = array([backoff_emb[j] for j in embids])
                loss += backoff_loss(sentembed, *rest)
    else:
        for x in data:
            embids, rest = x[0], x[1:]
            sentembed = array([embeddings[j] for j in embids])
            loss += loss_fn(sentembed, *rest)
    if sum_all:
        N = sum(len(x[-1]) for x in data)
    else:
        N = len(data)
    if RMSq:
        return 1 - sqrt(loss/N)
    else:
        return 1 - loss/N


if __name__ == '__main__':
    # To run from the command line
    parser = ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--dir', default='../data/model/{}.pk')
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('-dmrs', action='store_const', const=True, default=False)
    parser.add_argument('-labs', action='store_const', const=True, default=False)
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
    
    arg.norm = arg.init/arg.dim
    params = initialise(arg.gran, arg.norm, arg.dim, arg.dmrs)
    wQuad, wLin, wSent, embed = params
    
    if arg.dmrs:
        train, test, dev = get_dmrs_data()
    else:
        train, test, dev = get_data()
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
    
    if arg.dmrs:
        token_gradient, error, classes = gradient_dmrs(wQuad, wLin, wSent, maxlink+1, arg.gran, arg.neigh)
        if arg.labs:
            raise NotImplementedError
        else:
            lab_gradient = token_gradient
            lab_error = error
            lab_classes = classes
            def token_gradient(ids,chi,lab,sen):
                return lab_gradient(ids,chi,sen)
            def error(ids,chi,lab,sen):
                return lab_error(ids,chi,sen)
            def classes(ids,chi,lab,sen):
                return lab_classes(ids,chi,sen)
    else:
        token_gradient, error, classes = gradient_function(wQuad, wLin, wSent, arg.gran, arg.neigh)
    
    # Calculate the gradient from embedding indices, not embedding vectors
    def gradient(embids, *rest):
        embeddings = embed.get_value(borrow=True)  # This might slow down a GPU?
        sentembed = array([embeddings[j] for j in embids])
        grad = token_gradient(sentembed, *rest)
        embgrad = zeros_like(embeddings)
        for n, j in enumerate(embids):
            embgrad[j] += grad[3][n]
        grad[3] = embgrad
        return grad
    
    if arg.adareg:
        update, squareSum, stepSize = update_function(params, arg.rate, arg.ada, arg.mom, False, False)
        noreg_gradient = gradient
        def gradient(*args):
            grads = noreg_gradient(*args)
            for n,x in enumerate(grads):
                mat = params[n].get_value(borrow=True)
                x += reg2[n]*mat
                x += reg1[n]*sign(mat)
            return grads
    else: 
        update, squareSum, stepSize = update_function(params, arg.rate, arg.ada, arg.mom, reg_one, reg_two)
    
    aux_params = [squareSum, stepSize]
    
    arg.savefile = arg.dir.format(arg.filename)
    arg.auxsavefile  = arg.dir.format(arg.filename+'-aux')
    arg.loadfile = arg.dir.format(arg.load)
    arg.auxloadfile = arg.dir.format(arg.load+'-aux')
    
    def save():
        with open(arg.savefile, 'wb') as f:
            pickle.dump([x.get_value() for x in params], f)
        with open(arg.auxfile, 'wb') as f:
            pickle.dump([[x.get_value() for x in z] for z in aux_params], f)
    
    def load():
        with open(arg.loadfile, 'rb') as f:
            values = pickle.load(f)
            for n,x in enumerate(values):
                params[n].set_value(x)
        try:
            with open(arg.auxloadfile, 'rb') as f:
                values = pickle.load(f)
                for n,x in enumerate(values):
                    for m,y in enumerate(x):
                        aux_params[n][m].set_value(y)
        except FileNotFoundError:
            pass
    
    if arg.load: load()
    
    def accuracy(data=dev, soft=False):
        if not arg.gran: soft=True
        if soft:
            fn = error
        else:
            def fn(*args):
                loss = 0
                iterator = enumerate(classes(*args[:-1]))
                for n,x in iterator:
                    if x != args[-1][n]:
                        loss += 1
                return loss
        
        return evaluate(embed.get_value(borrow=True), fn, data, RMSq=not(arg.gran))

    
    print('Beginning training')
    
    for i in range(arg.epoch):
        shuffle(train)
        v = 1
        for x in train:
            # Find the gradient and descend
            grad = gradient(*x)
            update(*grad)
            # Print, periodically
            if v==100: v=1; print(wSent.get_value(borrow=True))
            else: v+=1
        print("\nEpoch {} complete!\nPerformance on devset:\n\nsoft {}\nhard {}\n\n".format(i+1, accuracy(soft=True), accuracy(soft=False)))
        save()
    

# For interactive mode
if False:
    from argparse import Namespace
    arg = Namespace()
    arg.dir = '../data/model/{}.pk'
    arg.dmrs = True
    arg.labs = False
    arg.dim = 25
    arg.adareg = True
    arg.init = 0.001
    arg.gran = 5
    arg.neigh = 0.5
    
    import numpy
    
    print(*(numpy.max(x.get_value()) for x in params), sep='\n')
    len([x for x in embed.get_value() if numpy.all(x==0)])/len(embed.get_value())
    len([x for x in embed.get_value() if numpy.all(x==0)])
    print(*(len([y for y in x.get_value().flat if y==0])/len(x.get_value().flatten()) for x in params), sep='\n')
    
    v = T.dvector()
    from theano import function
    from rnn import getid, getembid, getpredid
    softmax = function([v],T.nnet.softmax(v))
    
    pos = ['good','great']#,'interesting','wonderful','terrific','stunning','funny','fantastic']
    neg = ['bad','terrible']#,'boring','dull','uninteresting','awful','lifeless','poor']
    def word2sent(text):
        return numpy.round(softmax(numpy.tensordot(wSent.get_value(True),embed.get_value(True)[getembid(getid(text))],(1,0))), 3)
    
    pos = ['good_a_at-for-of_rel','great_a_1_rel'] #'great_a_at_rel'
    neg = ['bad_a_1_rel','terrible_a_1_rel'] #'bad_a_at_rel', 'terrible_a_for_rel'
    def word2sent(text):
        return numpy.round(softmax(numpy.tensordot(wSent.get_value(True),embed.get_value(True)[getpredid(text)],(1,0))), 3)
    
    def minitest(p,n):
        for w in p: print(word2sent(w))
        print('-')
        for w in n: print(word2sent(w))
    def testfile(name, data=dev, soft=False, items=(pos,neg)):
        arg.loadfile = arg.dir.format(name)
        load()
        if items: minitest(*items)
        print(evaluate(data, soft=soft))
    
    def embof(text):
        return getembid(getid(text))
    def sentof(tokens, leftch, rightch):
        sentembed = array([embed.get_value(borrow=True)[i] for i in map(embof, tokens)])
        return classes(sentembed, leftch, rightch)
    