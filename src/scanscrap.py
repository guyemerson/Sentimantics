from numpy import array
from theano import tensor as T, function, scan, shared, compile
#from theano.ifelse import ifelse
tdot = T.tensordot

compile.mode.Mode(linker='cvm', optimizer='fast_compile')

weights = shared(array([[1.,0,0,1,0,0],[0,1,0,0,1,0],[0,0,1,0,0,1]]), name='weights')
predict = shared(array([1.,1,-1]), name='predict')

embed = T.dmatrix('embed')
leftindices = T.bvector('leftindices')
rightindices= T.bvector('rightindices')
senti = T.dvector('senti')

def combine(left, right, cur, prev, matrix, num):
    first = prev[left]
    second= prev[right]
    cat = T.concatenate((first,second))
    out = tdot(matrix,cat,(1,0))
    rect = out * (out >= 0)
    #mask = T.concatenate([T.zeros((cur,)),T.ones((1,)),T.zeros((num-cur-1,))])
    #placeholder = T.outer(mask,rect)
    #new = prev + placeholder
    new = T.set_subtensor(prev[cur], rect, inplace=False)
    return [cur+1, new]

def diff(vector, gold, dual):
    pred = T.nnet.sigmoid(tdot(vector,dual,(0,0)))
    return (pred-gold)**2

# This seems rather hacky, but:
# Scan always passes tap slices to the function...
# No way to pass the entire list of results to the function?

# Is it possible to make a shared variable instead?

n,m = T.shape(embed)
padded = T.concatenate((embed,T.zeros((n-1,m),'float64')))

print('set up function...')
inds, up1 = scan(combine,
                 sequences=[leftindices,
                            rightindices],
                 outputs_info=[n,
                               padded],
                 non_sequences=[weights,
                                2*n-1])
vec = inds[-1][-1]
cost, up2 = scan(diff,
                 sequences=[vec,
                            senti],
                 outputs_info=None,
                 non_sequences=[predict])
total = T.sum(cost)
ge, gw, gp = T.grad(total, [embed,weights,predict])
find_grad = function([embed,leftindices,rightindices,senti],[ge,gw,gp,vec,cost],allow_input_downcast=True)


print('use function...')

embeddings = array([[1.,0,0.1],[0,1,0],[0.5,0,0.5]])
leftchildren = array([1,0])
rightchildren= array([2,3])
signal = array([0.,0,1,1,1])

stuff = find_grad(embeddings,leftchildren,rightchildren,signal)
print('\n'.join(str(x) for x in stuff))
