import pickle
from rnn import getid, getembid, getsent
from numpy import array
from math import ceil
#Take each node, find min and max
#Take each non-leaf, find all descendents, find min and max
#Find spans of constituents, and sentiment for each
#Take each min/max, find closest constituent span
# Check same, extra on the left, extra on the right,
# less on the right, less on the left, repeat

bankDir = '../data/stanfordSentimentTreebank/'

granularity = 5

train = []
test = []
dev = []

def getpredembid():
    pass

with open(bankDir+'SOStr.txt','r') as ftext, \
     open(bankDir+'STree.txt','r') as ftree, \
     open(bankDir+'datasetSplit.txt','r') as fsplit, \
     open('../data/sentibank123_align.txt','r') as falign, \
     open('../data/sentibank123_graph.pk','rb') as fgraph:
    all_graphs = pickle.load(fgraph)
    fsplit.readline()
    for n, line in enumerate(ftext):
        # Find: tokens, phrase ids, embedding ids, child nodes
        tokens = line.rstrip().split('|')
        N = len(tokens)
        ids = list(map(getid, tokens))
        graph = all_graphs[n]
        
        embids = tuple(map(getpredembid, ids))
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