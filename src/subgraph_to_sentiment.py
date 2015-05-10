import pickle
from rnn import getid, getsent
from numpy import array
from math import ceil
#from dmrs import Graph, Node


bankDir = '../data/stanfordSentimentTreebank/'

granularity = 5

train = []
test = []
dev = []

predid = {}
vocab = 0
with open('../data/sentibank123_graph.pk','rb') as f:
    for line in f:
        text = line.strip()
        if text:
            graph = pickle.loads(text)
            for node in graph:
                if not node.lemma in predid:
                    predid[node.lemma] = vocab
                    vocab += 1

def getpredid(lemma):
    return predid[lemma]

print(len(predid))
raise Exception

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
        align_line = falign.readline()
        graph = pickle.load(fgraph)
        if graph:
            # Look up IDs for lemmas 
            lemmas = [node.lemma for node in graph]
            predids = tuple(map(getpredid, lemmas))
            # Construct constituent phrases to find their spans and sentiments
            tokens = line.rstrip().split('|')
            N = len(tokens)
            tok_ids = list(map(getid, tokens))
            spans = [(i,i) for i in range(N)]
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
                spans.append((first,last))
                tok_ids.append(getid(text))
            tok_scores = list(map(getsent,tok_ids))
            if granularity:
                tok_scores = [max(ceil(x*granularity)-1,0) for x in tok_scores]  # e.g. [0,0.2],(0.2,0.4],(0.4,0.6],etc.
            span_to_score = {spans[n]:score for n,score in enumerate(tok_scores)}
            
            # Iterate through graph, bottom up
            # For leaves, find the score
            # For non-leaves, find both the individual score as well as the subgraph score
            
            # To find a score, find closest constituent span
            #   Check same, extra on the left, extra on the right,
            #   less on the right, less on the left, repeat
            
            # Record data
            datapoint = [predids] #, heads, deps1...k, scores
            section = split_line.strip().split(',')[-1]
            if section == '1':
                train.append(datapoint)
            elif section == '2':
                test.append(datapoint)
            else:
                dev.append(datapoint)

[train, test, dev]