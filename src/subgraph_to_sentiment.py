import pickle
from rnn import getid, getsent
from numpy import array
from math import ceil, floor
from dmrs import Graph, Node


bankDir = '../data/stanfordSentimentTreebank/'

granularity = 5

train = []
test = []
dev = []

predid = {}
linkid = {}
vocab = 0
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
                        predid[base] = vocab
                        vocab += 1
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

# vocab 10483 removing after first underscore
# vocab 12468 without
# vocab 20483 in original
# linknum 39
# maxlink 9

def min_ignore(a,b):
    if a == -1:
        return b
    elif b == -1:
        return a
    else:
        return min(a,b)
def max_ignore(a,b):
    if a == -1:
        return b
    elif b == -1:
        return a
    else:
        return max(a,b)

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
                    while len(child_labels[-1]) < maxlink:
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
            '''
            for node in graph:
                print(node)
                nid = node.nodeid
                if node.outgoing:
                    orig = nodeid_to_orig[nid]
                    print(spans[orig], scores[orig])
                index = nodeid_to_index[nid]
                print(spans[index], scores[index])
            break
            '''
            # Record data
            datapoint = [array(predids),
                         array(child_indices),
                         array(child_labels),
                         array(scores)]
            section = split_line.strip().split(',')[-1]
            if section == '1':
                train.append(datapoint)
            elif section == '2':
                test.append(datapoint)
            else:
                dev.append(datapoint)
            
            if len(predids) < 3:
                print(graph)
                print(datapoint)
                break

[train, test, dev]