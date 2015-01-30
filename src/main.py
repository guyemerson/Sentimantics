class Node():
    def __init__(self, lemma): #, id, lemma):
        #self.id = id
        self.lemma = lemma
        self.incoming = set()
        self.outgoing = set()

graph = dict()

with open('../data/example_dmrs_3.txt','r') as f:
    for line in f:
        tags = line.split('><')
        first = tags[0].split()
        if first[0] == '<node':
            nodeid = first[1].split("'")[1]
            if tags[1][0:5] == 'gpred':
                lemma = tags[1][6:-7]
            elif tags[1][0:8] == 'realpred':
                second = tags[1].split()
                lemma = second[1][7:-1]
            graph[nodeid] = Node(lemma)
        elif first[0] == '<link':
            from_node = first[1].split("'")[1]
            to_node = first[2].split("'")[1]
            rargname = tags[1][9:-10]
            post = tags[2][5:-6]
            label = rargname + '_' + post
            if (post == 'EQ' # Make modifiers directional
             #or post == 'HEQ' ## Do not reverse nominalization, perhaps we could later cut out the extra node
              or label == 'RSTR_H'  # Reverse quantifiers
              or label == 'ARG1_H'  # Make subordination directional
              or (graph[from_node].lemma == 'compound_name_rel' and label == 'ARG1_NEQ')  # Make ARG1 of the compound_name_rel the head...
              or (graph[from_node].lemma == 'appos_rel' and label == 'ARG1_NEQ')  # Make ARG1 of the appos_rel the head...
              ): 
                label += '_rev'
                from_node, to_node = to_node, from_node
            graph[from_node].outgoing.add((label, graph[to_node]))
            graph[to_node].incoming.add((label, graph[from_node]))
            

for id, node in graph.items():
    if node.incoming:
        continue
    print(id, node.lemma, end=': ')
    for label, new_node in node.outgoing:
        print(label, new_node.lemma, end=', ')
    print()
    #print(id, node.incoming, node.outgoing)