class Node():
    def __init__(self): #, id, lemma):
        #self.id = id
        #self.lemma = lemma
        self.incoming = set()
        self.outgoing = set()

graph = dict()

with open('../data/example_dmrs.txt','r') as f:
    for line in f:
        y = line.split('><')
        x = y[0].split()
        if x[0] == '<node':
            nodeid = x[1].split("'")[1]
            graph[nodeid] = Node()
        elif x[0] == '<link':
            from_node = x[1].split("'")[1]
            to_node = x[2].split("'")[1]
            graph[from_node].outgoing.add(graph[to_node])
            graph[to_node].incoming.add(graph[from_node])
            

for id, node in graph.items():
    print(id, node.incoming, node.outgoing)