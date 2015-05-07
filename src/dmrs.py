from delphin.mrs import simplemrs
from copy import copy

# Lemmas that should be skipped
SKIP = []#['udef_q_rel', 'unknown_rel', 'def_implicit_q_rel']

# Classes for the modified DMRS graph
class Node():
    def __init__(self, nodeid, lemma):
        self.nodeid = nodeid
        self.lemma = lemma
        self.incoming = set()
        self.outgoing = set()
    def __repr__(self):
        return "{}: {}".format(self.nodeid, self.lemma)
    def children(self):
        return {x[1] for x in self.outgoing}
    def parents(self):
        return {x[1] for x in self.incoming}
    def reverse(self,link):
        label, child = link
        self.outgoing.remove ((label,child))
        child.incoming.remove((label,self))
        self.incoming.add ((label+'_rev',child))
        child.outgoing.add((label+'_rev',self))

class Graph(dict):
    def __init__(self):
        super().__init__()
        self.root = None
    def __setitem__(self,key,value):
        assert type(key) == int
        assert type(value) == str
        super().__setitem__(key,Node(key,value))
    def __repr__(self):
        string = "ROOT: {}\n".format(self.root.nodeid)
        for node in self.values():
            string += str(node)
            if node.incoming:
                string += " ({})\n".format(" ".join(str(x[1].nodeid) for x in node.incoming))
            else:
                string += " (root)\n"
            for label, child in node.outgoing:
                string += "  {} {}".format(child.nodeid, label) + '\n'
        return string
    def __iter__(self):
        return iter(self.values())
    def remove(self,nodeid):
        for label,parent in self[nodeid].incoming:
            parent.outgoing.remove((label,self[nodeid]))
        for label,child in self[nodeid].outgoing:
            child.incoming.remove((label,self[nodeid]))
        self.pop(nodeid)

with open("../data/sentibank123.mrs",'r') as f:
    for line_num, line in enumerate(f):
        text = line.strip()
        if not text: continue
        mrs = simplemrs.loads_one(text)
        graph = Graph()  # The modified DMRS graph
        ignore = list()  # Nodes to be ignored
        # Set up all nodes in the graph, except those to be skipped
        for n in mrs.nodes:
            #print(n.nodeid, n.pred)
            nodeid = n.nodeid
            pred = str(n.pred).strip('"_')
            if pred in SKIP:
                ignore.append(nodeid)
            else:
                graph[nodeid] = pred
        # Set up links in the graph, reversing as necessary
        for l in mrs.links:
            #print(l.argname, l.post, l.start, l.end)
            # Ignore the ignored
            if l.start in ignore or l.end in ignore:
                continue
            if not l.argname:
                if l.start == 0:
                    graph.root = graph[l.end]
                else: #Do something with unlabelled EQ arcs?
                    print(l.argname, l.post, l.start, l.end)
                continue
            label = l.argname + '_' + l.post
            graph[l.start].outgoing.add((label, graph[ l.end ]))
            graph[ l.end ].incoming.add((label, graph[l.start]))
        # Reverse arcs as necessary
        for node in list(graph):
            # Remove coordination
            if node.lemma in ['and_c_rel']:
                leftenkel, rightenkel = set(), set()
                for label, child in node.outgoing:
                    if label[0] == 'L':
                        leftenkel.update(child.children())
                    else:
                        rightenkel.update(child.children())
                both = leftenkel & rightenkel
                if both:
                    for child in node.children():
                        for label, enkel in copy(child.outgoing):
                            if enkel in both:
                                child.reverse((label,enkel))
                    graph.remove(node.nodeid)
                
            # Reverse links for certain predicates
            for link in copy(node.outgoing):
                if (link[0] == 'ARG1_NEQ' and node.lemma in ['appos_rel', 'parenthetical_rel', 'compound_name_rel', 'compound_rel']) \
                  or (link[0] == 'RSTR_H'):
                    node.reverse(link)
            # Reverse links for modifiers
            modlinks = [x for x in node.outgoing if x[0][:3] == 'ARG' and x[0][-3:] == '_EQ']
            if len(modlinks) == 1:
                node.reverse(modlinks[0])
            elif len(modlinks) > 1:
                modlinks.sort(key=lambda x:x[0])
                node.reverse(modlinks[-1])
        
        """
        # Check that the whole graph is accessible from the 'root'
        cover = {graph.root}
        children = graph.root.children()
        while children:
            child = children.pop()
            cover.add(child)
            children.update(child.children())
            children.difference_update(cover)
        if len(cover) == len(graph):
            print('Rooted')
        else:
            print(line_num+1)
            #print(graph)
            for node in graph:
                if not node in cover:
                    print(node)
            #break
        """
        # Find roots and leaves of the graph
        root = list()
        leaf = list()
        for node in graph:
            if not node.incoming:
                root.append(node)
            if not node.outgoing:
                leaf.append(node)
        # Check for cycles
        discard = set()
        parents = set()
        for node in leaf:
            discard.add(node)
            for _, mother in node.incoming:
                parents.add(mother)
        n = True
        while n:
            n = 0
            new_parents = set()
            for mother in parents:
                if ({child for _, child in mother.outgoing} - discard):
                    new_parents.add(mother)
                else:
                    n += 1
                    discard.add(mother)
                    for _, grand in mother.incoming:
                        new_parents.add(grand)
            parents = new_parents
        
        if parents:
            print(line_num+1, "cycle", parents, set(graph) - discard)
        
        elif len(root) == 1:
            print(line_num+1, "good!")
            continue
        
        else:
            print(line_num+1, "multiple roots", end='')
            # Check if connected
            cover = set()
            queue = {child for _, child in root[0].outgoing}  # Set of nodes immediately dominated by the first root
            while queue:
                new = queue.pop()
                cover.add(new)
                for _, child in new.outgoing | new.incoming:
                    if not (child in queue or child in cover):
                        queue.add(child)
            if len(cover) == len(graph):
                print(":")
            else:
                print(" (disconnected):")
            
            for node in root:
                print(' ', node.nodeid, node.lemma, end=': ')
                for label, new_node in node.outgoing:
                    print(label, new_node.nodeid, end=', ')
                print()
            
        if line_num + 1 not in [2,3,5]:
            print(graph)
            break