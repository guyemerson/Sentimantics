from delphin.mrs import simplemrs
from copy import copy, deepcopy

# Lemmas that should be skipped
SKIP = ['udef_q_rel', 'def_implicit_q_rel', 'focus_d_rel', 'id_rel']
MAYBE = ['unknown_rel']

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
    def reverse(self,link,verbose=False):
        label, child = link
        self.outgoing.remove((label,child))
        child.incoming.remove((label,self))
        if label[-4:] == '_rev':
            new_label = label[:-4]
        else:
            new_label = label+'_rev'
        self.incoming.add((new_label,child))
        child.outgoing.add((new_label,self))
        if verbose: print('reversed', self.nodeid, child.nodeid)
    def add_link(self,link,verbose=False):
        label, child = link
        self.outgoing.add((label,child))
        child.incoming.add((label,self))
        if verbose: print('added', label, self.nodeid, child.nodeid)
    def remove_link(self,link,verbose=False):
        label, child = link
        self.outgoing.remove((label,child))
        child.incoming.remove((label,self))
        if verbose: print('removed', label, self.nodeid, child.nodeid)

class Graph(dict):
    def __init__(self):
        super().__init__()
        self.root = None
        self.undirected = set()
    def __setitem__(self,key,value):
        assert isinstance(key,int)
        if isinstance(value,str):
            super().__setitem__(key,Node(key,value))
        else:
            assert value.nodeid == key
            super().__setitem__(key,value)
    def __repr__(self):
        string = "ROOT: {}\n".format(self.root)
        string += "roots: "+", ".join(str(x) for x in self.roots())+'\n'
        for node in self:
            string += str(node)
            if node.incoming:
                string += " ({})\n".format(" ".join(str(x[1].nodeid) for x in node.incoming))
            else:
                string += " (root)\n"
            for label, child in node.outgoing:
                string += "  {} {}".format(child.nodeid, label) + '\n'
        string += "undirected:\n"
        for pair in self.undirected:
            string += "  {}\n".format(pair)
        return string
    def __iter__(self):
        for x in sorted(self.keys()):
            yield self[x]
    def remove(self,node,verbose=False):
        for label,parent in node.incoming:
            parent.outgoing.remove((label,node))
        for label,child in node.outgoing:
            child.incoming.remove((label,node))
        self.pop(node.nodeid)
        for x,y in copy(self.undirected):
            if x == node or y == node:
                self.undirected.remove((x,y))
        if verbose: print('removed',node.nodeid)
    def leaves(self):
        for node in self:
            if not node.outgoing:
                yield node
    def roots(self):
        for node in self:
            if not node.incoming:
                yield node
    def cycle(self):
        discard = set(self.leaves())
        parents = set()
        for node in discard:
            parents.update(node.parents())
        n = True
        while n:
            n = 0 # Count how many parents we can remove
            new_parents = set()
            for mother in parents:
                if mother.children() - discard:
                    new_parents.add(mother)
                else:
                    n += 1
                    discard.add(mother)
                    new_parents.update(mother.parents())
            parents = new_parents
        if not parents:
            return False
        else:
            discard_top = set(self.roots())
            children = set()
            for node in discard_top:
                children.update(node.children())
            n = True
            while n:
                n = 0
                new_children = set()
                for daughter in children:
                    if daughter.parents() - discard_top:
                        new_children.add(daughter)
                    else:
                        n += 1
                        discard_top.add(daughter)
                        new_children.update(daughter.children())
                children = new_children
            return set(graph) - discard - discard_top
    def connected(self, avoid=set()):
        cover = set()
        queue = {(set(graph) - avoid).pop()} # Take some element not being avoided
        while queue:
            new = queue.pop()
            cover.add(new)
            for adjacent in new.parents() | new.children():
                if not (adjacent in cover or adjacent in avoid):
                    queue.add(adjacent)
        if len(cover) == len(graph) - len(avoid):
            return True
        else:
            return False

with open("../data/sentibank123.mrs",'r') as f:
    for line_num, line in enumerate(f):
        text = line.strip()
        if line_num+1 < 1: continue
        debug = False
        if not text: continue
        mrs = simplemrs.loads_one(text)
        graph = Graph()  # The modified DMRS graph
        ignore = list()  # Nodes to be ignored
        maybe = list()   # Nodes that might be ignored
        # Set up all nodes in the graph, except those to be skipped
        for n in mrs.nodes:
            nodeid = n.nodeid
            pred = str(n.pred).strip('"_')
            if pred in SKIP:
                ignore.append(nodeid)
            else:
                graph[nodeid] = pred
                if pred in MAYBE:
                    maybe.append(graph[nodeid])
        # Set up links in the graph
        for l in mrs.links:
            # Ignore the ignored
            if l.start in ignore or l.end in ignore:
                continue
            # Record undirected EQ links, and the root predicate
            end = graph[l.end]
            if l.start == 0:
                graph.root = end
                continue
            start = graph[l.start]
            if not l.argname:
                if not l.post == 'EQ':
                    raise Exception
                graph.undirected.add((start,end))
                continue
            label = l.argname + '_' + l.post
            start.add_link((label,end))
        
        if debug: print(graph)
        
        # Modify links as necessary
        for node in list(graph):
            # Reverse links for certain predicates
            done = False
            for link in copy(node.outgoing):
                if (link[0] == 'ARG1_NEQ' and node.lemma in ['appos_rel', 'parenthetical_rel', 'compound_name_rel', 'compound_rel', 'of_p_rel', 'but_p_except_rel']) \
                  or link[0] == 'RSTR_H': #or node.lemma == 'focus_d_rel':
                    node.reverse(link,verbose=debug)
                    done = True
            if done: continue
            # Reverse links for modifiers
            modlinks = [x for x in node.outgoing if x[0][:3] == 'ARG' and x[0][-3:] == '_EQ']
            if len(modlinks) == 1:
                node.reverse(modlinks[0],verbose=debug)
            elif len(modlinks) > 1:
                modlinks.sort(key=lambda x:x[0])
                node.reverse(modlinks[-1],verbose=debug)
        
        # Remove singletons
        if not graph.connected():
            for node in graph.roots():
                if not node.children():
                    graph.remove(node,verbose=debug)
        
        if debug: print(graph)
        
        # Add undirected links if this fixes connectivity
        if not graph.connected():
            for x,y in graph.undirected:
                if y in graph.roots():
                    x.add_link(("NIL_EQ",y),verbose=debug)
                    if not graph.connected():
                        x.remove_link(("NIL_EQ",y),verbose=debug)
                elif x in graph.roots():
                    y.add_link(("NIL_EQ",x),verbose=debug)
                    if not graph.connected():
                        y.remove_link(("NIL_EQ",x),verbose=debug)
        
        # Check if maybes can be removed
        for node in maybe:  # this might not be stable if either of two nodes could be removed but not both 
            if graph.connected(avoid={node}):
                graph.remove(node,verbose=debug)
        
        # If it's still not connected, just join things together
        if not graph.connected():
            graph[0] = 'ROOT'
            if debug: print('added ROOT')
            for root in set(graph.roots()) - {graph[0]}:
                graph[0].add_link(('NIL',root),verbose=debug)
        
        # Remove coordination if they're ending up as roots
        if len(list(graph.roots())) > 1:
            root_lemmas = {x.lemma for x in graph.roots()}
            root_lemmas_trunk = {x[-6:] for x in root_lemmas}
            if 'implicit_conj_rel' in root_lemmas or '_c_rel' in root_lemmas_trunk:
                for node in reversed(list(graph)):
                    if node.lemma[-6:] == '_c_rel' or node.lemma == 'implicit_conj_rel':
                        leftenkel, rightenkel, other_link = set(), set(), set()
                        for label, child in node.outgoing:
                            if label[:2] == 'L-':
                                leftenkel.update(child.children())
                                leftenkel.update({parent for link, parent in child.incoming if link[-4:] == '_rev'})
                            elif label[:2] == 'R-':
                                rightenkel.update(child.children())
                                rightenkel.update({parent for link, parent in child.incoming if link[-4:] == '_rev'})
                            else:
                                other_link.add((label,child))
                        both = leftenkel & rightenkel
                        if both:
                            for ch_lab, child in node.outgoing:
                                if ch_lab[1] != '-': continue
                                for label, enkel in copy(child.outgoing):
                                    if enkel in both:
                                        child.reverse((label,enkel),verbose=debug)
                                for label, parent in node.incoming:
                                    parent.add_link((label,child),verbose=debug)
                                for label, dependent in other_link:
                                    child.add_link((label,dependent),verbose=debug)
                            graph.remove(node,verbose=debug)
        
        # Add undirected links if this removes extra roots
        # First out of a special list, and then in general
        subord_list = ['implicit_conj_rel','and_c_rel','or_c_rel','but_c_rel','subord_rel','neg_rel']
        if len(list(graph.roots())) > 1:
            for x,y in graph.undirected:
                if not x.parents() and x.lemma in subord_list:
                    y.add_link(('NIL_EQ',x),verbose=debug)
                if not y.parents() and y.lemma in subord_list:
                    x.add_link(('NIL_EQ',y),verbose=debug)
        if len(list(graph.roots())) > 1:
            for x,y in graph.undirected:
                if not x.parents():
                    y.add_link(('NIL_EQ',x),verbose=debug)
                if not y.parents():
                    x.add_link(('NIL_EQ',y),verbose=debug)
        
        if debug: print(graph)
        
        # Remove cycles (this might unstable for complicated cycles?
        cycle_nodes = graph.cycle()
        if cycle_nodes:
            if debug: print(cycle_nodes)
            for node in cycle_nodes:
                for parent in node.parents():
                    if not parent in cycle_nodes:
                        for label, comrade in copy(node.incoming):
                            if comrade in cycle_nodes and comrade != parent:
                                comrade.reverse((label,node),verbose=debug)
        
        cycle_nodes = graph.cycle()
        if cycle_nodes:
            print(line_num+1, "cycle", cycle_nodes)
        
        elif len(list(graph.roots())) == 1:
            print(line_num+1, "good!")
            '''if line_num+1 == 48:
                print(graph)
                break'''
            continue
        
        else:
            print(line_num+1, "multiple roots", end='')
        
            if graph.connected():
                print(":")
            else:
                print(" (disconnected):")
            
            for node in graph.roots():
                print(' ', node.nodeid, node.lemma, end=': ')
                for label, new_node in node.outgoing:
                    print(label, new_node.nodeid, end=', ')
                print()
        
        # Stop when something goes wrong!
        if line_num+1 > 0:
            print(graph)
            break
        
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