from copy import copy
from collections import deque

# Classes for the modified DMRS graph

class Node():
    """
    A single node in the graph, which has:
    * an ID (integer)
    * a lemma (string)
    * incoming and outgoing links, represented as (label, node) tuples
    """
    
    def __init__(self, nodeid, lemma):
        """Initialise ID and lemma, set up empty sets for links"""
        assert isinstance(nodeid, int)
        assert isinstance(lemma, str)
        self.nodeid = nodeid
        self.lemma = lemma
        self.incoming = set()
        self.outgoing = set()
    
    def __repr__(self):
        """Display only nodeid and lemma"""
        return "{}: {}".format(self.nodeid, self.lemma)
    
    def children(self):
        """Return child nodes, as a set"""
        return {x[1] for x in self.outgoing}
    
    def parents(self):
        """Return parent nodes, as a set"""
        return {x[1] for x in self.incoming}
    
    def reverse(self,link,verbose=False):
        """Reverse an outgoing link"""
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
        """Add an outgoing link"""
        label, child = link
        self.outgoing.add((label,child))
        child.incoming.add((label,self))
        if verbose: print('added', label, self.nodeid, child.nodeid)
    
    def remove_link(self,link,verbose=False):
        """Remove an outgoing link"""
        label, child = link
        self.outgoing.remove((label,child))
        child.incoming.remove((label,self))
        if verbose: print('removed', label, self.nodeid, child.nodeid)


class Graph(dict):
    """
    A graph of Node objects, indexed by their IDs.
    There is also an additional set of undirected links.
    """
    
    def __init__(self):
        """Initialise as a dict, but with an optional 'root', and undirected links"""
        super().__init__()
        self.root = None
        self.undirected = set()
    
    def __setitem__(self,key,value):
        """Allow initialising nodes by calling 'self[nodeid] = lemma'"""
        assert isinstance(key,int)
        if isinstance(value,str):
            super().__setitem__(key,Node(key,value))
        else:
            assert value.nodeid == key
            super().__setitem__(key,value)
    
    def __repr__(self):
        """Display all nodes with their links"""
        string = "ROOT: {}\n".format(self.root)
        string += "roots: "+", ".join(str(x) for x in self.roots())+'\n'
        for node in self:
            string += str(node)
            if node.incoming:
                string += " ({})\n".format(" ".join(str(x[1].nodeid) for x in node.incoming))
            else:
                string += " (root)\n"
            for label, child in sorted(node.outgoing, key=lambda x:x[1].nodeid):
                string += "  {} {}".format(child.nodeid, label) + '\n'
        string += "undirected:\n"
        for pair in self.undirected:
            string += "  {}\n".format(pair)
        return string
    
    def __iter__(self):
        """Iterate through nodes ordered by their IDs"""
        for x in sorted(self.keys()):
            yield self[x]
    
    def nodes(self):
        """Return nodes as a set"""
        return set(self.values())
    
    def remove(self,node,verbose=False):
        """Remove a node from the graph"""
        for label,parent in node.incoming:
            parent.outgoing.remove((label,node))
        for label,child in node.outgoing:
            child.incoming.remove((label,node))
        self.pop(node.nodeid)
        for x,y in copy(self.undirected):
            if x == node or y == node:
                self.undirected.remove((x,y))
        if self.root == node:
            self.root = None
        if verbose: print('removed',node)
    
    def remove_by_id(self,nodeid,verbose=False):
        """Remove a node from the graph, using the node ID"""
        self.remove(self[nodeid],verbose=verbose)
    
    def leaves(self):
        """Iterate through leaves of the graph"""
        for node in self:
            if not node.outgoing:
                yield node
    
    def roots(self):
        """Iterate through roots of the graph"""
        for node in self:
            if not node.incoming:
                yield node
    
    def rooted(self, safe=False):
        """Return true if there is a unique root"""
        if safe and not self.connected():
            return False
        return len(list(self.roots())) == 1
    
    def cycle(self):
        """
        Check for cycles.
        If there is a cycle, return the nodes in the largest subgraph with no roots or leaves
        """
        # Iteratively remove all leaves from the graph
        discard = set(self.leaves())
        parents = set()
        for node in discard:
            parents.update(node.parents())
        n = True
        while n: # Keep removing leaves until we can't remove any more
            n = 0 # Count how many leaves we can remove in this pass
            new_parents = set()
            for mother in parents:
                if mother.children() - discard:
                    new_parents.add(mother)
                else:
                    n += 1
                    discard.add(mother)
                    new_parents.update(mother.parents())
            parents = new_parents
        if not parents: # If we removed everything, there are no cycles 
            return False
        else: # If there is a cycle, do the same with roots
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
            return self.nodes() - discard - discard_top
    
    def connected(self, avoid=set()):
        """
        Check if the graph is connected.
        Optionally, specify nodes to be avoided,
        i.e. check if the graph is connected with those nodes removed
        """
        if len(self) <= 1:
            return True
        cover = set()
        queue = {(self.nodes() - avoid).pop()} # Take some element not being avoided
        while queue:
            new = queue.pop()
            cover.add(new)
            for adjacent in new.parents() | new.children():
                if not (adjacent in cover or adjacent in avoid):
                    queue.add(adjacent)
        if len(cover) == len(self) - len(avoid):
            return True
        else:
            return False
    
    def connected_pair(self, first, second):
        """
        Check if a pair of nodes are connected to each other
        """
        cover = set()
        queue = {first}
        while queue:
            new = queue.pop()
            cover.add(new)
            for adjacent in new.parents() | new.children():
                if adjacent == second:
                    return True
                elif not adjacent in cover:
                    queue.add(adjacent)
        return False
    
    def components(self):
        """
        Find out how many connected components are in the graph
        """
        comps = 0
        unexplored = self.nodes()
        while unexplored:
            comps += 1
            queue = {unexplored.pop()}
            while queue:
                new = queue.pop()
                unexplored.remove(new)
                for adjacent in new.parents() | new.children():
                    if adjacent in unexplored:
                        queue.add(adjacent)
        return comps
    
    def bottom_up(self, safe=False):
        """
        Iterate through the graph bottom up.
        Nodes are only returned once all their children have been.
        If the graph may have cycles, use the 'safe' parameter
        """
        if safe:
            assert not self.cycle()
        discard = set()
        queue = deque(self.leaves())
        while queue:
            new = queue.popleft()
            if new.children() - discard:
                queue.append(new)
            else:
                discard.add(new)
                for parent in sorted(new.parents(), key=lambda x:x.nodeid):
                    if not parent in discard and not parent in queue:
                        queue.append(parent)
                yield new
