from delphin.mrs import dmrx
from copy import copy
import pickle, os
from graph import Graph

# Lemmas that should be skipped
SKIP = ['approx_grad_rel',
        'cop_id_rel',
        'def_explicit_q_rel',
        'def_implicit_q_rel',
        'ellipsis_expl_rel',
        'ellipsis_ref_rel',  # This one might cause issues?
        'ellipsis_rel',
        'elliptical_n_rel',
        'eventuality_rel',
        'focus_d_rel',
        'generic_nom_rel',
        'generic_verb_rel',
        'hour_prep_rel',
        'idiom_q_i_rel',
        'id_rel',
        'interval_p_end_rel',
        'interval_p_start_rel',
        'interval_rel',
        'number_q_rel',
        'parg_d_rel',
        'pronoun_q_rel',
        'property_rel',
        'proper_q_rel',
        'prpstn_to_prop_rel',
        'string',
        'timezone_p_rel',
        'udef_q_rel',
        'unspec_adj_rel',
        'v_event_rel']

# Lemmas that should be removed if connectivity is kept
MAYBE = ['unknown_rel']

# Lemmas whose ARG1 should be reversed 
REVERSE_ARG1 = ['appos_rel',
                'parenthetical_rel',
                'compound_name_rel',
                'compound_rel',
                'of_p_rel',
                'but_p_except_rel',
                'poss_rel']

# Lemmas denoting coordination
def coord(rel):
    if rel == 'implicit_conj_rel':
        return True
    elif rel[-6:] == '_c_rel':
        return True
    else:
        return False

# Lemmas which should defer to other roots
def subord(rel):
    if rel in ['subord_rel', 'neg_rel']:
        return True
    elif coord(rel):
        return True
    else:
        return False

# Lemmas for adjectives
def adj(rel):
    if rel.split('_')[1] == 'a':
        return True
    elif len(rel) >= 17 and rel[-17:] in ['/JJ_u_unknown_rel', '/RB_u_unknown_rel']:
        return True
    else:
        return False

# Input and output files
dmrs_file = "../data/sentibank123.dmrs"
graph_file = "../data/sentibank123_graph.pk"
error_file = "../data/sentibank123.fail"

#Debug settings
debug = False
start_from = 1

if __name__ == "__main__":
    good = 0
    cycle = []
    multiroot = []
    disconnect = []
    
    # Work out how much of the data we've already processed
    N = 0
    try:
        with open(graph_file, 'rb') as f:
            try:
                while True:
                    pickle.load(f)
                    N += 1
            except EOFError:
                print(N, 'sentences already processed')
    except FileNotFoundError:
        pass
    
    with open(dmrs_file,'r') as fin, \
         open(graph_file,'ab') as fout, \
         open(error_file, 'a') as ferr:
        for line_num, line in enumerate(fin):
            if debug and line_num+1 < start_from: continue  # Start at a certain point when debugging
            if line_num < N: # Skip data we've already processed
                continue
            text = line.strip()
            if not text: # Ignore blank lines
                if not debug: pickle.dump(None,fout)
                continue
            try: # Load the data
                mrs = dmrx.loads_one(text)
            except KeyError:
                if not debug:
                    pickle.dump(None,fout)
                    ferr.write('{} KeyError\n'.format(line_num+1))
                continue
            
            ### Initialise the graph ###
            
            graph = Graph()  # The modified DMRS graph
            ignore = list()  # Nodes to be ignored
            maybe = list()   # Nodes that might be ignored
            # Set up all nodes in the graph, except those to be skipped
            for n in mrs.nodes:
                nodeid = n.nodeid
                pred = n.pred.string.strip('"_')
                if pred in SKIP:
                    if debug: print('skipped', nodeid, pred)
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
                if l.start == 0:  # 0 indicates a dummy node
                    graph.root = end
                    continue
                start = graph[l.start]
                if not l.argname:  # Undirected EQ links have no argname
                    assert l.post == 'EQ'
                    graph.undirected.add((start,end))
                    continue
                label = l.argname + '_' + l.post
                start.add_link((label,end))
            
            if debug: print(graph)
            
            ### Reverse links ###
            
            to_reverse = []
            for node in list(graph):
                # Reverse links for certain predicates
                if node.lemma in REVERSE_ARG1:
                    for link in copy(node.outgoing):
                        if link[0][:5] == 'ARG1_':
                            to_reverse.append((node, link))
                    continue
                # Reverse links for quantifiers
                if node.lemma != 'neg_rel': 
                    quant = False
                    for link in copy(node.outgoing):
                        if link[0] == 'RSTR_H':
                            to_reverse.append((node, link))
                            quant = True
                    if quant: continue
                # Reverse links for modifiers
                modlinks = [x for x in node.outgoing if x[0][:3] == 'ARG' and x[0][-3:] == '_EQ']
                if modlinks:
                    modlinks.sort(key=lambda x:x[0])
                    to_reverse.append((node, modlinks[-1]))
                    continue
                # Are there any other cases of _H or _HEQ that we should reverse?
                # We can't do this in general, e.g. for coordination
            for node, link in to_reverse:
                node.reverse(link, verbose=debug)
            
            ### Fix connectivity ###
            
            # Add undirected links if this fixes connectivity
            if not graph.connected():
                for x,y in graph.undirected:
                    if not graph.connected_pair(x,y):
                        if y in graph.roots():
                            x.add_link(("NIL_EQ",y),verbose=debug)
                        elif x in graph.roots():
                            y.add_link(("NIL_EQ",x),verbose=debug)
            
            # Remove singletons
            if not graph.connected():
                for node in graph.roots():
                    if not node.outgoing:
                        graph.remove(node,verbose=debug)
            # If nothing's left, skip this graph
            if not graph:
                if debug:
                    print('No nodes left')
                else:
                    pickle.dump(None,fout)
                    ferr.write('{} singletons\n'.format(line_num+1))
                continue
            
            # Check if maybes can be removed
            for node in maybe:  # Could either of two nodes could be removed but not both? This would end up removing the first one loaded
                if graph.connected(avoid={node}):
                    graph.remove(node,verbose=debug)
            
            # If it's still not connected, just join things together with an unknown_rel
            if not graph.connected():
                graph[0] = 'unknown_rel'
                if debug: print('added ROOT')
                for root in set(graph.roots()) - {graph[0]}:
                    graph[0].add_link(('ARG_NEQ',root),verbose=debug)
            
            ### Deal with extra roots ###
            
            # Remove coordination if they're ending up as roots
            if not graph.rooted():
                if [x for x in graph.roots() if coord(x.lemma)]:
                    for node in reversed(list(graph)):
                        # This allows recursion through stacks of coordinations,
                        # but also removes non-roots not even connected to a coordinating root...
                        if coord(node.lemma):
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
            #!# Directly delete if there's not both L- and R-?
            
            # Add undirected links if this removes extra roots
            # (First out of a special list, and then in general)
            # (If both are roots, put the earlier node on top)
            if not graph.rooted():
                for x,y in graph.undirected:
                    if not y.parents() and subord(y.lemma):
                        x.add_link(('NIL_EQ',y),verbose=debug)
                    elif not x.parents() and subord(x.lemma):
                        y.add_link(('NIL_EQ',x),verbose=debug)
            if not graph.rooted():
                for x,y in graph.undirected:
                    if not y.parents():
                        x.add_link(('NIL_EQ',y),verbose=debug)
                    elif not x.parents():
                        y.add_link(('NIL_EQ',x),verbose=debug)
            
            # Reverse lone arguments of roots
            if not graph.rooted():
                for node in graph.roots():
                    non_rev = [x for x in node.outgoing if x[0][-4:] != '_rev']
                    if len(non_rev) == 1 and non_rev[0][0][-1] == 'Q':
                        node.reverse(non_rev[0],verbose=debug)
            
            # Make up an undirected link for subord_rel
            if not graph.rooted():
                sub_roots = {x for x in graph.roots() if x.lemma == 'subord_rel'}
                top_roots = set(graph.roots()) - sub_roots
                for node in top_roots:
                    for sub in sub_roots:
                        node.add_link(('NIL_EQ',sub))
            
            if debug: print(graph)
            
            # 85: reverse ARG1_NEQ for a root with a single outgoing link?
            # 99: single outgoing non-reversed link?
            
            ### Fix cycles ###
            
            # For cycle nodes with an incoming link from outside, reverse incoming cyclic links
            # (this might unstable for complicated cycles?)
            for _ in range(3): #!# Make this a little more efficient...
                cycle_nodes = graph.cycle()
                if cycle_nodes:
                    if debug: print(cycle_nodes)
                    for node in cycle_nodes:
                        for parent in node.parents():
                            if not parent in cycle_nodes:
                                for label, comrade in copy(node.incoming):
                                    if comrade in cycle_nodes and comrade != parent:
                                        comrade.reverse((label,node),verbose=debug)
            
            # If there are still cycles, don't save the graph
            cycle_nodes = graph.cycle()
            if cycle_nodes:
                print(line_num+1, "cycle", cycle_nodes)
                cycle.append(line_num+1)
                if not debug:
                    pickle.dump(None,fout)
                    ferr.write('{} cycle {}\n'.format(line_num+1, cycle_nodes))
            
            # If there are multiple roots, don't save the graph
            elif not graph.rooted():
                print(line_num+1, "multiple roots", end='')
                if not debug:
                    pickle.dump(None,fout)
                    ferr.write('{} multiroot {}\n'.format(line_num+1, graph.roots()))
            
                if graph.connected():
                    print(":")
                    multiroot.append(line_num+1)
                else: # This shouldn't happen, since we add unknown_rels as necessary
                    print(" (disconnected):")
                    disconnect.append(line_num+1)
                
                for node in graph.roots():
                    print(' ', node.nodeid, node.lemma, end=': ')
                    for label, new_node in node.outgoing:
                        print(label, new_node.nodeid, end=', ')
                    print()
            
            # If it's rooted and acyclic, save!
            else:
                print(line_num+1, "good!")
                '''if line_num+1 == 48:
                    print(graph)
                    break'''
                good += 1
                if not debug:pickle.dump(graph,fout)
                fout.flush()
                os.fsync(fout.fileno())
                continue
            
            # When debugging, stop when something goes wrong!
            if debug:
                print(graph)
                _ = input('')
    
    
    print(cycle)
    print(multiroot)
    print(disconnect)
    
    print(good)
    print(len(cycle))
    print(len(multiroot))
    print(len(disconnect))

