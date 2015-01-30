class Node():
    def __init__(self, lemma):
        self.lemma = lemma
        self.incoming = set()
        self.outgoing = set()
    def __repr__(self):
        return self.lemma

YES = ['abstr_deg_rel',
    'all+too_rel',
    'appos_rel',
    'basic_card_rel',
    'card_rel',
    'compound_rel',
    'compound_name_rel',
    'comp_enough_rel',
    'comp_less_rel',
    'comp_not+so_rel',
    'comp_not+too_rel',
    'comp_rel',
    'comp_so_rel',
    'comp_too_rel',
    'discourse_rel',
    'dofm_rel',
    'dofw_rel',
    'every_q_rel',
    'excl_rel',
    'fraction_rel',
    'free_relative_ever_q_rel',
    'free_relative_q_rel',
    'fw_seq_rel',
    'generic_entity_rel',
    'greet_rel',
    'holiday_rel',
    'implicit_conj_rel',
    'little-few_a_rel',
    'loc_nonsp_rel',
    'manner_rel',
    'measure_rel',
    'meas_np_rel',
    'minute_rel',
    'mofy_rel',
    'much-many_a_rel',
    'named_n_rel',
    'named_rel',
    'neg_rel',
    'ne_x_rel',
    'numbered_hour_rel',
    'num_seq_rel',
    'of_p_rel',
    'ord_rel',
    'parenthetical_rel',
    'part_of_rel',
    'person_rel',
    'plus_rel',
    'polite_rel',
    'poss_rel',
    'pron_rel',
    'reason_rel',
    'recip_pro_rel',
    'refl_mod_rel',
    'season_rel',
    'year_range_rel',
    'yofc_rel',
    'some_q_rel',
    'subord_rel',
    'superl_rel',
    'temp_loc_x_rel',
    'temp_rel',
    'thing_rel',
    'times_rel',
    'time_n_rel',
    'unspec_manner_rel',
    'which_q_rel',
    'with_p_rel'
    'year_range_rel',
    'yofc_rel',]

NO = ['addressee_rel',
    'approx_grad_rel',
    'comp_equal_rel',
    'cop_id_rel',
    'def_explicit_q_rel',
    'def_implicit_q_rel',
    'ellipsis_expl_rel',
    'ellipsis_ref_rel',
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
    'nominalization_rel',
    'number_q_rel',
    'parg_d_rel',
    'place_n_rel',
    'prednom_state_rel',
    'pronoun_q_rel',
    'property_rel',
    'proper_q_rel',
    'prpstn_to_prop_rel',
    'relative_mod_rel',
    'string',
    'timezone_p_rel',
    'udef_q_rel',
    'unknown_rel',
    'unspec_adj_rel',
    'v_event_rel']

SKIP = [] #'unknown_rel', 'fw_seq_rel'

with open('../data/00101') as f:
    while True:
        graph = dict()
        ignore = list()
        while True:
            try:
                line = next(f)
            except StopIteration:
                break
            if line[0] != '<' or line[0:5] == '<dmrs':
                ident = line[-16:-3]
                continue
            if line[0:7] == '</dmrs>':
                break
            tags = line.split('><')
            first = tags[0].split()
            
            if first[0] == '<node':
                nodeid = first[1].split("'")[1]
                if tags[1][0:5] == 'gpred':
                    lemma = tags[1][6:-7]
                    if lemma in SKIP:
                        ignore.append(nodeid)
                        continue
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
                
                if from_node in ignore or to_node in ignore:
                    continue
                if rargname == 'NIL':
                    continue
                
                if ((rargname[:3] == 'ARG' and post == 'EQ') # Make modifiers directional
                  #or post == 'HEQ' ## Do not reverse nominalization, perhaps we could later cut out the extra node
                  or label == 'RSTR_H'  # Reverse quantifiers
                  #or label == 'ARG1_H'  # Make subordination directional
                  #or (label == 'NIL_EQ' and graph[from_node].lemma in ['appos_rel', 'parenthetical_rel'])
                  or (label == 'ARG1_NEQ' and graph[from_node].lemma in ['appos_rel', 'parenthetical_rel', 'compound_name_rel'])):
                    label += '_rev'
                    from_node, to_node = to_node, from_node
                graph[from_node].outgoing.add((label, graph[to_node]))
                graph[to_node].incoming.add((label, graph[from_node]))
        
        root = list()
        
        for idee, node in graph.items():
            if node.incoming:
                continue
            root.append((idee, node))
        
        if len(root) > 1:
            print(ident)
            
            # Check if connected
            cover = {idee:set() for idee, _ in root}
            for idee, node in root:
                queue = {x[1] for x in node.outgoing}
                while queue:
                    new = queue.pop()
                    cover[idee].add(new)
                    for _, child in new.outgoing:
                        if not (child in queue or child in cover[idee]):
                            queue.add(child)
            intersect = False
            for n, first in enumerate(root[:-1]):
                for second in root[n:]:
                    if cover[first[0]] & cover[second[0]]:
                        intersect = True
                        break
                if intersect: break
            if intersect:
                #print("Disconnected")
                continue
            
            for idee, node in root:
                print(idee, node.lemma, end=': ')
                for label, new_node in node.outgoing:
                    print(label, new_node.lemma, end=', ')
                print()
            print()
            #input()
        
        if not graph:
            break