# Take the output of the ERG and split between the parsed and unparsed sentences

with open("../data/sentibank3.mrs", 'w') as fgood:
    with open("../data/sentibank3.skip", 'w') as fskip:
        with open("../data/sentibank3.out", 'r') as f:
            for line in f:
                if line[0] == '[':
                    fgood.write(line)
                    #fskip.write('\n')
                elif line[0] == 'S':
                    fskip.write(line[6:])
                    fgood.write('\n')