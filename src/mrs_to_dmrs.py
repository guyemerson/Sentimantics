from delphin.mrs import simplemrs, dmrx

with open('../data/sentibank123.mrs', 'r') as fin, \
     open('../data/sentibank123.dmrs', 'w') as fout:
    for line in fin:
        if line.strip():
            mrs = simplemrs.loads_one(line)
            dmrs = dmrx.dumps_one(mrs)
            fout.write(dmrs)
        fout.write('\n')