from delphin.mrs import simplemrs, dmrx
import sys

mrs_file = '../data/sentibank123.mrs'
dmrs_file = '../data/sentibank123.dmrs'

try:
    with open(dmrs_file, 'r') as f:
        N = 0
        for line in f:
            N += 1
except FileNotFoundError:
    pass

with open(mrs_file, 'r') as fin, \
     open(dmrs_file, 'a') as fout:
    for n, line in enumerate(fin):
        if n < N:
            continue
        if line.strip():
            mrs = simplemrs.loads_one(line)
            try:
                dmrs = dmrx.dumps_one(mrs)
            except KeyError as e:
                print(n, e, file=sys.stderr)
            else:
                fout.write(dmrs)
        fout.write('\n')