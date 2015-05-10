from token_align import align
import xml.etree.ElementTree as ET

with open('../../Sentimantics/data/sentibank123.dmrs', 'r') as fdmrs, \
     open('../../Sentimantics/data/sentibank.txt', 'r') as fraw, \
     open('../../Sentimantics/data/stanfordSentimentTreebank/datasetSentences.txt', 'r') as ftoken, \
     open('../../Sentimantics/data/sentibank123_align.txt', 'w') as fout:
    ftoken.readline()
    for dmrs in fdmrs:
        if not dmrs.strip():
            fraw.readline()
            ftoken.readline()
            fout.write('\n')
            continue
        parser = ET.XMLParser(encoding='utf-8')
        xml, = ET.fromstring(dmrs, parser=parser)
        raw = fraw.readline().strip()
        token = ftoken.readline().split('\t', 1)[1].split()
        aligned = align(xml, raw, token)
        id_span = [()]
        for x in aligned:
            if x.tag == 'node':
                nodeid = x.attrib['nodeid']
                token_list = x.attrib['tokalign'].split()
                first = token_list[0]
                last = token_list[-1]
                fout.write(":".join([nodeid, first, last]))
                fout.write(" ")
        fout.write("\n")