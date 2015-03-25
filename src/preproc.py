# This script gives the raw text in "original_rt_snippets.txt" with the same sentence segmentation as "datasetSentences.txt".
# It is very hacky and not supposed to be for long term use.

ignore = ['464', '705', '938', '1214', '1494', '1667', '1892', '2084', '2610', '2946', '3109', '3361', '3417', '4309', '4714', '4901', '5323', '5597', '5607', '6372', '7271', '7755', '7996', '8210', '8906', '8918', '9166', '9259', '9731', '9831', '9920', '10083', '10667', '10863', '10939', '11572']

with open("../data/sentibank.txt", 'w') as fout:
    with open("../data/stanfordSentimentTreebank/datasetSentences.txt", 'r') as fref:
        with open("../data/stanfordSentimentTreebank/original_rt_snippets.txt", 'r') as fin:
            n = 0
            k = 0
            fref.readline()
            nextline = fref.readline()
            for line in fin:
                k += 1
                jump = True
                start = 0
                while jump:
                    jump = False
                    currline = nextline
                    nextline = fref.readline().split()
                    if not nextline:
                        nextline = [None,'.','.']
                    if (currline[-1] + ' ' + nextline[1] + ' ' + nextline[2] in line
                     or currline[-1] + ' ' + nextline[1] + ''  + nextline[2] in line): #Changed "No." to "No" in file
                        if not nextline[0] in ignore: 
                            jump = True
                            n += 1
                            #print(nextline[0], k)
                    if not jump:
                        for i in range(len(nextline)-3):
                            triple = ' '.join(nextline[i+1:i+4])
                            triple2 = nextline[i+1]+' '+nextline[i+2]+nextline[i+3]
                            triple3 = ''.join(nextline[i+1:i+4])
                            if ((triple in line
                             and triple != 'of the year'
                             and triple != 'but it is')
                             or triple2 in line
                             or triple3 in line):
                                if not nextline[0] in ignore:
                                    jump = True
                                    n += 1
                                    #print(nextline[0], k)
                                    break
                    # If we've found the next line, chop off the previous part
                    
                    if jump:
                        end = start+1
                        found = False
                        ending = currline[-2] + ' ' + currline[-1]
                        end_2  = currline[-2] + '' + currline[-1]
                        beginning = nextline[1] + ' ' + nextline[2]
                        beg_2     = nextline[1] + ''  + nextline[2]
                        while not found:
                            if end == len(line):
                                print(line)
                                print(currline)
                                print(nextline)
                                raise Exception
                            end += 1 
                            if (line[end:end+len(beginning)] == beginning
                             or line[end:end+len(beg_2)] == beg_2
                             or line[end-len(ending):end] == ending
                             or line[end-len(end_2):end] == end_2):
                                found = True
                    # If we haven't, write the rest
                    else:
                        end = len(line)
                    fout.write(line[start:end].strip()+'\n')
                    start = end
                    
                        
            print()
            print(n)