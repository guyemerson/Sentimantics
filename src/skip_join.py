# Take a file with missing parses, and fill in parses where possible

with open("../data/sentibank12.mrs", 'r') as fold:
    with open("../data/sentibank3.mrs", 'r') as fnew:
        with open("../data/sentibank123.mrs", 'w') as f:
            for line in fold:
                if line == '\n':
                    f.write(fnew.readline())
                else:
                    f.write(line)