with open("../data/gaps1.mrs", 'r') as fold:
    with open("../data/gaps2.mrs", 'r') as fnew:
        with open("../data/gaps12.mrs", 'w') as f:
            for line in fold:
                if line == '\n':
                    f.write(fnew.readline())
                else:
                    f.write(line)