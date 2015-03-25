label = {}
with open("../data/stanfordSentimentTreebank/dictionary.txt", 'r') as flabels:
    for line in flabels:
        key, value = line.strip().split('|')
        label[key] = value
with open("../data/stanfordSentimentTreebank/datasetSentences.txt", 'r') as fsents:
    with open("../data/sent_sentiment.txt", 'w') as f:
        fsents.readline()
        for line in fsents:
            line = line.strip().partition('\t')[2]
            f.write(label[line]+'\n')