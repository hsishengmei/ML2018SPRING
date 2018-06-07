import itertools

def parse_label(filename='data/training_label.txt'):
    x = []
    y = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            label = int(line[0])
            sentence = line[10:]
            y.append(label)
            x.append(sentence)
    return x, y       

def parse_no_label(filename='data/training_nolabel.txt'):
    x = []
    with open(filename, 'r', encoding='utf-8') as f:
        x = f.readlines()
    return x

def parse_test(filename='data/testing_data.txt'):
    x = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines()[1:]:
            comma = line.find(',')
            x.append(line[comma+1:])
    return x

def trim_list(text_list, threshold=2):
    return [trim(text, threshold) for text in text_list]

def trim(text, threshold=2):
    groups = []
    for _, g in itertools.groupby(text):
        groups.append(list(g))
    result = ''
    for g in groups:
        count = len(g) if len(g) < threshold else threshold
        result += g[0] * count
    return result

'''
def trim(text, threshold=2):
    groups = []
    for _, g in itertools.groupby(text):
        groups.append(list(g))
    result = ''.join([g[0] for g in groups])
    return result 
'''

def semi(data_un, model, tokenizer, threshold = 0.2):
    res = model.predict(data_un).flatten()
    data_new = []
    label_new = []
    for seq, label in zip(data_un, res):
        if (label > 1 - threshold) or (label < threshold):
            data_new.append(seq)
            label_new.append(round(label))
    return data_new, label_new