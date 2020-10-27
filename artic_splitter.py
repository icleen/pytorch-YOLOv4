import os, sys
from random import shuffle

def edit(line):
    line = line.strip().split(' ')
    nline = [line[0]]
    for inst in line[1:]:
        pts = inst.split(',')
        if len(pts) < 11:
            return None
        pts = pts[-1:] + pts[-3:-1] + pts[:-3]
        inst = ','.join(pts)
        nline.append(inst)
    return ' '.join(nline) + '\n'

if __name__ == '__main__':
    with open('data/artic_labels.txt', 'r') as f:
        labels = [line for line in f]

    labels = [edit(inst) for inst in labels]
    labels = [inst for inst in labels if inst is not None]

    shuffle(labels)
    split = int(0.1 * len(labels))
    train = labels[split:]
    valid = labels[:split]

    with open('data/artic_train.txt', 'w') as f:
        for inst in train:
            f.write(inst)

    with open('data/artic_valid.txt', 'w') as f:
        for inst in valid:
            f.write(inst)
