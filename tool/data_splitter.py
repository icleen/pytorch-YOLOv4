import os, sys, json, random
import os.path as osp
import numpy as np

"""
This script expects to be run from the directory of the saved data
(most likely something akin to ~/data/asus_video/).  When you run it
the splits will be generated and you will have three text files which designate
the data and labels in the format expected by Artic_dataset.
"""

def format_pts(points):
    pts = np.array(points)[1:]
    dists = np.sqrt(np.power(pts, 2).sum(1))
    pts = pts[np.argsort(dists)]
    pts = points[0] + pts.flatten().tolist()
    return [str(pt) for pt in pts]

folders = [f for f in os.listdir('.') if '.' not in f and f != 'gifs']

totals = []
classes = {}
for folder in folders:
    labels = []
    for root, dirs, files in os.walk(folder):
        if 'labels' in root:
            labels += [(fil, root) for fil in files if '.json' in fil and fil != 'curimg.json']
    classes[folder] = labels
    totals += labels

with open('files_classes.json', 'w') as f:
    json.dump(classes, f)

lines = []
for (fil, root) in totals:
    label = osp.join(root, fil)
    filpy = fil.replace('.json', '.npy')
    line = osp.join('/'.join(root.split('/')[:-1]), filpy)

    with open(label, 'r') as f:
        labels = json.load(f)

    save = True
    for label in labels:
        cls = label['type']
        pts = label['points']
        if len(pts) < 5:
            save = False
            continue
        line = line + ' ' + cls + ',' + ','.join(format_pts(pts))

    if save:
        lines.append(line + '\n')

train_perc = int(0.7 * len(lines))
val_perc = int(0.15 * len(lines))
random.shuffle(lines)
train = lines[:train_perc]
valid = lines[train_perc:(train_perc+val_perc)]
test = lines[(train_perc+val_perc):]

with open('artic_train.txt', 'w') as f:
    for line in train:
        f.write(line)

with open('artic_valid.txt', 'w') as f:
    for line in train:
        f.write(line)

with open('artic_test.txt', 'w') as f:
    for line in train:
        f.write(line)
