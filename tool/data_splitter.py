import os, sys, json, random
import os.path as osp
import numpy as np
from collections import defaultdict
from sklearn import cluster

"""
This script expects to be run from the directory of the saved data
(most likely something akin to ~/data/asus_video/).  When you run it
the splits will be generated and you will have three text files which designate
the data and labels in the format expected by Artic_dataset.
"""


root_dir = '/home/iain/data/asus_video'
folders = [osp.join(root_dir, f) for f in os.listdir(root_dir) if '.' not in f and f != 'gifs']

totals = []
classes = {}
for folder in folders:
    labels = []
    for root, dirs, files in os.walk(folder):
        if 'labels' in root:
            labels += [(fil, root.replace(root_dir+'/', '')) for fil in files
              if '.json' in fil and fil != 'curimg.json']
    classes[folder] = labels
    totals += labels

with open('files_classes.json', 'w') as f:
    json.dump(classes, f)

averages = defaultdict(lambda: [])
lines = []
for (fil, root) in totals:
    label = osp.join(root_dir, root, fil)
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
        all_pts = np.array(pts)
        temp = all_pts[1:]
        dists = np.sqrt(np.power(temp, 2).sum(1))
        temp = temp[np.argsort(dists)]
        sort_pts = pts[0] + temp.flatten().tolist()
        temp = temp - all_pts[0]
        averages[cls].append(temp)
        line = line + ' ' + cls + ',' + ','.join([str(pt) for pt in sort_pts])

    if save:
        lines.append(line + '\n')

n_clusters = 9
for cls in averages:
    anchors = [[] for n in range(n_clusters)]
    cavg = np.array(averages[cls])
    for pi in range(4):
        temp = cavg[:,pi]
        kmeans = cluster.KMeans(n_clusters=n_clusters).fit_predict(temp)
        anchors = [anchors[n] + temp[kmeans==n].mean(0).tolist()
          for n in range(n_clusters)]
    print('clusters:', anchors)

    cavg = cavg.reshape(-1, 8)
    print( '{} class medn: {}'.format(cls, np.median(cavg, 0)) )
    print( '{} class mean: {}'.format(cls, np.mean(cavg, 0))   )
    print( '{} class stdv: {}'.format(cls, np.std(cavg, 0))    )

train_perc = int(0.7 * len(lines))
val_perc = int(0.15 * len(lines))
random.shuffle(lines)
train = lines[:train_perc]
valid = lines[train_perc:(train_perc+val_perc)]
test = lines[(train_perc+val_perc):]

with open(osp.join(root_dir, 'artic_train.txt'), 'w') as f:
    for line in train:
        f.write(line)

with open(osp.join(root_dir, 'artic_valid.txt'), 'w') as f:
    for line in train:
        f.write(line)

with open(osp.join(root_dir, 'artic_test.txt'), 'w') as f:
    for line in train:
        f.write(line)
