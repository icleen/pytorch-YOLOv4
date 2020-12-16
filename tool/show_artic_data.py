import os, sys, json, random
import os.path as osp
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_pts(imgpath, objs, resize=False):
    color1 = (0,0.25,0.75)
    color2 = (0,0.75,0.25)
    oheight, owidth = 0, 0
    height, width = 480, 480
    img = np.load( imgpath,
      allow_pickle=True, encoding='bytes' ).item()
    try:
        img = img['rgb']
    except Exception as e:
        img = img[b'rgb']
    oheight, owidth = img.shape[0], img.shape[1]
    if resize:
        img = cv2.resize(img, (width, height), cv2.INTER_LINEAR)
    fig = plt.figure()
    ax_im = plt.subplot()
    ax_im.imshow(img,cmap='gray')
    for obj in objs:
        pts = np.array(objs['points'])
        if resize:
            pts[:,0] *= width / owidth
            pts[:,1] *= height / oheight
        pt0 = pts[-1]
        for pi, pt in enumerate(pts[:-1]):
            line = patches.Polygon(
              np.array([[pt0[0], pt0[1]], [pt[0], pt[1]]]),
              closed=False, linewidth=2,
              edgecolor=color2 if pi < 2 else color2
            )
            ax_im.add_patch( line )
    plt.show()

def show_from_labels():
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

    for (fil, root) in totals:
        label = osp.join(root_dir, root, fil)
        npyfil = osp.join( '/'.join(root.split('/')[:-1]),
          fil.replace('.json', '.npy') )

        with open(label, 'r') as f:
            labels = json.load(f)

        for label in labels:
            cls = label['type']
            pts = label['points']
            if len(pts) < 5:
                save = False
                continue
            plot_pts(osp.join(root_dir, npyfil), label, True)
            return

def show_from_train():

    root_dir = '/home/iain/data/asus_video'

    with open(osp.join(root_dir, 'artic_train.txt'), 'r') as f:
        data = [line for line in f]

    parsed = []
    for line in data:
        info = {}
        line = line.split(' ')
        info['img'] = osp.join(root_dir, line[0])
        info['objs'] = []
        for obj in line[1:]:
            iobj = {}
            obj = obj.split(',')
            iobj['cls'] = obj[0]
            pts = np.array([float(pt) for pt in obj[1:]]).reshape(-1, 2)
            iobj['points'] = pts
            info['objs'].append( iobj )
        parsed.append( info )

    color1 = (0,0.25,0.75)
    color2 = (0,0.75,0.25)
    oheight, owidth = 0, 0
    height, width = 480, 480
    for inst in parsed:
        plot_pts(inst['img'], inst['objs'], True)

def main():

    show_from_labels()

    show_from_train()


if __name__ == '__main__':
    main()
