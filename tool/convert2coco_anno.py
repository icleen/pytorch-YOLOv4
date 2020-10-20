import os,sys,json
import os.path as osp
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
      "indir", nargs='?', type=str,
      default="/home/ilee141/asus_video/wrench/tatami/window_light/angle1/labels/",
      help="path to input directory"
    )
    parser.add_argument(
      "-o", "--outpath", type=str,
      default=None,
      help="path to output directory"
    )
    parser.add_argument(
      "-r", "--reset", nargs='?', const=True, default=False,
      help="whether to overwrite any currently saved data in the designated folder"
    )
    config = parser.parse_args()

    labdir = config.indir
    imgdir = config.indir.split('/')
    imgdir = imgdir[:-2] if imgdir[-1] == '' else imgdir[:-1]
    imgdir = '/'.join(imgdir)
    labels = [f for f in os.listdir(config.indir)
      if '.json' in f and ('image_' in f or 'video_' in f)]
    labels.sort()

    if config.outpath is None:
        config.outpath = osp.join(imgdir, 'labels.txt')

    with open(config.outpath, 'w') as f:
        for inst in labels:
            imgpath = osp.join(imgdir, inst.replace('.json','.npy'))
            f.write(imgpath + ' ')

            label = json.load( open(osp.join(labdir, inst)) )
            for obj in label:
                for pt in obj['points']:
                    f.write(str(pt[0]) + ',' + str(pt[1]) + ',')
                f.write(obj['type'] + ' ')
            f.write('\n')

if __name__ == '__main__':
    main()
