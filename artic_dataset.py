import os.path as osp
from dataset import *

class Artic_dataset(Dataset):
    def __init__(self, label_path, config, train=True):
        super(Artic_dataset, self).__init__()
        self.train = train
        self.root_path = '/'.join(label_path.split('/')[:-1])

        numitr = 0
        self.classnums = {}
        self.w, self.h = config.width, config.height

        truth = {}
        f = open(label_path, 'r', encoding='utf-8')
        for line in f.readlines():
            data = [obj for obj in line.strip().split(' ') if obj != '']
            truth[data[0]] = []
            for obj in data[1:]:
                obj = obj.split(',')
                type = obj[0]
                if type not in self.classnums:
                    self.classnums[type] = numitr
                    numitr += 1
                type = self.classnums[type]
                truth[data[0]].append( obj[1:]+[type] )

        self.truth = truth
        self.imgs = list(self.truth.keys())

    def __len__(self):
        return len(self.truth.keys())

    def __getitem__(self, index):
        return self._get_val_item(index)

    def _get_val_item(self, index):
        """
        """
        img_path = self.imgs[index]
        # print(img_path)
        boxes = np.array( self.truth.get(img_path), dtype=np.float )
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)self.truth.get(img_path)
        img = np.load( osp.join(self.root_path, img_path),
          allow_pickle=True, encoding='bytes' ).item()
        img = cv2.resize(img['rgb'], (self.w, self.h), cv2.INTER_LINEAR)

        # boxes to coco format
        boxes[..., 2:-1:2] = boxes[..., 2:-1:2] - boxes[..., 0]
        boxes[..., 3:-1:2] = boxes[..., 3:-1:2] - boxes[..., 1]

        return img, boxes

        # target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        # target['labels'] = torch.as_tensor(bboxes_with_cls_id[...,-1].flatten(), dtype=torch.int64)
        # # target['image_id'] = torch.tensor([get_image_id(img_path)])
        # target['area'] = (target['boxes'][:,3])*(target['boxes'][:,2])
        # target['iscrowd'] = torch.zeros((num_objs,), dtype=torch.int64)
        # return img, target

    def collate(self, batch):
        if not self.train:
            return tuple(zip(*batch))

        # print('batch:', len(batch))
        images = []
        bboxes = []
        for img, box in batch:
            # print(type(img))
            images.append([img])
            bboxes.append([box])
        images = np.concatenate(images, axis=0)
        images = images.transpose(0, 3, 1, 2)
        images = torch.from_numpy(images).div(255.0)
        bboxes = np.concatenate(bboxes, axis=0)
        bboxes = torch.from_numpy(bboxes)
        return images, bboxes


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = Artic_dataset('data/artic_train.txt', train=True)
    inst = train_dataset[0]
    # print(inst)

    print('data_loader:')
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset,
      batch_size=3, shuffle=True,
      num_workers=1, pin_memory=True, drop_last=True,
      collate_fn=train_dataset.collate
    )

    inst = train_loader
    for ii, batch in enumerate(train_loader):
        images = batch[0]
        bboxes = batch[1]
        bboxes = bboxes.to(device=device)
        nlabel = (bboxes.sum(dim=2) > 0).sum(dim=1)
        print(bboxes)
        import pdb; pdb.set_trace()
        if ii == 0:
            break
