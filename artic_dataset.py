from dataset import *

class Artic_dataset(Dataset):
    def __init__(self, lable_path, train=True):
        super(Artic_dataset, self).__init__()
        self.train = train

        numitr = 0
        self.classnums = {}

        truth = {}
        f = open(lable_path, 'r', encoding='utf-8')
        for line in f.readlines():
            data = [obj for obj in line.strip().split(' ') if obj != '']
            truth[data[0]] = []
            for obj in data[1:]:
                obj = obj.split(',')
                type = obj[-1]
                if type not in self.classnums:
                    self.classnums[type] = numitr
                    numitr += 1
                type = self.classnums[type]
                truth[data[0]].append( [type]+obj[:-1] )

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
        print(img_path)
        bboxes_with_cls_id = np.array(
          self.truth.get(img_path), dtype=np.float
        )
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)self.truth.get(img_path)
        img = np.load(img_path, allow_pickle=True, encoding='bytes').item()
        img = img['rgb']

        num_objs = len(bboxes_with_cls_id)
        target = {}

        # boxes to coco format
        import pdb; pdb.set_trace()
        boxes = bboxes_with_cls_id[...,1:]
        boxes[..., 2:] = boxes[..., 2:] - boxes[..., :2]  # box width, box height
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor(bboxes_with_cls_id[...,-1].flatten(), dtype=torch.int64)
        target['image_id'] = torch.tensor([get_image_id(img_path)])
        target['area'] = (target['boxes'][:,3])*(target['boxes'][:,2])
        target['iscrowd'] = torch.zeros((num_objs,), dtype=torch.int64)
        return img, target

    def collate(self, batch):
        if not self.train:
            return tuple(zip(*batch))

        images = []
        bboxes = []
        for img, box in batch:
            images.append([img])
            bboxes.append([box])
        images = np.concatenate(images, axis=0)
        images = images.transpose(0, 3, 1, 2)
        images = torch.from_numpy(images).div(255.0)
        bboxes = np.concatenate(bboxes, axis=0)
        bboxes = torch.from_numpy(bboxes)
        return images, bboxes


if __name__ == '__main__':
    train_dataset = Artic_dataset('train.txt', train=True)
    inst = train_dataset[0]
    print(inst)
