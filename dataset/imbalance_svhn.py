import torchvision
import torchvision.transforms as transforms
import os
import pickle
import scipy.io as sio
import numpy as np


class ImbalanceSVHN(torchvision.datasets.SVHN):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, split='train',
                 transform=None, target_transform=None, download=False):
        super(ImbalanceSVHN, self).__init__(root, split, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        # img_max = len(self.data) / cls_num
        img_max = 1000
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.labels, dtype=np.int64)
        classes = np.unique(targets_np)
        # shift label 0 to the last (as original SVHN labels)
        # since SVHN itself is long-tailed, label 10 (0 here) may not contain enough images
        classes = np.concatenate([classes[1:], classes[:1]], axis=0)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            print(f"Class {the_class}:\t{len(idx)}")
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.labels = new_targets
        assert new_data.shape[0] == len(new_targets), 'Length of data & labels do not match!'

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class SemiSupervisedImbalanceSVHN(torchvision.datasets.SVHN):
    cls_num = 10
    unlabel_size_factor = 5

    def __init__(self, root, imb_type='exp', imb_factor=0.01, unlabel_imb_factor=1, rand_number=0, split='train',
                 transform=None, target_transform=None, download=False):
        super(SemiSupervisedImbalanceSVHN, self).__init__(root, split, transform, target_transform, download)
        # unlabeled
        self.unlabeled_pseudo = os.path.join(root, 'pseudo_labeled_svhn.pickle')  # pseudo-labels using model trained on imbalanced data
        self.imb_factor = imb_factor
        self.unlabel_imb_factor = unlabel_imb_factor
        self.num_per_cls_dict = dict()
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        img_num_list_unlabeled = self.get_img_num_per_cls_unlabeled(self.cls_num, img_num_list, unlabel_imb_factor)
        self.gen_imbalanced_data(img_num_list, img_num_list_unlabeled)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        # img_max = len(self.data) / cls_num
        img_max = 1000
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def get_img_num_per_cls_unlabeled(self, cls_num, labeled_img_num_list, imb_factor):
        img_unlabeled_total = np.sum(labeled_img_num_list) * self.unlabel_size_factor
        img_first_min = img_unlabeled_total // cls_num
        img_num_per_cls_unlabel = []
        for cls_idx in range(cls_num):
            num = img_first_min * (imb_factor**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls_unlabel.append(int(num))
        factor = img_unlabeled_total / np.sum(img_num_per_cls_unlabel)
        img_num_per_cls_unlabel = [int(num * factor) for num in img_num_per_cls_unlabel]
        print(f"Unlabeled est total:\t{img_unlabeled_total}\n"
              f"After processing:\t{np.sum(img_num_per_cls_unlabel)},\t{img_num_per_cls_unlabel}")
        return img_num_per_cls_unlabel

    def gen_imbalanced_data(self, img_num_per_cls, img_num_per_cls_unlabeled):
        new_data = []
        new_targets = []
        targets_np = np.array(self.labels, dtype=np.int64)
        classes = np.unique(targets_np)
        # shift label 0 to the last (as original SVHN labels)
        # since SVHN itself is long-tailed, label 10 (0 here) may not contain enough images
        classes = np.concatenate([classes[1:], classes[:1]], axis=0)
        # np.random.shuffle(classes)
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        print(f"Labeled data extracted:\t{len(new_targets)}")
        for i in range(self.cls_num):
            print(self.num_per_cls_dict[i])

        # unlabeled data
        aux_mat = sio.loadmat(os.path.join(self.root, 'extra_32x32.mat'))
        aux_data = aux_mat['X']
        aux_data = np.transpose(aux_data, (3, 2, 0, 1))
        aux_truth = aux_mat['y'].astype(np.int64).squeeze()
        np.place(aux_truth, aux_truth == 10, 0)
        print("Loading pseudo labels from %s" % self.unlabeled_pseudo)
        with open(self.unlabeled_pseudo, 'rb') as f:
            aux_targets = pickle.load(f)
        aux_targets = aux_targets['extrapolated_targets']

        for the_class, the_img_num in zip(classes, img_num_per_cls_unlabeled):
            # ground truth is only used to select samples
            idx = np.where(aux_truth == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(aux_data[selec_idx, ...])
            # append pseudo-label
            new_targets.extend(aux_targets[selec_idx])
            for pseudo_class in aux_targets[selec_idx]:
                self.num_per_cls_dict[pseudo_class] += 1
        new_data = np.vstack(new_data)
        self.data = new_data
        self.labels = new_targets
        assert new_data.shape[0] == len(new_targets), 'Length of data & labels do not match!'
        print(f"Unlabeled data extracted:\t{len(new_targets)}")
        for i in range(self.cls_num):
            print(self.num_per_cls_dict[i])

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = SemiSupervisedImbalanceSVHN(root='./data',
                                           split='train', download=True, transform=transform)
    trainloader = iter(trainset)
    data, label = next(trainloader)
    import pdb; pdb.set_trace()
