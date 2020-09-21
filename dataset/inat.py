from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image

RGB_statistics = {
    'iNaturalist18': {
        'mean': [0.466, 0.471, 0.380],
        'std': [0.195, 0.194, 0.192]
    }
}


def get_data_transform(split, rgb_mean, rbg_std):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ])
    }
    return data_transforms[split]


class INaturalist(Dataset):

    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label  # , index


def load_data_inat(data_root, batch_size, phase, sampler_dic=None, num_workers=4, shuffle=True):
    assert phase in {'train', 'val'}
    key = 'iNaturalist18'
    txt = f'./imagenet_inat/data/iNaturalist18/iNaturalist18_{phase}.txt'
    print(f'===> Loading iNaturalist18 data from {txt}')
    rgb_mean, rgb_std = RGB_statistics[key]['mean'], RGB_statistics[key]['std']
    transform = get_data_transform(phase, rgb_mean, rgb_std)

    set_inat = INaturalist(data_root, txt, transform)
    print(f'===> {phase} data length {len(set_inat)}')

    # if phase == 'test' and test_open:
    #     open_txt = './data/%s/%s_open.txt' % (dataset, dataset)
    #     print('Testing with open sets from %s' % open_txt)
    #     open_set_ = INaturalist('./data/%s/%s_open' % (dataset, dataset), open_txt, transform)
    #     set_ = ConcatDataset([set_, open_set_])

    if sampler_dic and phase == 'train':
        print('Using sampler: ', sampler_dic['sampler'])
        print('Sampler parameters: ', sampler_dic['params'])
        return DataLoader(dataset=set_inat, batch_size=batch_size, shuffle=False,
                          sampler=sampler_dic['sampler'](set_inat, **sampler_dic['params']), num_workers=num_workers)
    else:
        print('No sampler.')
        print('Shuffle is %s.' % shuffle)
        return DataLoader(dataset=set_inat, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
