import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

RGB_statistics = {
    'ImageNet': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
}


def get_data_transform(split, rgb_mean, rbg_std):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
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


class ImageNetLT(Dataset):
    
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


def load_data_imagenet(data_root, batch_size, phase, sampler_dic=None, num_workers=4, shuffle=True):
    assert phase in {'train', 'val'}
    key = 'ImageNet'
    txt = f'./imagenet_inat/data/ImageNet_LT/ImageNet_LT_{phase}.txt'
    print(f'===> Loading ImageNet data from {txt}')
    rgb_mean, rgb_std = RGB_statistics[key]['mean'], RGB_statistics[key]['std']
    transform = get_data_transform(phase, rgb_mean, rgb_std)

    set_imagenet = ImageNetLT(data_root, txt, transform)
    print(f'===> {phase} data length {len(set_imagenet)}')

    # if phase == 'test' and test_open:
    #     open_txt = './data/%s/%s_open.txt' % (dataset, dataset)
    #     print('Testing with open sets from %s' % open_txt)
    #     open_set_ = INaturalist('./data/%s/%s_open' % (dataset, dataset), open_txt, transform)
    #     set_ = ConcatDataset([set_, open_set_])

    if sampler_dic and phase == 'train':
        print('Using sampler: ', sampler_dic['sampler'])
        print('Sampler parameters: ', sampler_dic['params'])
        return DataLoader(dataset=set_imagenet, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                          sampler=sampler_dic['sampler'](set_imagenet, **sampler_dic['params']))
    else:
        print('No sampler.')
        print('Shuffle is %s.' % shuffle)
        return DataLoader(dataset=set_imagenet, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
