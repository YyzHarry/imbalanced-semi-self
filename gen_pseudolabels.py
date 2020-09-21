import torch.backends.cudnn as cudnn

import logging
import os
import pickle
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, SVHN
from torchvision import transforms
import models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='Apply standard trained model to generate labels on unlabeled data')
parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'svhn'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names))
parser.add_argument('--loss_type', default="CE", type=str, choices=['CE', 'Focal', 'LDAM'])
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')
# load trained models
parser.add_argument('--resume', type=str, default='')
# data related
parser.add_argument('--data_dir', default='./data', type=str,
                    help='directory that has unlabeled data')
parser.add_argument('--data_filename', default='ti_80M_selected.pickle', type=str)
parser.add_argument('--output_dir', default='./data', type=str)
parser.add_argument('--output_filename', default='pseudo_labeled_cifar.pickle', type=str)


args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.data_dir, 'prediction.log')),
        logging.StreamHandler()
    ])
logger = logging.getLogger()

logging.info('Prediction on unlabeled data')
logging.info('Args: %s', args)


# Loading unlabeled data
if args.dataset == 'cifar10':
    with open(os.path.join(args.data_dir, args.data_filename), 'rb') as f:
        data = pickle.load(f)

# Loading model
print(f"===> Creating model '{args.arch}'")
assert args.dataset in {'cifar10', 'svhn'}
num_classes = 10
use_norm = True if args.loss_type == 'LDAM' else False
model = models.__dict__[args.arch](num_classes=num_classes, use_norm=use_norm)

if args.gpu is not None:
    torch.cuda.set_device(args.gpu)
    model = model.cuda()
else:
    model = torch.nn.DataParallel(model).cuda()

assert args.resume is not None
if os.path.isfile(args.resume):
    print(f"===> Loading checkpoint '{args.resume}'")
    checkpoint = torch.load(args.resume, map_location=torch.device(f'cuda:{str(args.gpu)}'))
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        if 'linear' in k:
            new_state_dict[k.replace('linear', 'fc')] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    print(f'===> Checkpoint weights found in total: [{len(list(new_state_dict.keys()))}]')
else:
    raise ValueError(f"No checkpoint found at '{args.resume}'")

cudnn.benchmark = True

model.eval()

mean = [0.4914, 0.4822, 0.4465] if args.dataset.startswith('cifar') else [.5, .5, .5]
std = [0.2023, 0.1994, 0.2010] if args.dataset.startswith('cifar') else [.5, .5, .5]
transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

if args.dataset == 'cifar10':
    unlabeled_data = CIFAR10('./data', train=False, transform=transform_val)
    unlabeled_data.data = data['data']
    unlabeled_data.targets = list(data['extrapolated_targets'])
else:
    unlabeled_data = SVHN('./data', split='extra', transform=transform_val)

data_loader = torch.utils.data.DataLoader(unlabeled_data,
                                          batch_size=200,
                                          num_workers=100,
                                          pin_memory=True)

# Running model on unlabeled data
predictions, truths = [], []
for i, (batch, targets) in enumerate(data_loader):
    _, preds = torch.max(model(batch.cuda()), dim=1)
    predictions.append(preds.cpu().numpy())
    if args.dataset == 'svhn':
        truths.append(targets.cpu().numpy())

    if (i+1) % 10 == 0:
        print('Done %d/%d' % (i+1, len(data_loader)))

new_extrapolated_targets = np.concatenate(predictions)
if args.dataset == 'svhn':
    ground_truth = np.concatenate(truths)
    new_targets = dict(extrapolated_targets=new_extrapolated_targets,
                       ground_truth=ground_truth,
                       prediction_model=args.resume)
else:
    new_targets = dict(extrapolated_targets=new_extrapolated_targets,
                       prediction_model=args.resume)

out_path = os.path.join(args.output_dir, args.output_filename)
assert(not os.path.exists(out_path))
with open(out_path, 'wb') as f:
        pickle.dump(new_targets, f)
