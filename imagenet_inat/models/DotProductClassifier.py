import torch.nn as nn
from imagenet_inat.utils import *
from os import path


class DotProduct_Classifier(nn.Module):
    
    def __init__(self, num_classes=1000, feat_dim=2048, *args):
        super(DotProduct_Classifier, self).__init__()
        # print('<DotProductClassifier> contains bias: {}'.format(bias))
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x, *args):
        x = self.fc(x)
        return x, None


def create_model(feat_dim, num_classes=1000, stage1_weights=False, dataset=None, log_dir=None, test=False, *args):
    print('Loading Dot Product Classifier.')
    clf = DotProduct_Classifier(num_classes, feat_dim)

    if not test:
        if stage1_weights:
            assert dataset
            print('Loading %s Stage 1 Classifier Weights.' % dataset)
            if log_dir is not None:
                subdir = log_dir.strip('/').split('/')[-1]
                subdir = subdir.replace('stage2', 'stage1')
                weight_dir = path.join('/'.join(log_dir.split('/')[:-1]), subdir)
                # weight_dir = path.join('/'.join(log_dir.split('/')[:-1]), 'stage1')
            else:
                weight_dir = './logs/%s/stage1' % dataset
            print('==> Loading classifier weights from %s' % weight_dir)
            clf.fc = init_weights(model=clf.fc,
                                  weights_path=path.join(weight_dir, 'final_model_checkpoint.pth'),
                                  classifier=True)
        else:
            print('Random initialized classifier weights.')

    return clf
