from email.policy import default
import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
import json
import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dataloaders.dataset import VideoDataset
from network import C3D_model


def evaluation(args):
    model = C3D_model.C3D(num_classes=101, pretrained=False)
    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    checkpoint = torch.load(args.trained_model_path,map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
   
    print("Initializing weights from: {}...".format(args.trained_model_path))
    model.load_state_dict(checkpoint['state_dict'])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)
    criterion.to(device)

    test_dataloader  = DataLoader(VideoDataset(dataset='ucf101', split='test', clip_len=16, preprocess=False), batch_size=20, num_workers=args.workers)

    model.eval()

    result = dict()
    total_preds = list()
    total_labels = list()
    
    for inputs, labels, test_vid_name in tqdm(test_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
        preds = torch.argmax(outputs, dim = 1)

        total_preds.extend(np.asarray(preds.to('cpu')))
        total_labels.extend(np.asarray(labels.to('cpu')))


    total_preds = [ int(i) for i in total_preds ]
    total_labels = [ int(i) for i in total_labels ]

    print("[Result] : Acc: {}".format(accuracy_score(total_labels, total_preds)))
    with open('./submit.json', 'w') as f:
        json.dump({'preds' : total_preds, 'labels' : total_labels}, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='>>> Start training')

    parser.add_argument('--workers', default= 16, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--trained_model_path', default='/path/to/model', type=str, required=True)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    evaluation(args)
#/raid/kjlee/workspace/2022.School.ComputerVision/pytorch-video-recognition/run/run_3/models/C3D-ucf101_epoch-9.pth.tar
