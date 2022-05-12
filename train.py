import timeit
from datetime import datetime
import os
import glob
from tqdm import tqdm
import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dataloaders.dataset import VideoDataset
from network import C3D_model
import natsort

def train_model(args, dataset, save_dir, saveName, resume_epoch, num_classes, lr, num_epochs, save_epoch, useTest, test_interval):
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """
    model = C3D_model.C3D(num_classes=num_classes, pretrained=False)
    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum = 0.9, weight_decay = 5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    if resume_epoch == 0:
        print("Training C3D from scratch...")
    else:
        checkpoint = torch.load(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'), map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)
    criterion.to(device)

    print('Training model on {} dataset...'.format(dataset))
    train_dataloader = DataLoader(
        VideoDataset(dataset=dataset, split='train',clip_len=16, preprocess=args.preprocess), 
        batch_size=20, shuffle=True, num_workers=args.workers
    )
    train_size = len(train_dataloader.dataset)

    test_dataloader  = DataLoader(
        VideoDataset(dataset=dataset, split='test', clip_len=16, preprocess=args.preprocess), 
        batch_size=20, num_workers=args.workers
    )
    test_size = len(test_dataloader.dataset)

    for epoch in range(resume_epoch, num_epochs):
        # each epoch has a training and validation step
        start_time = timeit.default_timer()

        # reset the running loss and corrects
        running_loss = 0.0
        running_corrects = 0.0

        model.train()

        for inputs, labels in tqdm(train_dataloader):
            # move inputs and labels to the device the training is taking place on
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim = 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            #print(">>> DBG : ", loss.item() * inputs.size(0), torch.sum(preds == labels.data))

        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects.double() / train_size

        print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format('train', epoch+1, num_epochs, epoch_loss, epoch_acc))
        stop_time = timeit.default_timer()
        print("Execution time: " + str(stop_time - start_time) + "\n")

        if epoch % save_epoch == (save_epoch - 1):
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar'))
            print("Save model at {}\n".format(os.path.join(save_dir, 'models', str(num_epochs) + '_epoch-' + str(epoch) + '.pth.tar')))

        if useTest and epoch % test_interval == (test_interval - 1):
            model.eval()
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels, fnames in tqdm(test_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = model(inputs)
                preds = torch.argmax(outputs, dim = 1)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects.double() / test_size

            print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch+1, num_epochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")
        scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='>>> Start training')

    parser.add_argument('--workers', default= 16, type=int)
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--gpu', default=0, type=int)

    parser.add_argument('--nEpochs', default= 100, type=int)
    parser.add_argument('--nTestInterval', default= 10, type=int)
    parser.add_argument('--snapshot', default= 10, type=int)

    parser.add_argument('--lr', default= 0.005, type=float)
    parser.add_argument('--useTest', action='store_true')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    dataset = 'ucf101'
    num_classes = 101

    resume_epoch = 0  # Default is 0, change if want to resume
    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    if resume_epoch != 0:
        runs = natsort.natsorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) if runs else 0
    else:
        runs = natsort.natsorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

    save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
    saveName = 'C3D-' + dataset
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    if not os.path.exists(save_dir+'/models'):
        os.mkdir(save_dir+'/models')

    train_model(args, dataset, save_dir, saveName, resume_epoch, num_classes, args.lr, args.nEpochs, args.snapshot, args.useTest, args.nTestInterval)

# OMP_NUM_THREADS=1 python train.py --gpu 7
# OMP_NUM_THREADS=1 python train.py --workers 32 --gpu 7