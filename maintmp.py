import os
import tqdm
import numpy as np
from torchsummary import summary

import torch
import torchvision
from torchvision import transforms

from copy import deepcopy

from nets.nn import resnet152
from utils.loss import yoloLoss
from utils.dataset import Dataset
from utils.earlystoping import EarlyStopping

import argparse
import re

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    root = args.data_dir
    num_epochs = args.epoch
    batch_size = args.batch_size
    learning_rate = args.lr
    early_stopping = EarlyStopping(args.patience) if args.early_stopping else None
    image_size = (3, args.img_size, args.img_size)
    
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # net = resnet50()s5
    # net = swintransformer(NET_CONFIG, SwinTransformerVersion.SWIN_T)
    # net = resnext50(pretrained=False)
    # net = visionTransformer(NET_CONFIG["BACKBONE"]["VIT"])
    # net = resnext152()
    net = resnet152()
    
    if(args.pre_weights != None): # 학습된 모델 불러오기
        pattern = 'yolov1_([0-9]+)'
        splited = args.pre_weights.split('_')
        # f_name = strs.split('/')[-1]
        # epoch_str = re.search(pattern,f_name).group(1)
        epoch_str = splited[-1]
        epoch_start = int(epoch_str) + 1
        net.load_state_dict( \
            torch.load(f'./weights/{args.pre_weights}.pth')["state_dict"])
    else:
        epoch_start = 1

    print('NUMBER OF CUDA DEVICES:', torch.cuda.device_count())

    criterion = yoloLoss().to(device)
    net = net.to(device)

    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    summary(net,input_size=(3,448,448))
    # different learning rate

    net.train()

    params = []
    params_dict = dict(net.named_parameters())
    for key, value in params_dict.items():
        if "features" in key:
            continue
            # params += [{'params': [value], 'lr': learning_rate * 10}]
        else:
            params += [{'params': [value], 'lr': learning_rate}]

    # pred = net(torch.randn((1, 3, 448, 448), device=device))
    # print(pred.shape)

    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, 
                                                                  T_mult=1, eta_min=0.00001)
    # optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=5e-6)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=2, eta_min=1e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 9, 15, 20, 30], gamma=0.05)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    if args.pre_weights:
        optimizer.load_state_dict(torch.load(f"./weights/{args.pre_weights}.pth")["optimizer"])
        scheduler.load_state_dict(torch.load(f"./weights/{args.pre_weights}.pth")["scheduler"])
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.05, last_epoch=epoch_start-1)
        # scheduler.load_state_dict(torch.load(f"./weights/{args.pre_weights}.pth")["scheduler"])

    # optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    with open('./Dataset/train.txt') as f:
        train_names = f.readlines()
    train_dataset = Dataset(root, train_names, train=True, transform=[transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                            num_workers=os.cpu_count())

    with open('./Dataset/test.txt') as f:
        test_names = f.readlines()
    test_dataset = Dataset(root, test_names, train=False, transform=[transforms.ToTensor()])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size // 2, shuffle=False,
                                            num_workers=os.cpu_count())

    print(f'NUMBER OF DATA SAMPLES: {len(train_dataset)}')
    print(f'BATCH SIZE: {batch_size}')

    for epoch in range(epoch_start,num_epochs):
        net.train()

        # if epoch in [30, 40]:
        #     learning_rate /= 10
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = learning_rate

        # training
        total_loss = 0.
        print(('\n' + '%10s' * 3) % ('epoch', 'loss', 'gpu'))
        progress_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (images, target) in progress_bar:
            images = images.to(device)
            target = target.to(device)

            pred = net(images)
            
            optimizer.zero_grad()
            loss = criterion(pred, target.float())

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            s = ('%10s' + '%10.4g' + '%10s') % ('%g/%g' % (epoch, num_epochs), total_loss / (i + 1), mem)
            progress_bar.set_description(s)
        scheduler.step()
        
        
        # validation
        validation_loss = 0.0
        net.eval()
        with torch.no_grad():
            progress_bar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader))
            for i, (images, target) in progress_bar:
                images = images.to(device)
                target = target.to(device)

                prediction = net(images)
                loss = criterion(prediction, target)
                validation_loss += loss.data
            
        validation_loss /= len(test_loader)
        print(f'Validation_Loss:{validation_loss:07.3}')

        if early_stopping:
            isbest = early_stopping(validation_loss, net)
            if isbest:
                best_epoch = epoch
                save = {'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()}
                torch.save(save, f"./weights/yolov1_{args.backbone}_{best_epoch:04}.pth")
            if early_stopping.early_stop:
                break
        elif epoch % 5 == 0:
            save = {'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()}
            torch.save(save, f"./weights/yolov1_{args.backbone}_{epoch:04}.pth")

    # save = {'state_dict': net.state_dict()}
    save = {'state_dict': torch.load(f'./weights/yolov1_{args.backbone}_{best_epoch:04}.pth')['state_dict']}
    torch.save(save, './weights/yolov1_final.pth')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument('--early-stopping', dest='early_stopping', action='store_true')
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data_dir", type=str, default='./Dataset')
    parser.add_argument("--pre_weights", type=str, help="pretrained weight")
    parser.add_argument("--save_dir", type=str, default="./weights")
    parser.add_argument("--img_size", type=int, default=448)
    parser.add_argument("--backbone", type=str, default='resnet50')
    args = parser.parse_args()
    
    # args.pre_weights = 'yolov1_0010.pth'
    main(args)