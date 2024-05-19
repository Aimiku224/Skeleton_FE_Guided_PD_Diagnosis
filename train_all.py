import os
import time
import argparse
import numpy as np
import json
import torch
import wandb

from torch.utils.data import DataLoader
from torchvision import transforms
from models.stgcn_skeleton_freeze import Stgcnplusplus_skeleton_freeze
from models.res_fer_nofreeze import Res_fer_nofreeze
from models.res_pd_freeze import Res_pd_freeze
from compared_model.MobileNetV3 import woclassifier_mobilenet_v3_small
from models.pd_skeandfer_freeze import PD_skeandfer_freeze
from stgcnplusplus.datasets import build_dataset
from datasets.PD_Dataset import PdDataSet
from datasets.SKEandFER_Dataset import SKEandFERDataset
from utils import seed_model


def parse_args(stgcn, train, validation, flag, pdfer_pretrained_model):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='Skeleton_FE_Guided_PD_Diagnosis/data/face',
                        help='dataset path.')
    parser.add_argument('--label_path', type=str, 
                        default='Skeleton_FE_Guided_PD_Diagnosis/label_lists/Ske_pdface_label',
                        help='dataset label path.')
    parser.add_argument('--pdfer_pretrained_model', type=str,
                        default=pdfer_pretrained_model,
                        help='pretrained_weights')
    parser.add_argument('--train', type=str, default=train, help='train label file name')
    parser.add_argument('--validation', type=str, default=validation, help='eval label file name')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='Pytorch checkpoint file path')
    parser.add_argument('--train_batch_size', type=int, default=8, help='Batch size for train.')
    parser.add_argument('--val_batch_size', type=int, default=8, help='Batch size for validation.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--log_dir', type=str, default='Skeleton_FE_Guided_PD_Diagnosis/log', 
                        help='file path for logging trian details')
    parser.add_argument('--ckpt_dir', type=str, default='Skeleton_FE_Guided_PD_Diagnosis/checkpoints', 
                        help='file path for saving model checkpoints')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs.')
    parser.add_argument('--wandb', type=bool, default=False)
    parser.add_argument('--skeleton_model', type=json.loads, default=stgcn['model'], help='model configuration')
    parser.add_argument('--ske_pretrained_model', type=str, default=stgcn['pretrained_model'],
                        help='skeleton stgcn++ pretrained')
    parser.add_argument('--skedataset', type=json.loads, default=stgcn['data'], help='skeleton dataset')
    parser.add_argument('--flag', type=int, default=flag, help='feature fusion falg')
    return parser.parse_args()


def run_train(args):
    pdpremodel_filename, extension = os.path.splitext(os.path.basename(args.pdfer_pretrained_model))
    skepremodel_filename, extension = os.path.splitext(os.path.basename(args.ske_pretrained_model))

    if args.wandb:
        wandb.login()
        num = args.train[-5:-4]
        wadb = wandb.init(
            # Set the project where this run will be logged
            project="Skeleton_FE_Guided_PD_Diagnosis",
            name=f"{pdpremodel_filename}-{skepremodel_filename}-{str(args.flag)}-{num}",
            # Track hyperparameters and run metadata
            config={
                "epochs": args.epochs, 
                "learning_rate": args.lr, 
                "train_batch_size": args.train_batch_size,
                "val_batch_size": args.val_batch_size,
                "momentum": args.momentum,
                "train": args.train,
                "validation": args.validation,
                "pdpremodel_filename": pdpremodel_filename,
                "skepremodel_filename": skepremodel_filename
            },
        )

    seed_model(gpu_id=1)
    
    log_path = os.path.join(args.log_dir, skepremodel_filename, pdpremodel_filename, str(args.flag))
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        print(f"Created directory: {log_path}")
    else:
        print(f"Directory already exists: {log_path}")

    with open(os.path.join(log_path, 'log-'+time.strftime("%Y-%m-%d-%H-%M-%S-",
                                                           time.localtime())+args.train), 'a') as f_log:
        print(args.train)
        print(args.validation)
        time_start=time.time()

        skemodel = Stgcnplusplus_skeleton_freeze(args.skeleton_model, pretrained=args.ske_pretrained_model)
        # pdfermodel = Res_fer_nofreeze(num_classes = 6)
        # pdfermodel = Res_pd_freeze(modelname='ResNet18', model=pdfermodel, pretrained=args.pdfer_pretrained_model, 
        #                            num_classes=2)
        pdfermodel = woclassifier_mobilenet_v3_small(num_classes=6)
        pdfermodel = Res_pd_freeze(modelname='MobileNetV3', model=pdfermodel, pretrained=args.pdfer_pretrained_model, 
                                   num_classes=2)
        model = PD_skeandfer_freeze(modelname='MobileNetV3', skemodel=skemodel, pdfermodel=pdfermodel, num_classes=2, flag=args.flag)

        if args.checkpoint is not None:
            print("Loading pretrained weights...", args.checkpoint)
            model.load_state_dict(torch.load(args.checkpoint))
            f_log.write("checkpoint:" + args.checkpoint + "\n")

        ckpt_path = os.path.join(args.ckpt_dir, skepremodel_filename, pdpremodel_filename, 
                                 str(args.flag), args.train[:-4])
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
            print(f"Created directory: {ckpt_path}")
        else:
            print(f"Directory already exists: {ckpt_path}")

        f_log.write("pdfer_pretrained_model:" + args.pdfer_pretrained_model + "\n")
        f_log.write("ske_pretrained_model:" + args.ske_pretrained_model + "\n")
        f_log.write("train_batch_size:" + str(args.train_batch_size) + "\n")
        f_log.write("val_batch_size:" + str(args.val_batch_size) + "\n")
        
        # prepare data loaders
        dataloader_setting = dict(
            videos_per_gpu=args.skedataset['videos_per_gpu'],
            workers_per_gpu=args.skedataset['workers_per_gpu'],
            persistent_workers=False,
            shuffle=False
        )
        train_skedatasets = build_dataset(args.skedataset['train'])
        train_dataloader_setting = dict(dataloader_setting, **args.skedataset['train_dataloader'])

        data_transforms_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(scale=(0.02,0.25))
        ])
        train_pdferdatasets = PdDataSet(args.data_path, args.label_path, args.train, mode='train',
                                         transform=data_transforms_train)

        train_datasets = SKEandFERDataset(args, train_skedatasets, train_dataloader_setting, train_pdferdatasets)
        train_num = train_datasets.__len__()
        print('Train set size:', train_num)
        f_log.write('Train set size:' + str(train_num) + '\n')
        # train_datasets.print_sample()

        train_loader = DataLoader(train_datasets, batch_size=args.train_batch_size, shuffle=True,
                                   num_workers=args.workers, drop_last=True)

        val_skedatasets = build_dataset(args.skedataset['val'], dict(test_mode=True))
        val_dataloader_setting = dict(dataloader_setting, **args.skedataset['val_dataloader'])

        data_transforms_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_pdferdatasets = PdDataSet(args.data_path, args.label_path, args.validation, mode='eval',
                                       transform=data_transforms_val)
        
        val_datasets = SKEandFERDataset(args, val_skedatasets, val_dataloader_setting, val_pdferdatasets)
        val_num = val_datasets.__len__()
        print('Validation set size:', val_num)
        f_log.write('Validation set size:' + str(val_num) + '\n')

        val_loader = DataLoader(val_datasets, batch_size=args.val_batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=True, drop_last=False)

        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                          lr=args.lr, weight_decay=1e-5)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                         args.lr, momentum=args.momentum, weight_decay=1e-4)
        else:
            raise ValueError("Optimizer not supported.")
        print(optimizer)
        f_log.write(str(optimizer)+'\n')
        f_log.write("model:" + str(model) + "\n")
        for k,v in model.named_parameters():
            f_log.write('{}: {}'.format(k, v.requires_grad) + "\n")
            print('{}: {}'.format(k, v.requires_grad))

        model = model.cuda()
        CE_criterion = torch.nn.CrossEntropyLoss()

        best_acc = 0
        for i in range(1, args.epochs + 1):
            train_loss = 0.0
            correct_sum = 0
            iter_cnt = 0
            model.train()
            for batch_i, (imgs, keypoints, labels) in enumerate(train_loader):
                iter_cnt += 1
                optimizer.zero_grad()
                # print(np.array(keypoints).shape)
                images = [img.cuda() for img in imgs[:6]]
                outputs = model(images, keypoints.cuda())
                targets = labels.cuda().squeeze()
                # print(targets.shape, outputs.shape)

                loss = CE_criterion(outputs, targets)
                loss.backward()
                optimizer.step()     
                train_loss += loss
                _, predicts = torch.max(outputs, 1)
                correct_num = torch.eq(predicts, targets).sum()
                correct_sum += correct_num

                train_wrong_indices = torch.nonzero(torch.ne(predicts, targets))
                
                if args.wandb and i == args.epochs:
                    for j in range(len(train_wrong_indices)):
                        train_wrong_examples = []
                        target = train_wrong_indices[j].item()
                        for k in range(6):        
                            img = np.squeeze(imgs[k])[target]
                            image = wandb.Image(img, caption=f"epoch{i}-{batch_i}-{j}-{k}")
                            train_wrong_examples.append(image)
                        wadb.log({"train_wrong_img": train_wrong_examples})

            train_acc = correct_sum.float() / float(train_datasets.__len__())
            train_loss = train_loss/iter_cnt
            print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f LR: %.6f' %
                (i, train_acc, train_loss, optimizer.param_groups[0]["lr"]))
            f_log.write('[Epoch %d] Training accuracy: %.4f. Loss: %.3f LR: %.6f\n' %
                (i, train_acc, train_loss, optimizer.param_groups[0]["lr"]))  
              
            if i == args.epochs: 
                ckpt_name = 'ckpt_pd_'+args.train[:-4]+'_last'+'.pth'
                torch.save(model.state_dict(), os.path.join(ckpt_path, ckpt_name))
            
            if i%10 == 0:
                ckpt_name = 'ckpt_pd_'+args.train[:-4]+'_epoch'+ str(i) +'.pth'
                torch.save(model.state_dict(), os.path.join(ckpt_path, ckpt_name))

            if args.wandb:
                wadb.log({'train loss': train_loss, 'epoch': i})
                
            with torch.no_grad():
                val_loss = 0.0
                iter_cnt = 0
                bingo_cnt = 0
                model.eval()
                for batch_i, (imgs, keypoints, labels) in enumerate(val_loader):
                    iter_cnt += 1
                    images = [img.cuda() for img in imgs[:6]]
                    outputs = model(images, keypoints.cuda())
                    targets = labels.cuda().squeeze()

                    CE_loss = CE_criterion(outputs, targets)
                    loss = CE_loss
                    val_loss += loss

                    _, predicts = torch.max(outputs, 1)
                    correct_or_not = torch.eq(predicts, targets)
                    bingo_cnt += correct_or_not.sum().cpu()

                    val_wrong_indices = torch.nonzero(torch.ne(predicts, targets))
                    
                    if args.wandb and i == args.epochs:
                        for j in range(len(val_wrong_indices)):
                            val_wrong_examples = []
                            target = val_wrong_indices[j].item()
                            for k in range(6):        
                                img = np.squeeze(imgs[k])[target]
                                image = wandb.Image(img, caption=f"epoch{i}-{batch_i}-{j}-{k}")
                                val_wrong_examples.append(image)
                            wadb.log({"val_wrong_img": val_wrong_examples})
                    
                val_loss = val_loss/iter_cnt
                val_acc = bingo_cnt.float()/float(val_num)
                val_acc = np.around(val_acc.numpy(), 4)
                print("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f" % (i, val_acc, val_loss))
                f_log.write("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f\n" % (i, val_acc, val_loss))

                if args.wandb:
                    wadb.log({'val loss': val_loss, 'val acc': val_acc, 'epoch': i})

                if val_acc > best_acc:
                    best_acc = val_acc
                    print("best_acc:" + str(best_acc))
                    ckpt_name = 'ckpt_pd_'+args.train[:-4]+'_best_acc'+'_'+str(int(val_acc*10000))+'.pth'
                    torch.save(model.state_dict(), os.path.join(ckpt_path, ckpt_name))
                    f_log.write("best_acc:" + str(best_acc) + '\n')

                    if args.wandb:
                        wadb.log({'best acc': best_acc})
                
        time_end=time.time()
        time_elapsed = time_end - time_start
        print('Time cost:',time_elapsed,'s')
        f_log.write('Time cost:' + str(time_elapsed) + 's\n')
        print("best_acc:", best_acc)
        f_log.write("best_acc:" + str(best_acc) + '\n')

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    with open('Skeleton_FE_Guided_PD_Diagnosis/models/stgcn++.json', 'r') as f:
        # 读取 JSON 数据
        stgcn = json.load(f)
    pdpremodel = 'Skeleton_FE_Guided_PD_Diagnosis/pretrained_model/MobileNetV3_9086.pth'
    for i in range(2,3):
        for j in range(4,5):
            train = 'train{}.txt'.format(j)
            validation = 'validation{}.txt'.format(j)
            args = parse_args(stgcn, train, validation, i, pdpremodel)
            run_train(args)
    
