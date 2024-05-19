import os
import time
import argparse
import numpy as np
import json
import torch

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

def parse_args(stgcn, test):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='Skeleton_FE_Guided_PD_Diagnosis/data/face', 
                        help='dataset path.')
    parser.add_argument('--label_path', type=str, 
                        default='Skeleton_FE_Guided_PD_Diagnosis/label_lists/Ske_pdface_label', 
                        help='dataset label path.')
    parser.add_argument('--test', type=str, default=test, help='test label file name')
    parser.add_argument('-c', '--checkpoint', type=str, 
                        default='Skeleton_FE_Guided_PD_Diagnosis/checkpoints/stgcn++_fold1_t1_epoch_200/MobileNetV3_9086/2/train4/ckpt_pd_train4_epoch50.pth', 
                        help='Pytorch checkpoint file path')
    parser.add_argument('--test_batch_size', type=int, default=8, help='Batch size for test.')
    parser.add_argument('--log_dir', type=str, default='Skeleton_FE_Guided_PD_Diagnosis/log', 
                        help='file path for logging trian details')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--skeleton_model', type=json.loads, default=stgcn['model'], help='model configuration')
    parser.add_argument('--skedataset', type=json.loads, default=stgcn['data'], help='skeleton dataset')
    parser.add_argument('--ske_pretrained_model', type=str, default=stgcn['pretrained_model'],
                        help='skeleton stgcn++ pretrained')
    parser.add_argument('--pdfer_pretrained_model', type=str,
                        default='Skeleton_FE_Guided_PD_Diagnosis/pretrained_model/MobileNetV3_9086.pth',
                        # Skeleton_FE_Guided_PD_Diagnosis/pretrained_model/ResNet18_8848.pth
                        # Skeleton_FE_Guided_PD_Diagnosis/pretrained_model/MobileNetV3_9086.pth
                        help='pretrained_weights')
    parser.add_argument('--flag', type=int, default=2, help='feature fusion falg')
    return parser.parse_args()


def run_test(args):
    pdpremodel_filename, extension = os.path.splitext(os.path.basename(args.pdfer_pretrained_model))
    skepremodel_filename, extension = os.path.splitext(os.path.basename(args.ske_pretrained_model))

    seed_model(gpu_id=0)

    log_path = os.path.join(args.log_dir, skepremodel_filename, pdpremodel_filename, str(args.flag))
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        print(f"Created directory: {log_path}")
    else:
        print(f"Directory already exists: {log_path}")

    with open(os.path.join(log_path, 'log-'+time.strftime("%Y-%m-%d-%H-%M-%S-", 
                                                              time.localtime())+args.test), 'a') as f_log:
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

        print("batch_size:", args.test_batch_size)
        f_log.write("batch_size:" + str(args.test_batch_size) + "\n")
        f_log.write("checkpoint:" + os.path.basename(args.checkpoint) + "\n")

        # prepare data loaders
        dataloader_setting = dict(
            videos_per_gpu=args.skedataset['videos_per_gpu'],
            workers_per_gpu=args.skedataset['workers_per_gpu'],
            shuffle=False
        )
        test_skedatasets = build_dataset(args.skedataset['test'], dict(test_mode=True))
        test_dataloader_setting = dict(dataloader_setting, **args.skedataset['test_dataloader'])

        data_transforms_test = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        test_pdferdatasets = PdDataSet(args.data_path, args.label_path, args.test, mode='test',
                                       transform=data_transforms_test)

        test_datasets = SKEandFERDataset(args, test_skedatasets, test_dataloader_setting, test_pdferdatasets)
        test_num = test_datasets.__len__()
        print('Test set size:', test_num)
        f_log.write('Test set size:' + str(test_num) + '\n')

        test_loader = DataLoader(test_datasets, batch_size=args.test_batch_size, num_workers=args.workers, 
                                 shuffle=False,drop_last=False, pin_memory=True)

        model = model.cuda()
        CE_criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            test_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            model.eval()
            for batch_i, (imgs, keypoints, labels) in enumerate(test_loader):
                iter_cnt += 1
                images = [img.cuda() for img in imgs[:6]]
                outputs = model(images, keypoints.cuda())    
                targets = labels.cuda().squeeze()

                CE_loss = CE_criterion(outputs, targets)
                loss = CE_loss
                test_loss += loss

                _, predicts = torch.max(outputs, 1)
                correct_or_not = torch.eq(predicts, targets)
                bingo_cnt += correct_or_not.sum().cpu()
                
            test_loss = test_loss/iter_cnt
            test_acc = bingo_cnt.float()/float(test_num)
            test_acc = np.around(test_acc.numpy(), 4)
            print("Test accuracy:%.4f. Loss:%.3f" % (test_acc, test_loss))
            f_log.write("Test accuracy:%.4f. Loss:%.3f\n" % (test_acc, test_loss))

        time_end=time.time()
        time_elapsed = time_end - time_start
        print('Time cost:',time_elapsed,'s')
        f_log.write('Time cost:' + str(time_elapsed) + 's\n')

if __name__ == "__main__":
    with open('Skeleton_FE_Guided_PD_Diagnosis/models/stgcn++.json', 'r') as f:
        # 读取 JSON 数据
        stgcn = json.load(f)
    for i in range(1,3):
        test = 'test{}.txt'.format(i)
        args = parse_args(stgcn, test)
        run_test(args)    
    
