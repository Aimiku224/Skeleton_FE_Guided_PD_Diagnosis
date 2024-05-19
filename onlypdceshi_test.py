import os
import time
import argparse
import numpy as np

from compared_model.ConvNeXt import convnext_wo_head_tiny as create_model
from compared_model.GoogleNet import GoogLeNet
from compared_model.DenseNet import woclassifier_densenet121
from compared_model.EfficientNetV2 import woclassifier_efficientnetv2_s
from compared_model.RegNet import create_wofc_regnet
from compared_model.ResNet import resnext50_32x4d
from compared_model.SwinTransformer import swin_tiny_patch4_window7_224
from compared_model.MobileViT import woclassifier_mobile_vit_xx_small
from compared_model.MobileNetV3 import woclassifier_mobilenet_v3_small
from compared_model.ShuffleNet import wofc_shufflenet_v2_x1_0

from models.new_head_model import newmodel_freeze
from utils import seed_model
from datasets.OnlyPDface_Dataset import OnlyPdFaceDataSet

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms


def parse_args(test):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='Skeleton_FE_Guided_PD_Diagnosis/data/face',
                        help='dataset path.')
    parser.add_argument('--label_path', type=str, 
                        default='Skeleton_FE_Guided_PD_Diagnosis/label_lists/Onlypdface_label',
                        help='dataset label path.')
    parser.add_argument('--test', type=str, default=test, help='test label file name')
    parser.add_argument('-c', '--checkpoint', type=str, 
                        default='Skeleton_FE_Guided_PD_Diagnosis/checkpoints/ckpt_onlypdface/MobileNetV3/train1/ckpt_pd_train1_last.pth', 
                        help='Pytorch checkpoint file path')
    parser.add_argument('--test_batch_size', type=int, default=8, help='Batch size for test.')
    parser.add_argument('--log_dir', type=str, default='Skeleton_FE_Guided_PD_Diagnosis/log',
                        help='file path for logging trian details')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate for sgd.')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--model', default='MobileNetV3', help='model')
    parser.add_argument('--logdir', default='MobileNetV3', help='')
    return parser.parse_args()


def run_test(args):
    seed_model(gpu_id=0)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    log_path = os.path.join(args.log_dir, 'comparedmodel', args.logdir)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        print(f"Created directory: {log_path}")
    else:
        print(f"Directory already exists: {log_path}")

    with open(os.path.join(log_path, 'log-'+time.strftime("%Y-%m-%d-%H-%M-%S-",
                                                               time.localtime())+args.test), 'a') as f_log:
        print(args.test)
        time_start=time.time()

    # ConvNeXt
        if args.model == 'ConvNeXt':
            model = create_model()
            model = newmodel_freeze(modelname=args.model, model=model, num_classes=2)
        
    # GoogleNet
        if args.model == 'GoogleNet':
            model = GoogLeNet(num_classes=6, aux_logits=True, init_weights=True)
            model = newmodel_freeze(modelname=args.model, model=model, num_classes=2)

    # DenseNet
        if args.model == 'DenseNet':
            model = woclassifier_densenet121(num_classes=6).to(device)
            model = newmodel_freeze(modelname=args.model, model=model, num_classes=2)

    # EfficientNetV2
        if args.model == 'EfficientNetV2':
            model = woclassifier_efficientnetv2_s(num_classes=6).to(device)
            model = newmodel_freeze(modelname=args.model, model=model, num_classes=2)

    # RegNet
        if args.model == 'RegNet':
            model = create_wofc_regnet(num_classes=6).to(device)
            model = newmodel_freeze(modelname=args.model, model=model, num_classes=2)

    # ResNeXt50_32x4d
        if args.model == 'ResNeXt50_32x4d':
            model = resnext50_32x4d(num_classes=6)
            model = newmodel_freeze(modelname=args.model, model=model, num_classes=2)

    # SwinTransformer
        if args.model == 'SwinTransformer': 
            model = swin_tiny_patch4_window7_224(num_classes=6).to(device)
            model = newmodel_freeze(modelname=args.model, model=model, num_classes=2)

    # MobileViT
        if args.model == 'MobileViT':
            model = woclassifier_mobile_vit_xx_small(num_classes=6).to(device)
            model = newmodel_freeze(modelname=args.model, model=model, num_classes=2)

    # MobileNetV3
        if args.model == 'MobileNetV3':
            model = woclassifier_mobilenet_v3_small(num_classes=6).to(device)
            model = newmodel_freeze(modelname=args.model, model=model, num_classes=2)
    
    # ShuffleNet
        if args.model == 'ShuffleNet':   
            model = wofc_shufflenet_v2_x1_0(num_classes=6).to(device)
            model = newmodel_freeze(modelname=args.model, model=model, num_classes=2)

        print("batch_size:", args.test_batch_size)
        f_log.write("batch_size:" + str(args.test_batch_size) + "\n")

        if args.checkpoint is not None:
            print("Loading pretrained weights...", args.checkpoint)
            model.load_state_dict(torch.load(args.checkpoint))
            f_log.write("checkpoint:" + args.checkpoint + "\n")


        data_transforms_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        test_dataset = OnlyPdFaceDataSet(args.data_path, args.label_path, args.test, mode='test',
                                  transform=data_transforms_test)
        test_num = test_dataset.__len__()
        print('Test set size:', test_num)
        f_log.write('Test set size:' + str(test_num) + '\n')
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.workers,
                                 shuffle=False, pin_memory=True, drop_last=False)


        model = model.cuda()
        CE_criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            test_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            model.eval()
            for batch_i, (imgs, targets) in enumerate(test_loader):
                iter_cnt += 1
                images = [img.cuda() for img in imgs[:6]]
                outputs = model(images)  
                targets = targets.cuda().squeeze()

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
    for i in range(1,3):
        test = 'test{}.txt'.format(i)
        args = parse_args(test)
        run_test(args)

