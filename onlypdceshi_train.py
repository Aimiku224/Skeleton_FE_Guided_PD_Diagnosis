import os
import time
import argparse
import numpy as np
import wandb

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

def parse_args(train, validation):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='Skeleton_FE_Guided_PD_Diagnosis/data/face',
                        help='dataset path.')
    parser.add_argument('--label_path', type=str, 
                        default='Skeleton_FE_Guided_PD_Diagnosis/label_lists/Onlypdface_label',
                        help='dataset label path.')
    parser.add_argument('--pretrained_model', type=str,
                        default=None,
                        help='pretrained_weights')
    parser.add_argument('--train', type=str, default=train, help='train label file name')
    parser.add_argument('--validation', type=str, default=validation, help='eval label file name')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='Pytorch checkpoint file path')
    parser.add_argument('--train_batch_size', type=int, default=8, help='Batch size.')
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
    parser.add_argument('--weights', type=str, 
                        default='Skeleton_FE_Guided_PD_Diagnosis/checkpoints/ckpt_fer/wopre_ShuffleNet/ckpt_fer_train&val_8015.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--model', default='ShuffleNet', help='model')
    parser.add_argument('--ckptdir', default='wopre_ShuffleNet', help='')
    parser.add_argument('--logdir', default='wopre_ShuffleNet', help='')
    return parser.parse_args()


def run_train(args):
    premodel_filename, extension = os.path.splitext(os.path.basename(args.weights))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.wandb:
        wandb.login()
        num = args.train[-5:-4]
        wadb = wandb.init(
            # Set the project where this run will be logged
            project="FE_Guided_PD_Diagnosis",
            name=f"{premodel_filename}-{num}",
            # Track hyperparameters and run metadata
            config={
                "epochs": args.epochs, 
                "learning_rate": args.lr, 
                "train_batch_size": args.train_batch_size,
                "val_batch_size": args.val_batch_size,
                "momentum": args.momentum,
                "train": args.train,
                "validation": args.validation,
                "pretrained_model": premodel_filename
            },
        )

    seed_model(gpu_id=1)

    log_path = os.path.join(args.log_dir, 'comparedmodel', args.logdir)
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

    # ConvNeXt
        if args.model == 'ConvNeXt':
            model = create_model()
            if args.weights != "":
                assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
                weights_dict = torch.load(args.weights, map_location=device)
                # 删除有关分类类别的权重
                for k in list(weights_dict.keys()):
                    if "classifier" in k:
                        del weights_dict[k]
                print(model.load_state_dict(weights_dict, strict=False))
            model = newmodel_freeze(modelname=args.model, model=model, num_classes=2)
            if args.freeze_layers:
                for name, para in model.named_parameters():
                    # 除classifier外，其他权重全部冻结
                    if "classifier" not in name:
                        para.requires_grad_(False)
                    else:
                        print("training {}".format(name))
        
    # GoogleNet
        if args.model == 'GoogleNet':
            model = GoogLeNet(num_classes=6, aux_logits=True, init_weights=True)
            model = newmodel_freeze(modelname=args.model, model=model, num_classes=2)

    # DenseNet
        if args.model == 'DenseNet':
            model = woclassifier_densenet121(num_classes=6).to(device)
            if args.weights != "":
                if os.path.exists(args.weights):
                    weights_dict = torch.load(args.weights, map_location=device)
                    load_weights_dict = {k: v for k, v in weights_dict.items() 
                                         if 'classifier' not in k and  model.state_dict()[k].numel() == v.numel()}
                    print(model.load_state_dict(load_weights_dict, strict=False))
                else:
                    raise FileNotFoundError("not found weights file: {}".format(args.weights))
            model = newmodel_freeze(modelname=args.model, model=model, num_classes=2)
            # 是否冻结权重
            if args.freeze_layers:
                for name, para in model.named_parameters():
                    # 除最后的全连接层外，其他权重全部冻结
                    if "classifier" not in name:
                        para.requires_grad_(False)

    # EfficientNetV2
        if args.model == 'EfficientNetV2':
            # 如果存在预训练权重则载入
            model = woclassifier_efficientnetv2_s(num_classes=6).to(device)
            if args.weights != "":
                if os.path.exists(args.weights):
                    weights_dict = torch.load(args.weights, map_location=device)
                    load_weights_dict = {k: v for k, v in weights_dict.items() 
                                         if 'classifier' not in k and  model.state_dict()[k].numel() == v.numel()}
                    print(model.load_state_dict(load_weights_dict, strict=False))
                else:
                    raise FileNotFoundError("not found weights file: {}".format(args.weights))
            model = newmodel_freeze(modelname=args.model, model=model, num_classes=2)
            # 是否冻结权重
            if args.freeze_layers:
                for name, para in model.named_parameters():
                    # 除classifier外，其他权重全部冻结
                    if "classifier" not in name:
                        para.requires_grad_(False)
                    else:
                        print("training {}".format(name))     

    # RegNet
        if args.model == 'RegNet':
            model = create_wofc_regnet(num_classes=6).to(device)
            if args.weights != "":
                if os.path.exists(args.weights):
                    weights_dict = torch.load(args.weights, map_location=device)
                    load_weights_dict = {k: v for k, v in weights_dict.items()
                                        if 'head.fc' not in k and model.state_dict()[k].numel() == v.numel()}
                    print(model.load_state_dict(load_weights_dict, strict=False))
                else:
                    raise FileNotFoundError("not found weights file: {}".format(args.weights))
            model = newmodel_freeze(modelname=args.model, model=model, num_classes=2)
            #是否冻结权重
            if args.freeze_layers:
                for name, para in model.named_parameters():
                    # 除最后的全连接层外，其他权重全部冻结
                    if "classifier" not in name:
                        para.requires_grad_(False)
                    else:
                        print("train {}".format(name))

     # ResNeXt50_32x4d
        if args.model == 'ResNeXt50_32x4d':
            model = resnext50_32x4d(num_classes=6)
            if args.weights != "":
                if os.path.exists(args.weights):
                    weights_dict = torch.load(args.weights, map_location=device)
                    load_weights_dict = {k: v for k, v in weights_dict.items()
                                        if 'fc' not in k and model.state_dict()[k].numel() == v.numel()}
                    print(model.load_state_dict(load_weights_dict, strict=False))
                else:
                    raise FileNotFoundError("not found weights file: {}".format(args.weights))
            model = newmodel_freeze(modelname=args.model, model=model, num_classes=2)
            # 是否冻结权重
            if args.freeze_layers:
                for name, para in model.named_parameters():
                    # 除最后的全连接层外，其他权重全部冻结
                    if "fc" not in name:
                        para.requires_grad_(False)
                    else:
                        print("train {}".format(name))
    
    # SwinTransformer
        if args.model == 'SwinTransformer':    
            model = swin_tiny_patch4_window7_224(num_classes=6).to(device)
            if args.weights != "":
                assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
                weights_dict = torch.load(args.weights, map_location=device)["model"]
                # 删除有关分类类别的权重
                for k in list(weights_dict.keys()):
                    if "head" in k:
                        del weights_dict[k]
                print(model.load_state_dict(weights_dict, strict=False))
            model = newmodel_freeze(modelname=args.model, model=model, num_classes=2)
            if args.freeze_layers:
                for name, para in model.named_parameters():
                    # 除head外，其他权重全部冻结
                    if "head" not in name:
                        para.requires_grad_(False)
                    else:
                        print("training {}".format(name))

    # MobileViT
        if args.model == 'MobileViT':   
            model = woclassifier_mobile_vit_xx_small(num_classes=6).to(device)
            if args.weights != "":
                assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
                weights_dict = torch.load(args.weights, map_location=device)
                weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
                # 删除有关分类类别的权重
                for k in list(weights_dict.keys()):
                    if "classifier" in k:
                        del weights_dict[k]
                print(model.load_state_dict(weights_dict, strict=False))
            model = newmodel_freeze(modelname=args.model, model=model, num_classes=2)
            if args.freeze_layers:
                for name, para in model.named_parameters():
                    # 除head外，其他权重全部冻结
                    if "classifier" not in name:
                        para.requires_grad_(False)
                    else:
                        print("training {}".format(name))

    # MobileNetV3
        if args.model == 'MobileNetV3':   
            model = woclassifier_mobilenet_v3_small(num_classes=6).to(device)
            if args.weights != "":
                if os.path.exists(args.weights):
                    weights_dict = torch.load(args.weights, map_location=device)
                    load_weights_dict = {k: v for k, v in weights_dict.items()
                                        if 'classifier.3' not in k and model.state_dict()[k].numel() == v.numel()}
                    print(model.load_state_dict(load_weights_dict, strict=False))
                else:
                    raise FileNotFoundError("not found weights file: {}".format(args.weights))
            if args.freeze_layers:
                for name, para in model.named_parameters():
                    # 除head外，其他权重全部冻结
                    if "classifier" not in name:
                        para.requires_grad_(False)
                    else:
                        print("training {}".format(name))
            model = newmodel_freeze(modelname=args.model, model=model, num_classes=2)
            
    # ShuffleNet
        if args.model == 'ShuffleNet':   
            model = wofc_shufflenet_v2_x1_0(num_classes=6).to(device)
            if args.weights != "":
                if os.path.exists(args.weights):
                    weights_dict = torch.load(args.weights, map_location=device)
                    load_weights_dict = {k: v for k, v in weights_dict.items()
                                        if 'fc' not in k and model.state_dict()[k].numel() == v.numel()}
                    print(model.load_state_dict(load_weights_dict, strict=False))
                else:
                    raise FileNotFoundError("not found weights file: {}".format(args.weights))
            model = newmodel_freeze(modelname=args.model, model=model, num_classes=2)
            if args.freeze_layers:
                for name, para in model.named_parameters():
                    # 除head外，其他权重全部冻结
                    if "classifier" not in name:
                        para.requires_grad_(False)
                    else:
                        print("training {}".format(name))
        
        if args.checkpoint:
            print("Loading pretrained weights...", args.checkpoint)
            model.load_state_dict(torch.load(args.checkpoint))
            f_log.write("checkpoint:" + args.checkpoint + "\n")

        ckpt_path = os.path.join(args.ckpt_dir, 'ckpt_onlypdface', args.ckptdir, 
                                 args.train[:-4])
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
            print(f"Created directory: {ckpt_path}")
        else:
            print(f"Directory already exists: {ckpt_path}")

        f_log.write("pretrained_model:" + args.weights + "\n")
        f_log.write("train_batch_size:" + str(args.train_batch_size) + "\n")
        f_log.write("val_batch_size:" + str(args.val_batch_size) + "\n")

        data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(scale=(0.02,0.25))])

        train_dataset = OnlyPdFaceDataSet(args.data_path, args.label_path, args.train, mode='train', 
                                          transform=data_transforms)
        print('Train set size:', train_dataset.__len__())
        f_log.write('Train set size:' + str(train_dataset.__len__()) + '\n')
        
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, 
                                  num_workers=args.workers, drop_last=True)
        
        data_transforms_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        val_dataset = OnlyPdFaceDataSet(args.data_path, args.label_path, args.validation, mode='eval', 
                                        transform=data_transforms_val)
        val_num = val_dataset.__len__()
        print('Validation set size:', val_num)
        f_log.write('Validation set size:' + str(val_num) + '\n')
        
        val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, num_workers=args.workers, 
                                shuffle=False, pin_memory=True, drop_last=False)

        if args.optimizer == 'adam':
            if args.model == 'SwinTransformer':
                pg = [p for p in model.parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)
            # if args.model == 'MobileViT':
            #     pg = [p for p in model.parameters() if p.requires_grad]
            #     optimizer = torch.optim.AdamW(pg, lr=args.lr, weight_decay=1E-2)
            else:
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                                    weight_decay=1e-5)
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
            for batch_i, (imgs, targets) in enumerate(train_loader):
                iter_cnt += 1
                optimizer.zero_grad()
                images = [img.cuda() for img in imgs[:6]]
                outputs = model(tuple(images))
                targets = targets.cuda().squeeze()
                #print(targets.shape, outputs.shape)

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
                            img = imgs[k][target]
                            image = wandb.Image(img, caption=f"epoch{i}-{batch_i}-{j}-{k}")
                            train_wrong_examples.append(image)
                        wadb.log({"train_wrong_img": train_wrong_examples})

            train_acc = correct_sum.float() / float(train_dataset.__len__())
            train_loss = train_loss/iter_cnt
            print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f LR: %.6f' %
                (i, train_acc, train_loss, optimizer.param_groups[0]["lr"]))
            f_log.write('[Epoch %d] Training accuracy: %.4f. Loss: %.3f LR: %.6f\n' %
                (i, train_acc, train_loss, optimizer.param_groups[0]["lr"]))
            
            if i == args.epochs: 
                ckpt_name = 'ckpt_pd_'+args.train[:-4]+'_last'+'.pth'
                torch.save(model.state_dict(), os.path.join(ckpt_path, ckpt_name))

            # if i%10 == 0:
            #     ckpt_name = 'ckpt_pd_'+args.train[:-4]+'_epoch'+ str(i) +'.pth'
            #     torch.save(model.state_dict(), os.path.join(ckpt_path, ckpt_name))

            if args.wandb:
                wadb.log({'train loss': train_loss, 'epoch': i})

            with torch.no_grad():
                if i%1==0:
                    val_loss = 0.0
                    iter_cnt = 0
                    bingo_cnt = 0
                    model.eval()
                    for batch_i, (imgs, targets) in enumerate(val_loader):
                        iter_cnt += 1
                        images = [img.cuda() for img in imgs[:6]]
                        outputs = model(tuple(images))
                        targets = targets.cuda().squeeze()

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
                                    img = imgs[k][target]
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
                        ckpt_name = 'ckpt_pd_'+args.train[:-4]+'_'+str(int(val_acc*10000))+'.pth'
                        torch.save(model.state_dict(), os.path.join(ckpt_path,ckpt_name))
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
    for i in range(1,6):
        train = 'train{}.txt'.format(i)
        validation = 'validation{}.txt'.format(i)
        args = parse_args(train,validation)
        run_train(args)