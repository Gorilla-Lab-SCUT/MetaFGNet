import os
import shutil
import torch
import scipy.io as scio
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def split_train_test_images(data_dir):
    list_dir = data_dir + 'lists/'
    test_list = scio.loadmat(list_dir + 'test_list.mat')
    train_list = scio.loadmat(list_dir + 'train_list.mat')
    print(test_list)
    src_dir = os.path.join(data_dir, 'Images')
    target_dir_train = os.path.join(data_dir, 'splited_image/train/')
    target_dir_val = os.path.join(data_dir, 'splited_image/val/')
    if not os.path.isdir(target_dir_val):
        os.makedirs(target_dir_val)
        print('the splited images for the dogs not exist, creat a new one.', target_dir_val)

    for i in range(len(test_list['file_list'])):
        full_image_source = os.path.join(src_dir, test_list['file_list'][i][0][0])
        full_image_target = os.path.join(target_dir_val, test_list['file_list'][i][0][0])
        pos = full_image_target.rfind('/')
        full_dir_target = full_image_target[:pos]
        if not os.path.isdir(full_dir_target):
            os.makedirs(full_dir_target)
        shutil.copyfile(full_image_source, full_image_target)

    if not os.path.isdir(target_dir_train):
        os.makedirs(target_dir_train)
        print('the splited images for the dogs not exist, creat a new one.', target_dir_train)

    for i in range(len(train_list['file_list'])):
        full_image_source = os.path.join(src_dir, train_list['file_list'][i][0][0])
        full_image_target = os.path.join(target_dir_train, train_list['file_list'][i][0][0])
        pos = full_image_target.rfind('/')
        full_dir_target = full_image_target[:pos]
        if not os.path.isdir(full_dir_target):
            os.makedirs(full_dir_target)
        shutil.copyfile(full_image_source, full_image_target)

def generate_dataloader(args):
    # Data loading code
    traindir = os.path.join(args.data_path, 'splited_image/train')
    valdir = os.path.join(args.data_path, 'splited_image/val')
    if not os.path.isdir(traindir):
        split_train_test_images(args.data_path)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            # transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers = args.workers, pin_memory=True, sampler=None
    )
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    return train_loader, val_loader

