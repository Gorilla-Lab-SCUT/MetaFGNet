import os
import shutil
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from data.folder_new import ImageFolder_new
from data.dataloader_new import DataLoader_new

def split_train_test_images(data_dir):
    #data_dir = '/home/lab-zhangyabin/project/fine-grained/.../'
    src_dir = os.path.join(data_dir, 'images')
    target_dir = os.path.join(data_dir, 'splited_image')
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
        print(src_dir)
    train_test_split = open(os.path.join(data_dir, 'train_test_split.txt'))
    line = train_test_split.readline()
    images = open(os.path.join(data_dir, 'images.txt'))
    images_line = images.readline()
    ##########################
    print(images_line)
    image_list = str.split(images_line)
    print(image_list[1])
    subclass_name = image_list[1].split('/')[0]
    print(subclass_name)

    print(line)
    class_list = str.split(line)[1]
    print(class_list)


    while images_line:
        image_list = str.split(images_line)
        subclass_name = image_list[1].split('/')[0]  # get the name of the subclass
        print(image_list[0])
        class_label = str.split(line)[1]  # get the label of the image
        # print(type(int(class_label)))
        test_or_train = 'train'
        if class_label == '0':  # the class belong to the train dataset
            test_or_train = 'test'
        train_test_dir = os.path.join(target_dir, test_or_train)
        if not os.path.isdir(train_test_dir):
            os.makedirs(train_test_dir)
        subclass_dir = os.path.join(train_test_dir, subclass_name)
        if not os.path.isdir(subclass_dir):
            os.makedirs(subclass_dir)

        souce_pos = os.path.join(src_dir, image_list[1])
        targer_pos = os.path.join(subclass_dir, image_list[1].split('/')[1])
        shutil.copyfile(souce_pos, targer_pos)
        images_line = images.readline()
        line = train_test_split.readline()


def generate_dataloader(args):
    # Data loading code
    if args.auxiliary_dataset == 'imagenet':
        traindir = os.path.join(args.data_path, 'Data/CLS-LOC/train')
    else:
        traindir = args.data_path
    # traindir = args.data_path # traindir = os.path.join(args.data_path, 'train')
    # valdir = os.path.join(args.data_path, 'val')
    if not os.path.isdir(traindir):
        print('error directory')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # train_dataset = datasets.ImageFolder(
    #     traindir,
    #     transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         # transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    # )

    train_dataset = ImageFolder_new(
        traindir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True, sampler=None
    # )
    train_loader = DataLoader_new(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None
    )
    # val_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(valdir, transforms.Compose([
    #         transforms.Scale(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True
    # )
    return train_loader  #, val_loader

