import os
import torch
import torchvision.transforms as transforms
from data.folder_new import ImageFolder_new
def generate_dataset_new(args):
    # Data loading code
    # print(args.data_path)
    if args.auxiliary_dataset == 'imagenet':
        traindir = os.path.join(args.data_path, 'Data/CLS-LOC/train')
    else:
        traindir = args.data_path
    # valdir = os.path.join(args.data_path, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = ImageFolder_new(
        traindir,
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=False,
    #     num_workers = args.workers, pin_memory=True, sampler=None
    # )
    # test_dataset = ImageFolder_new(
    #     valdir,
    #     transforms.Compose([
    #         transforms.Scale(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    # )
    # val_loader = torch.utils.data.DataLoader(
    #     ImageFolder_new(valdir, transforms.Compose([
    #         transforms.Scale(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True
    # )
    return train_dataset#, test_dataset

