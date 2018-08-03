# -*- coding: utf-8 -*-
import subprocess
import numpy as np
import torch
import argparse
import threading
import time
import queue
import os
import shutil
import requests
# from HTMLParser import HTMLParser
import re
import urllib3
from PIL import Image
import copy
import ipdb

urllib3.disable_warnings()
http = urllib3.ProxyManager('http://127.0.0.1:8123/')
SHARE_Q = queue.Queue()
_WORKER_THREAD_NUM = 128
class MyThread(threading.Thread):
    """
    doc of class

    Attributess:
        func:
    """
    def __init__(self, func):
        super(MyThread, self).__init__()  
        self.func = func 

    def run(self) :
        self.func()

def do_something(item):
    global score_dir_list, selected_ind
    if item % 100 ==0:
        print('here is the %dth selected image' % item)
    if args.auxiliary_dataset == 'imagenet':
        original_image_dir = score_dir_list[selected_ind[item]].replace(score_dir, source_dir).replace('.pth.tar', '.JPEG')
    elif args.auxiliary_dataset == 'l_bird':
        original_image_dir = score_dir_list[selected_ind[item]].replace(score_dir, source_dir).replace('.pth.tar', '.jpg')
    else:
        raise ValueError('Unavailable auxiliary dataset!')
    selected_image_dir = original_image_dir.replace(source_dir, target_dir)
    if not os.path.isfile(selected_image_dir):
        subprocess.call(['cp', original_image_dir, selected_image_dir])
        print('Have copied the %dth selected image' % item)


def worker():
    global SHARE_Q
    while True :
        if not SHARE_Q.empty():
            item = SHARE_Q.get()
            # ipdb.set_trace()
            do_something(item)
            time.sleep(1)
            SHARE_Q.task_done()
        else:
            break
            
#parser = argparse.ArgumentParser(description='copy selected images from source dir. to target dir.')
#parser.add_argument('--rm_dset', default=False, action='store_true',
#                    help='true to remove existing selected data before reproducing it')
#parser.add_argument('--auxiliary-dataset', type=str, default='imagenet',
#                    help='choose auxiliary dataset between imagenet/l_bird')
#args = parser.parse_args()

def selected_images_multiprocess(arg):
    global score_dir_list, selected_ind, score_dir, source_dir, target_dir, args
    args = arg
    print('the score dir, source dir, and target dir need to be verified.')
    score_dir = args.score_path   ##'/data1/ILSVRC2015/scores_metafgnet_imagenet_5shotCUB_firstIter/Data/CLS-LOC/train/'
    source_dir = args.data_path # '/data1/ILSVRC2015/Data/CLS-LOC/train/'
    target_dir = args.selected_image_path  #'/data1/ILSVRC2015/images_selected_imagenet_5shotCUB_metafgnet_FirstIter_6_per/Data/CLS-LOC/train/'
    
    log_dir = args.log
    score_dir_list = torch.load(os.path.join(log_dir, 'score_dir_list.pth.tar'))
    selected_ind = torch.load(os.path.join(log_dir, 'selected_ind.pth.tar'))
    
    #if args.rm_dset:
    #    if os.path.exists(target_dir):
    #        shutil.rmtree(target_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for cls in sorted(os.listdir(source_dir)): # whether it is in order? test! it's not in order!! sort!!!
        if not os.path.exists(os.path.join(target_dir, cls)):
            os.makedirs(os.path.join(target_dir, cls))
    print('the number of selected images is', len(selected_ind))
    # ipdb.set_trace()
    global SHARE_Q
    threads = []

    for num in range(0, len(selected_ind)):
        SHARE_Q.put(num)
    for i in range(_WORKER_THREAD_NUM) :
        thread = MyThread(worker)
        thread.start()  
        threads.append(thread)
    for thread in threads :
        thread.join()
    SHARE_Q.join()

# if __name__ == '__main__':
#     main()
