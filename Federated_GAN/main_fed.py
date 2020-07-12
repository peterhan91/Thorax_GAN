#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim
from torchvision import transforms, datasets
from torch.autograd import grad

from utils.sampling import *
from utils.options import args_parser
from utils.utils import *
from models.Update import *
from models.conwgan import Generator, Discriminator
from models.Fed import FedAvg

import libs as lib
import libs.plot
from tensorboardX import SummaryWriter
from torchsummary import summary


if __name__ == '__main__':
#----------------------Parse args----------------------
    args = args_parser()
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

#----------------------Create local users----------------------
    if args.dataset == 'cxr':
        df = pd.read_csv(args.pf_path)
        df = df.sample(args.sample, random_state=42)
        dict_users = cxr_noniid(df, args.num_users, args.iid)
    else:
        exit('Error: unrecognized dataset')

#----------------------Create model----------------------
    if args.model == 'gan' and args.dataset == 'cxr':
        aG_glob = Generator().apply(weights_init)
        aD_glob = Discriminator(num_class=args.num_classes).apply(weights_init)
        aG_glob = aG_glob.to(device)
        aD_glob = aD_glob.to(device)
        # print('Generator architecture')
        # print(summary(aG_glob, (128,1,1)))
        # print('Discriminator architecture')
        # print(summary(aD_glob, (1,128,128)))       
    else:
        exit('Error: unrecognized model')
    
    aG_glob.train()
    aD_glob.train()
    writer = SummaryWriter() 
    # copy weights for both G and D
    w_globG = aG_glob.state_dict()
    w_globD = aD_glob.state_dict()

#----------------------Update global model----------------------
    loss_trainG = []
    loss_trainD = []
    for iter in range(args.epochs):
        w_localsG, loss_localsG = [], []
        w_localsD, loss_localsD = [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # idxs_users: randomly pick for instance 10% out of 100 users
        for idx in idxs_users:
            # print(idx, dict_users[idx])
            local = LocalGDUpdate(args=args, dataframe=dict_users[idx], device=device)
            wG, wD, lossG, lossD = local.train(aG=copy.deepcopy(aG_glob), aD=copy.deepcopy(aD_glob))
            w_localsG.append(copy.deepcopy(wG))
            w_localsD.append(copy.deepcopy(wD))
            loss_localsG.append(copy.deepcopy(lossG))
            loss_localsD.append(copy.deepcopy(lossD))
        # update global weights
        w_globG = FedAvg(w_localsG)
        w_globD = FedAvg(w_localsD)
        # copy weight to net_glob
        aG_glob.load_state_dict(w_globG)
        aD_glob.load_state_dict(w_globD)

#----------------------Log losses----------------------
        loss_avgG = sum(loss_localsG) / len(loss_localsG)
        loss_avgD = sum(loss_localsD) / len(loss_localsD)
        print('Round {:3d}, Average G loss {:.3f}, Average D loss {:.3f}'.format(iter, loss_avgG, loss_avgD))
        loss_trainG.append(loss_avgG)
        loss_trainD.append(loss_avgD)
        writer.add_scalar('data/global_gencost', loss_avgG, iter)
        writer.add_scalar('data/global_discloss', loss_avgD, iter)

#----------------------Plot example images----------------------
        fixed_label = []
        for c in range(args.bs):
            fixed_label.append(c%args.num_classes)
        fixed_noise = gen_rand_noise_with_label(args, device, args.bs, fixed_label) 
        
        if iter > 1 and iter % 10 == 0:	
            gen_images = generate_image(args, aG_glob, fixed_noise)
            torchvision.utils.save_image(gen_images, 
                                        args.OUTPUT_PATH + 'samples_{}.png'.format(iter), 
                                        nrow=8, padding=2)
            grid_images = torchvision.utils.make_grid(gen_images, nrow=8, padding=2)
            writer.add_image('images', grid_images, iter)
#----------------------Save model----------------------
            torch.save(aG_glob, args.OUTPUT_PATH + "generator_"+str(iter)+".pt")
            torch.save(aD_glob, args.OUTPUT_PATH + "discriminator_"+str(iter)+".pt")
        

