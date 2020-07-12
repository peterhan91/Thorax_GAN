#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import random
from sklearn import metrics
from cxr_dataset import PatchDataset


def gen_rand_noise_with_label(args, device, bs, label=None, cxr=False):
    if label is None:
        label = np.random.randint(0, args.num_classes, bs)
    #attach label into noise
    noise = np.random.normal(0, 1, (bs, 128))
    if cxr:
        prefix = label
    else:
        prefix = np.zeros((bs, args.num_classes))
        prefix[np.arange(bs), label] = 1

    noise[np.arange(bs), :args.num_classes] = prefix[np.arange(bs)]

    noise = torch.from_numpy(noise).float()
    noise = noise.to(device)

    return noise


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class LocalGDUpdate(object):
    def __init__(self, args, dataframe=None, device=None):
        self.args = args
        self.dataframe = dataframe
        print('local dataframe length: ', len(self.dataframe))
        self.device = device
        self.loss_func = nn.BCELoss()
        self.transform = transforms.Compose([
                                            transforms.Resize(128),
                                            transforms.CenterCrop(128),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5],std=[0.5])
                                            ])
        self.ldr_train = DataLoader(PatchDataset(args.path_to_folder,
                                    df = self.dataframe,
                                    transform=self.transform,
                                    ), 
                                    batch_size=self.args.local_bs, 
                                    shuffle=True, 
                                    num_workers=48, 
                                    drop_last=True, 
                                    pin_memory=True,
                                    )

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(self.args.local_bs, 1)
        alpha = alpha.expand(self.args.local_bs, 
                        int(real_data.nelement()/self.args.local_bs)).contiguous()
        alpha = alpha.view(self.args.local_bs, 1, 128, 128)
        alpha = alpha.to(self.device)

        fake_data = fake_data.view(self.args.local_bs, 1, 128, 128)
        interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

        interpolates = interpolates.to(self.device)
        interpolates.requires_grad_(True)   

        disc_interpolates, _ = netD(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(gradients.size(0), -1)                              
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
        return gradient_penalty
    
    def train(self, aG, aD):
        dataiter = iter(self.ldr_train)
        gen_iterations = 0
        GENER_ITERS, CRITIC_ITERS = 1, 5

        optimizer_g = torch.optim.SGD(aG.parameters(), lr=self.args.lr, momentum=0.5)
        optimizer_d = torch.optim.SGD(aD.parameters(), lr=self.args.lr, momentum=0.5)
        iteration_lossG, iteration_lossD = [], []

        for iteration in range(self.args.local_ep):
            #---------------------TRAIN G------------------------
            for p in aD.parameters():
                p.requires_grad_(False)  # freeze D
            gen_cost = None
            for i in range(GENER_ITERS):
                # Gather labels for conditional generation
                try: 
                    batch = next(dataiter, None)
                    if batch is None:
                        dataiter = iter(self.ldr_train)
                        batch = dataiter.next()
                except StopIteration:
                    dataiter = iter(self.ldr_train)
                    batch = dataiter.next()

                real_label = batch[1]
                aG.zero_grad()
                noise = gen_rand_noise_with_label(self.args, 
                                                self.device, 
                                                self.args.local_bs, 
                                                real_label, cxr=True)
                noise.requires_grad_(True)
                fake_data = aG(noise)
                gen_cost, gen_aux_output = aD(fake_data)
                aux_label = real_label.to(self.device).float()
                aux_errG = self.loss_func(torch.sigmoid(gen_aux_output), aux_label).mean()
                gen_cost = -gen_cost.mean()
                g_cost = aux_errG + gen_cost
                g_cost.backward()
                iteration_lossG.append(gen_cost.item())
            optimizer_g.step()
            gen_iterations += 1

            #---------------------TRAIN D------------------------
            if gen_iterations < 25 or gen_iterations %500 == 0:
                CRITIC_ITERS = 10
            else:
                CRITIC_ITERS = CRITIC_ITERS_
            
            for p in aD.parameters():  # reset requires_grad
                p.requires_grad_(True)  # they are set to False below in training G
            
            for i in range(CRITIC_ITERS):
                aD.zero_grad()
                # gen fake data and load real data
                f_label = np.random.randint(0, self.args.num_classes, self.args.local_bs)
                noise = gen_rand_noise_with_label(self.args, self.device, 
                                                    self.args.local_bs, f_label)
                with torch.no_grad():
                    noisev = noise  # totally freeze G, training D
                fake_data = aG(noisev).detach()

                try: 
                    batch = next(dataiter, None)
                    if batch is None:
                        dataiter = iter(self.ldr_train)
                        batch = dataiter.next() 
                except StopIteration:
                    dataiter = iter(self.ldr_train)
                    batch = dataiter.next() 
                      
                real_data = batch[0] #batch[1] contains labels
                real_data.requires_grad_(True)
                real_label = batch[1]
                real_data = real_data.to(self.device).float()
                real_label = real_label.to(self.device).float()
                # train with real data
                disc_real, aux_output = aD(real_data)
                aux_errD_real = self.loss_func(torch.sigmoid(aux_output), real_label)
                errD_real = aux_errD_real.mean()
                disc_real = disc_real.mean()
                # train with fake data
                disc_fake, aux_output = aD(fake_data)
                disc_fake = disc_fake.mean()
                # train with interpolates data
                gradient_penalty = self.calc_gradient_penalty(aD, real_data, fake_data)
                # final disc cost
                disc_cost = disc_fake - disc_real + gradient_penalty
                disc_acgan = errD_real #+ errD_fake
                (disc_cost + disc_acgan).backward(retain_graph=True)
                w_dist = disc_fake - disc_real
                optimizer_d.step()
                iteration_lossD.append(disc_cost.item())

        return aG.state_dict(), aD.state_dict(), sum(iteration_lossG)/len(iteration_lossG), sum(iteration_lossD)/len(iteration_lossD)


