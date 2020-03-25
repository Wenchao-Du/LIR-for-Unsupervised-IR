"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks import AdaINGen, MsImageDis, Dis_content, VAEGen
from utils import weights_init, get_model_list, vgg_preprocess, load_vgg19, get_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.nn import functional as F
from GaussianSmoothLayer import GaussionSmoothLayer, GradientLoss
import os

class UNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(UNIT_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.gen_a = VAEGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.dis_cont = MsImageDis(256, hyperparameters['dis'])  # discriminator for domain a
        self.dis_a = Dis_content()
        self.gpuid = hyperparameters['gpuID']
        # @ add backgound discriminator for each domain
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters())
        gen_params = list(self.gen_a.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])

        self.content_opt = torch.optim.Adam(self.dis_cont.parameters(), lr= lr / 2., betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)
        self.content_scheduler = get_scheduler(self.content_opt, hyperparameters)

        # Network weight initialization
        self.gen_a.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_cont.apply(weights_init('gaussian'))

        # initialize the blur network
        self.BGBlur_kernel = [5, 9, 15]
        self.BlurNet = [GaussionSmoothLayer(3, k_size, 25).cuda(self.gpuid) for k_size in self.BGBlur_kernel]
        self.BlurWeight = [0.25, 0.5, 1.]
        # self.Gradient = GradientLoss(3, 3)

        # Load VGG model if needed for test
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg19()
            if torch.cuda.is_available():
                self.vgg.cuda(self.gpuid)
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.eval()
        h_a = self.gen_a.encode_cont(x_a)
        output = self.gen_a.dec_cont(h_a)
        return output

    def __compute_kl(self, mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def gen_update(self, x_a, hyperparameters):
        self.gen_opt.zero_grad()
        self.content_opt.zero_grad()
        # encode
        h_c = self.gen_a.encode_cont(x_a)
        h_a = self.gen_a.encode_sty(x_a)
        noise_a = torch.randn(h_c.size()).cuda(h_c.data.get_device())
        h_r = self.gen_a.decode_cont(h_c + noise_a)
        # second encode
        h_cr = self.gen_a.encode_cont(h_r)
        h_a_cont = torch.cat((h_a, h_cr), 1)
        noise_c = torch.randn(h_a_cont.size()).cuda(h_a_cont.data.get_device())
        x_a_recon = self.gen_a.decode_recs(h_a_cont + noise_c)

        # update the content discriminator
        self.loss_ContentD = self.dis_cont.calc_gen_loss(h_c)
        self.loss_gen_adv_a = 0
        # add domain adverisal loss for generator 
        out_a = self.dis_a(h_r)
        
        if hyperparameters['gan_type'] == 'lsgan':
            self.loss_gen_adv_a += torch.mean((out_a - 1)**2)
        elif hyperparameters['gan_type'] == 'nsgan':
            all1 = Variable(torch.ones_like(out_b.data).cuda(self.gpuid), requires_grad=False)
            self.loss_gen_adv_a += torch.mean(F.binary_cross_entropy(F.sigmoid(out_a), all1))
        else:
            assert 0, "Unsupported GAN type: {}".format(hyperparameters['gan_type'])

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_kl_c = self.__compute_kl(h_c)
        self.loss_gen_recon_kl_sty = self.__compute_kl(h_a)

        self.loss_gen_recon_kl_cyc = self.__compute_kl(h_cr)
        # GAN loss
        

        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, h_r, x_a) if hyperparameters['vgg_w'] > 0 else 0
        
        # add background guide loss
        self.loss_bgm = 0
        if hyperparameters['BGM'] != 0: 
            for index, weight in enumerate(self.BlurWeight):
                out_a = self.BlurNet[index](h_r)
                out_real_a = self.BlurNet[index](x_a)
                grad_loss_a = self.recon_criterion(out_a, out_real_a)
                self.loss_bgm += weight * grad_loss_a                           # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['recon_kl_w'] * (self.loss_gen_recon_kl_sty + self.loss_gen_recon_kl_c + self.loss_gen_recon_kl_cyc) + \
                              hyperparameters['BGM'] * self.loss_bgm + \
                              hyperparameters['gan_w'] * self.loss_ContentD                
        self.loss_gen_total.backward()
        self.gen_opt.step()
        self.content_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, x_a, x_b):
        if x_a is None or x_b is None:
            return None
        self.eval()
        x_a_recon, x_ab = [], []
        for i in range(x_a.size(0)):
            h_a = self.gen_a.encode_cont(x_a[i].unsqueeze(0))
            h_a_sty = self.gen_a.encode_sty(x_a[i].unsqueeze(0))
            xab = self.gen_a.decode_cont(h_a)
            x_ab.append(xab)
            hac = self.gen_a.encode_cont(xab)
            h_ba_cont = torch.cat((hac, h_a_sty), 1)
            x_a_recon.append(self.gen_a.decode_recs(h_ba_cont))
        x_a_recon = torch.cat(x_a_recon)
        x_ab = torch.cat(x_ab)
        self.train()
        return x_a, x_a_recon, x_ab, x_b

    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()
        self.content_opt.zero_grad()
        
        # first encode
        h_cont = self.gen_a.encode_cont(x_a)
        h_t_sty = self.gen_a.encode_sty(x_a)
        noise_c = torch.randn(h_cont.size()).cuda(h_cont.data.get_device())
        h_trans = self.gen_a.decode_cont(h_cont + noise_c)
        # seconde encode
        ht_c = self.gen_a.encode_cont(h_trans)
        h_cat = torch.cat((ht_c, h_t_sty), 1)
        noise_h = torch.randn(h_cat.size()).cuda(h_cat.data.get_device())
        h_rec = self.gen_a.dec_recs(h_cat + noise_h)
        
        
        # # @ add content adversial
        self.loss_ContentD = self.dis_cont.calc_dis_loss(h_c, ht_c)
        
        self.loss_dis = 0
        # decode (cross domain)
        out_fake = self.dis_a(h_cr.detach())
        out_real = self.dis_a(x_b)
        
        
        if hyperparameters['gan_type'] == 'lsgan':
            self.loss_dis += torch.mean((out_fake - 0)**2) + torch.mean((out_real - 1)**2)
        elif hyperparameters['gan_type'] == 'nsgan':
            all0 = Variable(torch.zeros_like(out_fake.data).cuda(self.gpuid), requires_grad=False)
            all1 = Variable(torch.ones_like(out_real.data).cuda(self.gpuid), requires_grad=False)
            self.loss_dis += torch.mean(F.binary_cross_entropy(F.sigmoid(out_fake), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out_real), all1))
        else:
            assert 0, "Unsupported GAN type: {}".format(hyperparameters['gan_type'])
        
 

        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis + 0.5 * hyperparameters['gan_w'] * self.loss_ContentD
        self.loss_dis_total.backward()      
        # nn.utils.clip_grad_norm_(self.loss_dis.parameters(), 5) # dis_content update
        self.dis_opt.step()
        self.content_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()
        if self.content_scheduler is not None:
            self.content_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        
        # load discontent discriminator
        last_model_name = get_model_list(checkpoint_dir, "dis_Content")
        state_dict = torch.load(last_model_name)
        self.dis_cont.load_state_dict(state_dict['dis_c'])


        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        self.content_opt.load_state_dict(state_dict['dis_content'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        self.content_scheduler = get_scheduler(self.content_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        dis_Con_name = os.path.join(snapshot_dir, 'dis_Content_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict()}, dis_name)
        torch.save({'dis_c':self.dis_cont.state_dict()}, dis_Con_name)
        #  opt state  'dis_content':self.content_opt.state_dict()
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)