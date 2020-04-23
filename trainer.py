# The CODE is implemented for unir, which is updated based on "UNIT" (NIPS 2016)
# author: Wenchao. Du

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
        self.gen_b = VAEGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b
        self.dis_content = Dis_content()
        self.gpuid = hyperparameters['gpuID']
        # @ add backgound discriminator for each domain
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])

        self.content_opt = torch.optim.Adam(self.dis_content.parameters(), lr= lr / 2., betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)
        self.content_scheduler = get_scheduler(self.content_opt, hyperparameters)

        # Network weight initialization
        self.gen_a.apply(weights_init(hyperparameters['init']))
        self.gen_b.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))
        self.dis_content.apply(weights_init('gaussian'))

        # initialize the blur network
        self.BGBlur_kernel = [5, 9, 15]
        self.BlurNet = [GaussionSmoothLayer(3, k_size, 25).cuda(self.gpuid) for k_size in self.BGBlur_kernel]
        self.BlurWeight = [0.25, 0.5, 1.]
        self.Gradient = GradientLoss(3, 3)

        # # Load VGG model if needed for test
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
        # h_a_sty = self.gen_a.encode_sty(x_a)
        # h_b = self.gen_b.encode_cont(x_b)
        
        x_ab = self.gen_b.decode_cont(h_a)
        # h_c = torch.cat((h_b, h_a_sty), 1)
        # x_ba = self.gen_a.decode_recs(h_c)
        # self.train()
        return x_ab #, x_ba

    def __compute_kl(self, mu):
        # def _compute_kl(self, mu, sd):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def content_update(self, x_a, x_b, hyperparameters): #
        # encode
        self.content_opt.zero_grad()
        enc_a = self.gen_a.encode_cont(x_a)
        enc_b = self.gen_b.encode_cont(x_b)
        pred_fake = self.dis_content.forward(enc_a)
        pred_real = self.dis_content.forward(enc_b)
        loss_D = 0
        if hyperparameters['gan_type'] == 'lsgan':
            loss_D += torch.mean((pred_fake - 0)**2) + torch.mean((pred_real - 1)**2)
        elif hyperparameters['gan_type'] == 'nsgan':
            all0 = Variable(torch.zeros_like(pred_fake.data).cuda(self.gpuid), requires_grad=False)
            all1 = Variable(torch.ones_like(pred_real.data).cuda(self.gpuid), requires_grad=False)
            loss_D += torch.mean(F.binary_cross_entropy(F.sigmoid(pred_fake), all0) +
                                   F.binary_cross_entropy(F.sigmoid(pred_real), all1))
        else:
            assert 0, "Unsupported GAN type: {}".format(hyperparameters['gan_type'])
        loss_D.backward()
        nn.utils.clip_grad_norm_(self.dis_content.parameters(), 5)
        self.content_opt.step()

    def gen_update(self, x_a, x_b, hyperparameters):
        self.gen_opt.zero_grad()
        self.content_opt.zero_grad()
        # encode
        h_a = self.gen_a.encode_cont(x_a)
        h_b = self.gen_b.encode_cont(x_b)
        h_a_sty = self.gen_a.encode_sty(x_a)

        # add domain adverisal loss for generator 
        out_a = self.dis_content(h_a)
        out_b = self.dis_content(h_b)
        self.loss_ContentD = 0
        if hyperparameters['gan_type'] == 'lsgan':
            self.loss_ContentD += torch.mean((out_a - 0.5)**2) + torch.mean((out_b - 0.5)**2)
        elif hyperparameters['gan_type'] == 'nsgan':
            all1 = Variable(0.5 * torch.ones_like(out_b.data).cuda(self.gpuid), requires_grad=False)
            self.loss_ContentD += torch.mean(F.binary_cross_entropy(F.sigmoid(out_a), all1) +
                                   F.binary_cross_entropy(F.sigmoid(out_b), all1))
        else:
            assert 0, "Unsupported GAN type: {}".format(hyperparameters['gan_type'])

        # decode (within domain)
        h_a_cont = torch.cat((h_a, h_a_sty), 1)
        noise_a = torch.randn(h_a_cont.size()).cuda(h_a_cont.data.get_device())
        x_a_recon = self.gen_a.decode_recs(h_a_cont + noise_a)
        noise_b = torch.randn(h_b.size()).cuda(h_b.data.get_device())
        x_b_recon = self.gen_b.decode_cont(h_b + noise_b)

        # decode (cross domain)
        h_ba_cont = torch.cat((h_b, h_a_sty), 1)
        x_ba = self.gen_a.decode_recs(h_ba_cont + noise_a)
        x_ab = self.gen_b.decode_cont(h_a + noise_b)

        # encode again
        h_b_recon = self.gen_a.encode_cont(x_ba)
        h_b_sty_recon = self.gen_a.encode_sty(x_ba)

        h_a_recon = self.gen_b.encode_cont(x_ab)

        # decode again (if needed)
        h_a_cat_recs = torch.cat((h_a_recon, h_b_sty_recon), 1)

        x_aba = self.gen_a.decode_recs(h_a_cat_recs + noise_a) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode_cont(h_b_recon + noise_b) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_kl_a = self.__compute_kl(h_a)
        self.loss_gen_recon_kl_b = self.__compute_kl(h_b)
        self.loss_gen_recon_kl_sty = self.__compute_kl(h_a_sty)

        self.loss_gen_cyc_x_a = self.recon_criterion(x_aba, x_a) if x_aba is not None else 0
        self.loss_gen_cyc_x_b = self.recon_criterion(x_bab, x_b) if x_aba is not None else 0
        self.loss_gen_recon_kl_cyc_aba = self.__compute_kl(h_a_recon)
        self.loss_gen_recon_kl_cyc_bab = self.__compute_kl(h_b_recon)
        self.loss_gen_recon_kl_cyc_sty = self.__compute_kl(h_b_sty_recon)

        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)

        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0
        
        

        # add background guide loss
        self.loss_bgm = 0
        if hyperparameters['BGM'] != 0: 
            for index, weight in enumerate(self.BlurWeight):
                out_b = self.BlurNet[index](x_ba)
                out_real_b = self.BlurNet[index](x_b)
                out_a = self.BlurNet[index](x_ab)
                out_real_a = self.BlurNet[index](x_a)
                grad_loss_b = self.recon_criterion(out_b, out_real_b)
                grad_loss_a = self.recon_criterion(out_a, out_real_a)
                self.loss_bgm += weight * (grad_loss_a + grad_loss_b)
        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_b + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_sty + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_a + \
                              hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_aba + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_b + \
                              hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_bab + \
                              hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_sty + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b + \
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
        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []
        for i in range(x_a.size(0)):
            h_a = self.gen_a.encode_cont(x_a[i].unsqueeze(0))
            h_a_sty = self.gen_a.encode_sty(x_a[i].unsqueeze(0))
            h_b = self.gen_b.encode_cont(x_b[i].unsqueeze(0))

            h_ba_cont = torch.cat((h_b, h_a_sty), 1)

            h_aa_cont = torch.cat((h_a, h_a_sty), 1)

            x_a_recon.append(self.gen_a.decode_recs(h_aa_cont))
            x_b_recon.append(self.gen_b.decode_cont(h_b))

            x_ba.append(self.gen_a.decode_recs(h_ba_cont))
            x_ab.append(self.gen_b.decode_cont(h_a))
            
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba = torch.cat(x_ba)
        x_ab = torch.cat(x_ab)
        self.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba

    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()
        self.content_opt.zero_grad()
        # encode
        h_a = self.gen_a.encode_cont(x_a)
        h_a_sty = self.gen_a.encode_sty(x_a)
        h_b = self.gen_b.encode_cont(x_b)

        # # @ add content adversial
        out_a = self.dis_content(h_a)
        out_b = self.dis_content(h_b)
        self.loss_ContentD = 0
        if hyperparameters['gan_type'] == 'lsgan':
            self.loss_ContentD += torch.mean((out_a - 0)**2) + torch.mean((out_b - 1)**2)
        elif hyperparameters['gan_type'] == 'nsgan':
            all0 = Variable(torch.zeros_like(out_a.data).cuda(self.gpuid), requires_grad=False)
            all1 = Variable(torch.ones_like(out_b.data).cuda(self.gpuid), requires_grad=False)
            self.loss_ContentD += torch.mean(F.binary_cross_entropy(F.sigmoid(out_a), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out_b), all1))
        else:
            assert 0, "Unsupported GAN type: {}".format(hyperparameters['gan_type'])
        
        # decode (cross domain)
        h_cat = torch.cat((h_b, h_a_sty), 1)
        noise_b = torch.randn(h_cat.size()).cuda(h_cat.data.get_device())
        x_ba = self.gen_a.decode_recs(h_cat + noise_b)
        noise_a = torch.randn(h_a.size()).cuda(h_a.data.get_device())
        x_ab = self.gen_b.decode_cont(h_a + noise_a)
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)


        self.loss_dis_total = hyperparameters['gan_w'] * (self.loss_dis_a + self.loss_dis_b + self.loss_ContentD)
        self.loss_dis_total.backward()        
        nn.utils.clip_grad_norm_(self.dis_content.parameters(), 5) # dis_content update
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
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis_00188000")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b']) 
        
        # load discontent discriminator
        last_model_name = get_model_list(checkpoint_dir, "dis_Content")
        state_dict = torch.load(last_model_name)
        self.dis_content.load_state_dict(state_dict['dis_c'])


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
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'dis_c':self.dis_content.state_dict()}, dis_Con_name)

        #  opt state
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict(), \
                                                    'dis_content':self.content_opt.state_dict()}, opt_name)