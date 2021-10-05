from utilities import iter_counter
from torch.utils.data import dataloader
from trainers import AETrainer, StyleEncoderTrainer, AppearanceModelTrainer, AllTrainer, TrainerFlowAAE, TrainerFlowGAN

import utilities.util as util
from collections import OrderedDict
from utilities import IterationCounter, Visualizer

from dataset_loaders import MRIDataset, MRIDatasetRandom, MRIDatasetClass, MRIDatasetClassBaseAtlas, MRIDatasetRandomBaseAtlas, FlowSampler

from torch.utils.data import DataLoader
from evaluations.evaluate_segmentation import run_brain_evaluation

import torch
import os
import numpy as np

from collections import namedtuple

from evaluations.constants import Constants

import argparse

K = Constants()
KOLORS = torch.from_numpy(K.colors).cuda()

def color_seg(img):
    img = torch.round(img).to(torch.int64)

    seg = KOLORS[img]
    seg = seg.permute(0, 3, 1, 2)
    return seg


def create_paths(opt):
    # create checkpoints directory
    util.ensure_dir(opt.checkpoints_dir)
    util.ensure_dir(os.path.join(opt.checkpoints_dir, 'models'))
    util.ensure_dir(os.path.join(opt.checkpoints_dir, 'tf_logs'))

    util.ensure_dir(opt.checkpoints_dir_pretrained)
    util.ensure_dir(os.path.join(opt.checkpoints_dir_pretrained, 'models'))
    util.ensure_dir(os.path.join(opt.checkpoints_dir_pretrained, 'tf_logs'))


def train_ae(opt):
    dataset = MRIDatasetRandom(opt.dataroot, opt.out_shape, dataset_name = opt.dataset_name)

    batch_size = opt.batch_size

    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=opt.num_workers)

    len_dataset = len(dataset)

    # create trainer to train the models
    trainer = AETrainer(opt)

    # visualizer for tensorboard summary
    visualizer = Visualizer(opt.checkpoints_dir_pretrained)

    # create tool for counting iterations
    iter_counter = IterationCounter(opt, len_dataset)

    # update batch_size
    iter_counter.set_batch_size(batch_size)

    print("Length of dataset: ", len_dataset)

    use_gpu = len(opt.gpu_ids) != 0

    for epoch in iter_counter.training_epochs():

        if epoch > iter_counter.total_epochs//2:
            lr = 2.0*opt.lr*(iter_counter.total_epochs + 1.0 - epoch)/(iter_counter.total_epochs + 2.0)
            trainer.update_learning_rate(lr)

        iter_counter.record_epoch_start(epoch)

        for i, (base_img, base_label, real_img) in enumerate(dataloader, start=iter_counter.epoch_iter):
            iter_counter.record_one_iteration()
            if use_gpu:
                real_img = real_img.cuda()
                base_img = base_img.cuda()
                base_label = base_label.cuda()
            
            trainer.run_step(base_img, base_label, real_img)

            # Visualizations
            if iter_counter.needs_printing():
                losses = trainer.get_latest_losses()
                visualizer.print_current_errors(epoch, iter_counter.epoch_iter, losses, iter_counter.time_per_iter)
                visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)
            
            if iter_counter.needs_displaying():
                visuals = OrderedDict([
                    ('synthesized_imgs', trainer.predicted_img[0][0][56:72].unsqueeze(1)),
                    ('synthesized_labels', color_seg(trainer.predicted_label[0][0][56:72])),
                    ('base_imgs', base_img[0][0][56:72].unsqueeze(1)),
                    ('base_labels', color_seg(base_label[0][0][56:72])),
                    ('real_img', real_img[0][0][56:72].unsqueeze(1))
                ])

                visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

            if iter_counter.needs_saving():
                print('saving the latest model (epoch %d, total_steps %d)' %
                    (epoch, iter_counter.total_steps_so_far))
                trainer.save('latest')
                iter_counter.record_current_iter()

            if opt.max_iterations < iter_counter.total_steps_so_far:
                break

        iter_counter.record_epoch_end()

        if epoch % opt.save_epoch_freq == 0 or epoch == iter_counter.total_epochs:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            trainer.save(epoch)
    
    print('Training was finished successfully')

def train_style_encoder_moco(opt):
    dataset = MRIDatasetRandom(opt.dataroot, opt.out_shape, dataset_name=opt.dataset_name)
    batch_size = opt.batch_size

    dataloder = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=opt.num_workers, drop_last=True, pin_memory=True)
    len_dataset = len(dataset)

    trainer = StyleEncoderTrainer(opt)

    # visualizer for tensorboard summary
    visualizer = Visualizer(opt.checkpoints_dir_pretrained)
    
    # create tool for counting iterations
    iter_counter = IterationCounter(opt, len_dataset)

    # update batch_size
    iter_counter.set_batch_size(batch_size)

    print("Length of dataset: ", len_dataset)

    use_gpu = len(opt.gpu_ids) != 0

    for epoch in iter_counter.training_epochs():
        if epoch > iter_counter.total_epochs//2:
            lr = 2.0*opt.lr*(iter_counter.total_epochs + 1.0 - epoch)/(iter_counter.total_epochs + 2.0)
            trainer.update_learning_rate(lr)

        iter_counter.record_epoch_start(epoch)

        for i, (img_q, _, img_k) in enumerate(dataloder, start=iter_counter.epoch_iter):
            iter_counter.record_one_iteration()

            if use_gpu:
                img_q = img_q.cuda(non_blocking=True)
                img_k = img_k.cuda(non_blocking=True)

            trainer.run_step(img_q, img_k)

            if iter_counter.needs_printing():
                losses = trainer.get_latest_losses()
                visualizer.print_current_errors(epoch, iter_counter.epoch_iter, losses, iter_counter.time_per_iter)
                visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)
            
            if iter_counter.needs_displaying():
                visuals = OrderedDict([
                    ('img_q', img_q[0][0][56:72].unsqueeze(1)),
                    ('img_k', img_k[0][0][56:72].unsqueeze(1)),
                    ('q_styled_img_k', trainer.imgk[0][0][56:72].unsqueeze(1))
                ])
                visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

            if iter_counter.needs_saving():
                print('saving the latest model (epoch %d, total_steps %d)' %
                    (epoch, iter_counter.total_steps_so_far))
                trainer.save('latest')
                iter_counter.record_current_iter()

            if opt.max_iterations < iter_counter.total_steps_so_far:
                break
        
        iter_counter.record_epoch_end()
        if epoch % opt.save_epoch_freq == 0 or epoch == iter_counter.total_epochs:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            trainer.save(epoch)
    
    print('Training was finished successfully')

def train_appearance_model(opt):
    dataset = MRIDatasetRandom(opt.dataroot, opt.out_shape, dataset_name = opt.dataset_name)

    batch_size = opt.batch_size

    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=opt.num_workers)

    len_dataset = len(dataset)

    # create trainer to train the models
    trainer = AppearanceModelTrainer(opt)

    # visualizer for tensorboard summary
    visualizer = Visualizer(opt.checkpoints_dir_pretrained)

    # create tool for counting iterations
    iter_counter = IterationCounter(opt, len_dataset)

    # update batch_size
    iter_counter.set_batch_size(batch_size)

    print("Length of dataset: ", len_dataset)

    use_gpu = len(opt.gpu_ids) != 0

    for epoch in iter_counter.training_epochs():

        if epoch > iter_counter.total_epochs//2:
            lr = 2.0*opt.lr*(iter_counter.total_epochs + 1.0 - epoch)/(iter_counter.total_epochs + 2.0)
            trainer.update_learning_rate(lr)

        iter_counter.record_epoch_start(epoch)

        for i, (img_x, _, img_y) in enumerate(dataloader, start=iter_counter.epoch_iter):
            iter_counter.record_one_iteration()

            if use_gpu:
                img_x = img_x.cuda()
                img_y = img_y.cuda()
            
            trainer.run_step(img_x, img_y)

            # Visualizations
            if iter_counter.needs_printing():
                losses = trainer.get_latest_losses()
                visualizer.print_current_errors(epoch, iter_counter.epoch_iter, losses, iter_counter.time_per_iter)
                visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)
            
            if iter_counter.needs_displaying():
                visuals = OrderedDict([
                    ('img_x_with_style_y', trainer.x_with_style_y[0][0][56:72].unsqueeze(1)),
                    ('img_y_with_style_x', trainer.y_with_style_x[0][0][56:72].unsqueeze(1)),
                    ('img_x', img_x[0][0][56:72].unsqueeze(1)),
                    ('img_y', img_y[0][0][56:72].unsqueeze(1))
                ])

                visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

            if iter_counter.needs_saving():
                print('saving the latest model (epoch %d, total_steps %d)' %
                    (epoch, iter_counter.total_steps_so_far))
                trainer.save('latest')
                iter_counter.record_current_iter()

            if opt.max_iterations < iter_counter.total_steps_so_far:
                break

        iter_counter.record_epoch_end()

        if epoch % opt.save_epoch_freq == 0 or epoch == iter_counter.total_epochs:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            trainer.save(epoch)
        
    print('Training was finished successfully')


def train_end_to_end(opt):
    #dataset = MRIDatasetClassBaseAtlas(opt.dataroot, opt.base_imgs) for candi
    dataset  = MRIDatasetRandomBaseAtlas(opt.dataroot, opt.base_imgs, base_path=opt.base_imgs_path, dataset_name = opt.dataset_name)

    batch_size = opt.batch_size
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=opt.num_workers,  drop_last=True, pin_memory=True)

    len_dataset = len(dataset)

    trainer = AllTrainer(opt)

    visualizer = Visualizer(opt.checkpoints_dir)

    iter_counter = IterationCounter(opt, len_dataset)
    
    # update batch_size
    iter_counter.set_batch_size(batch_size)

    print("Length of dataset: ", len_dataset)

    use_gpu = len(opt.gpu_ids) != 0

    lr = opt.lr

    for epoch in iter_counter.training_epochs():
        # if epoch > iter_counter.total_epochs//2:
        #     lr = 2.0*opt.lr*(iter_counter.total_epochs + 1.0 - epoch)/(iter_counter.total_epochs + 2.0)
        #     lr = lr/2.0
        #     trainer.update_learning_rate(lr)
        
        # print('learning rate: ', lr)
        iter_counter.record_epoch_start(epoch)
        
        # no need for real_segmentation
        for i, (base_img, base_label, real_img, real_img_k) in enumerate(dataloader, start=iter_counter.epoch_iter):
            iter_counter.record_one_iteration()

            if use_gpu:
                base_img, base_label = base_img.cuda(non_blocking=True), base_label.cuda(non_blocking=True)
                real_img, real_img_k = real_img.cuda(non_blocking=True), real_img_k.cuda(non_blocking=True)

            # trainer.run_step(base_img, base_label, real_img, real_img_k)
            
            trainer.run_step_no_base(base_img, base_label, real_img, real_img_k)

            # Visualizations
            if iter_counter.needs_printing():
                losses = trainer.get_latest_losses()
                visualizer.print_current_errors(epoch, iter_counter.epoch_iter, losses, iter_counter.time_per_iter)
                visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)
            
            if iter_counter.needs_displaying():
                visuals = OrderedDict([
                    ('synthesized_imgs', trainer.predicted_img[0][0][56:72].unsqueeze(1)),
                    ('base_labels', color_seg(base_label[0][0][56:72])),
                    ('synthesized_labels', color_seg(trainer.predicted_label[0][0][56:72])),
                    ('base_imgs', base_img[0][0][56:72].unsqueeze(1)),
                    ('real_img', real_img_k[0][0][56:72].unsqueeze(1)),
                    ('base_img_target_style', trainer.base_img_with_target_style[0][0][56:72].unsqueeze(1))
                ])

                visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

            if iter_counter.needs_saving():
                print('saving the latest model (epoch %d, total_steps %d)' %
                    (epoch, iter_counter.total_steps_so_far))
                trainer.save('latest')
                iter_counter.record_current_iter()

                #running test evaluations
                run_brain_evaluation(opt.dataroot, '../seg_evaluations/'+ opt.dataset_name + opt.checkpoints_dir, opt.checkpoints_dir,  base_name=opt.base_imgs[0], base_path=opt.base_imgs_path, dataset_name=opt.dataset_name)
        
            if opt.max_iterations < iter_counter.total_steps_so_far:
                break
        
        iter_counter.record_epoch_end()

        if epoch % opt.save_epoch_freq == 0 or epoch == iter_counter.total_epochs:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            trainer.save(epoch)
    
    print('Training was finished successfully')

def train_aae(opt):
    dataset = FlowSampler(opt.dataroot_flow, opt.dataroot, opt.base_imgs)
    batch_size = opt.batch_size
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=opt.num_workers)

    len_dataset = len(dataset)

    # create trainer to train the models
    trainer = TrainerFlowAAE(opt)

    # visualizer for tensorboard summary
    visualizer = Visualizer(opt.checkpoints_dir_pretrained)

    # create tool for counting iterations
    iter_counter = IterationCounter(opt, len_dataset)

    # update batch_size
    iter_counter.set_batch_size(batch_size)

    print("Length of dataset: ", len_dataset)

    use_gpu = len(opt.gpu_ids) != 0

    for epoch in iter_counter.training_epochs():

        if epoch > iter_counter.total_epochs//2:
            lr = 2.0*opt.lr*(iter_counter.total_epochs + 1.0 - epoch)/(iter_counter.total_epochs + 2.0)
            trainer.update_learning_rate(lr)

        iter_counter.record_epoch_start(epoch)

        for i, (flow, base_img, base_label) in enumerate(dataloader, start=iter_counter.epoch_iter):
            iter_counter.record_one_iteration()
            if use_gpu:
                flow = flow.cuda()
            
            trainer.run_step(base_img, base_label, flow)

            # Visualizations
            if iter_counter.needs_printing():
                losses = trainer.get_latest_losses()
                visualizer.print_current_errors(epoch, iter_counter.epoch_iter, losses, iter_counter.time_per_iter)
                visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)
            
            if iter_counter.needs_displaying():
                visuals = OrderedDict([
                    ('real_imgs', trainer.real_img[0][0][56:72].unsqueeze(1)),
                    ('real_labels', color_seg(trainer.real_label[0][0][56:72])),
                    ('fake_imgs', trainer.fake_img[0][0][56:72].unsqueeze(1)),
                    ('fake_labels', color_seg(trainer.fake_label[0][0][56:72]))
                ])

                visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

            if iter_counter.needs_saving():
                print('saving the latest model (epoch %d, total_steps %d)' %
                    (epoch, iter_counter.total_steps_so_far))
                trainer.save('latest')
                iter_counter.record_current_iter()
        
            if opt.max_iterations < iter_counter.total_steps_so_far:
                break
        
        iter_counter.record_epoch_end()

        if epoch % opt.save_epoch_freq == 0 or epoch == iter_counter.total_epochs:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            trainer.save(epoch)
    
    print('Training was finished successfully')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train one-shot')
    parser.add_argument('--ngpus', default=1, type=int, help='Number of GPUS to train on')
    parser.add_argument('--dataroot', default='', type=str, help='Dataset Directory')
    parser.add_argument('--dataroot_flow', default='', type=str, help='Flow Fields Directory')
    parser.add_argument('--base_imgs', default='BPDwoPsy_049_MR', type=str, help='Base Image to Choose')
    parser.add_argument('--base_imgs_path', default='', type=str, help='path to base images')
    parser.add_argument('--nepochs', default=1, type=int, help="Number of epochs to run")
    parser.add_argument('--batch_size', default=1, type=int, help='batch size to use')
    parser.add_argument('--num_workers', default=16, type=int, help='Number of dataloader workers')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--beta1', default=0.9)
    parser.add_argument('--beta2', default=0.999)
    parser.add_argument('--save_epoch_freq', default=1, type=int, help='Frequency of saved model checkpoints')
    parser.add_argument('--save_latest_freq', default=1000, type=int, help='Frequency of saving to the latest model')
    parser.add_argument('--print_freq', default=10, type=int, help='Printing frequency of training logs to the output terminal')
    parser.add_argument('--display_freq', default=500, type=int, help='Display frequency of Images on tensorbord summary')
    parser.add_argument('--continue_train', action='store_true')
    parser.add_argument('--tf_log', action='store_false')
    parser.add_argument('--style_code_dim', type=int, default=128)
    parser.add_argument('--gan_type', type=str, default='lsgan')
    parser.add_argument('--out_shape', default=[128, 160, 160])
    parser.add_argument('--lambda_grad', default=0.3, type=float)
    parser.add_argument('--lambda_grad_img', default=10.0, type=float)
    parser.add_argument('--lambda_app', default=2.0, type=float)
    parser.add_argument('--lambda_struct', default=1.0, type=float)
    parser.add_argument('--lambda_gan', default=0.5)
    parser.add_argument('--lambda_style_consistency',default= 1.0)
    parser.add_argument('--lambda_style_identity', default=1.0)
    parser.add_argument('--lambda_moco', default=0.5)
    parser.add_argument('--lambda_aae_l1', default=1.0)

    ### sensitivity test
    parser.add_argument('--final_lambda_1', default=5.0) # check values of 1.0, 10.0 with defalut
    parser.add_argument('--final_lambda_2', default=1.0) # check values of 2.0, 5.0 with default
    parser.add_argument('--final_lambda_3', default=0.1) # no change
    parser.add_argument('--final_lambda_reg', default=0.1) # check 0.2, 0.5


    parser.add_argument('--use_pretrain', action='store_false')
    parser.add_argument('--n_style_keys', type=int, default=1024)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--nf', type=int, default=16)
    parser.add_argument('--checkpoints_dir', default='./oasis_checkpoints/', type=str, help='checkpoint directory')
    parser.add_argument('--checkpoints_dir_pretrained', default='./oasis_checkpoints_pretrained', type=str, help='pretrained models checkpoint directory')
    parser.add_argument('--train_mode', default='end_to_end', type=str, help='training mode')
    parser.add_argument('--max_iterations', default=1000000)
    args = parser.parse_args()
    
    ngpus = list(range(args.ngpus))
    args.gpu_ids = ngpus

    args.base_imgs = args.base_imgs.split(',')

    gpustr = ",".join([str(i) for i in ngpus])

    os.environ["CUDA_VISIBLE_DEVICES"]=gpustr
    os.environ["KMP_WARNINGS"] = "FALSE"

    create_paths(args)
    train_mode = args.train_mode

    args.dataset_name = 'oasis' if 'OASIS' in args.dataroot else 'CANDIShare'

    if train_mode == 'ae':
        # args.nepochs = 600
        # args.batch_size = 16
        args.save_epoch_freq = 40
        train_ae(args)
    
    elif train_mode == 'style_moco':
        # args.nepochs = 200
        # args.batch_size = 32
        args.save_epoch_freq = 20
        train_style_encoder_moco(args)
    
    elif train_mode == 'appearance_only':
        # args.nepochs = 200
        # args.batch_size = 16
        args.save_epoch_freq = 20
        train_appearance_model(args)
    
    elif train_mode == 'aae':
        # args.nepochs = 600
        # args.batch_size = 2
        args.save_epoch_freq = 40
        train_aae(args)
    else:
        # args.nepochs = 600
        # args.batch_size = 2
        args.save_epoch_freq = 40
        train_end_to_end(args)

"""
1. Train FlowModel without Appearance Model.
python train.py --ngpus 1  --batch_size 4 --checkpoints_dir_pretrained ./checkpoints_pretrained --dataroot ../CANDIShare_clean_gz --train_mode ae --nepochs 10

2. Train StyleEncoder
python train.py --ngpus 1 --batch_size 16 --checkpoints_dir_pretrained ./checkpoints_pretrained --dataroot ../CANDIShare_StyleAug_gz --train_mode style_moco --nepochs 10

3. Train Appearance Model
python train.py --ngpus 1 --batch_size 1 --checkpoints_dir_pretrained ./checkpoints_pretrained --dataroot ../CANDIShare_clean_gz --train_mode appearance_only --nepochs 10

4. Train Adversarial Autoencoder Flow
python train.py --ngpus 1 --batch_size 2 --checkpoints_dir_pretrained ./checkpoints_pretrained --train_mode aae --nepochs 600

5. Train End to End
python train.py --ngpus 1 --batch_size 1 --checkpoints_dir ./checkpoints --dataroot ../CANDIShare_StyleAug_gz --train_mode end_to_end --nepochs 4



Steps:
a. First train Unet based flow model by running 1. This will be used to generate dataset ../CANDIShare_StyleAug_gz having spatial transformations of origianl images.

b. python generate_data.py

c. Pre-train style-encoder by running 2. This will pre-train our style encoder using volumetric contrastive loss

d. Train end-to-end by running 5. This will train Appearance Model, Style Encoder and Flow Model end to end using pre-trained style-encoder. set --use_pretrain to False
   for training Style Encoder from scratch

e. Generate Flow Fields in the folder ../FlowFields using trained end to end model by running the following command:
   python generate_flow.py

f. Train Flow Adversarial Autoencoder by running 4.


Evaluation Script:

All evaluation scripts used to generate plots and compute dice score are included in the folder evaluations. To run a particular evaluation, run the following command:

python run_evaluations opt
"""
