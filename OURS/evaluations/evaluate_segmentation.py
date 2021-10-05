import torch
from models import ApperanceModelStyle, EncoderStyle, UnetFlowModel

import numpy as np
import os
from .helpers import compute_accuracy_brain, convert_to_image, color_seg

from utilities.util import load_network, load_numpy, save_numpy

from .constants import Constants
import pprint

@torch.no_grad()
def run_brain_evaluation(input_dir, output_dir, checkpoint_dir, base_name, base_path, dataset_name, cuda=True):
    K = Constants(dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(os.path.join(output_dir, 'images', 'train'))
        os.makedirs(os.path.join(output_dir, 'images', 'test'))
        os.makedirs(os.path.join(output_dir, 'atlas', 'train'))
        os.makedirs(os.path.join(output_dir, 'atlas', 'test'))
    
    encoder_style_k = EncoderStyle(128)
    encoder_style_k = load_network(encoder_style_k, 'Encoder_style_k', 'latest', checkpoint_dir)
    if cuda:
        encoder_style_k.cuda()
    encoder_style_k.eval()

    app_model = ApperanceModelStyle(16, 128)
    app_model = load_network(app_model, 'Appearance_Model', 'latest', checkpoint_dir)

    if cuda:
        app_model.cuda()
    app_model.eval()

    flow_model = UnetFlowModel([128, 160, 160])
    flow_model_name = 'UnetFlowModel'

    flow_model = load_network(flow_model, flow_model_name, 'latest', checkpoint_dir)
    if cuda:
        flow_model.cuda()
    flow_model.eval()

    if base_path == '':
        base_dir = input_dir
    else:
        base_dir = base_path
    # load the base image and labels
    base_img = load_numpy(os.path.join(base_dir, base_name + '.npy.gz')).astype(np.float32).transpose(2, 1, 0)
    base_label = np.rint(load_numpy(os.path.join(base_dir, base_name + '_seg.npy.gz'))).astype(np.float32).transpose(2, 1, 0)

    base_img = torch.from_numpy(base_img).unsqueeze(0).unsqueeze(0)
    base_label = torch.from_numpy(base_label).unsqueeze(0).unsqueeze(0)

    base_img = 2.0*base_img - 1.0
    if cuda:
        base_img = base_img.cuda()
        base_label = base_label.cuda()

    overall_accuracy = []
    classwise_accuracy = {}

    folder = 'test'
    for p in K.brain_tests:
        target_img = load_numpy(os.path.join(input_dir, p + '_MR.npy.gz')).astype(np.float32).transpose(2, 1, 0)
        target_label = np.rint(load_numpy(os.path.join(input_dir, p + '_MR_seg.npy.gz'))).astype(np.float32).transpose(2, 1, 0)

        target_img = torch.from_numpy(target_img).unsqueeze(0).unsqueeze(0)
        target_label = torch.from_numpy(target_label).unsqueeze(0).unsqueeze(0)
        target_img = 2.0*target_img -1.0
        if cuda:
            target_img = target_img.cuda()

        target_style = encoder_style_k(target_img)
        base_with_target_style = app_model(base_img, target_style)
        predicted_img, predicted_label, _ = flow_model(base_with_target_style, base_label, target_img)

        # save images
        convert_to_image(predicted_img).save(os.path.join(output_dir, 'images', folder, p + '_pred_img.png'))
        convert_to_image(target_img).save(os.path.join(output_dir, 'images', folder, p + '_target_img.png'))
        color_seg(predicted_label).save(os.path.join(output_dir, 'images', folder, p + '_pred_label.png'))
        color_seg(target_label).save(os.path.join(output_dir, 'images', folder, p + '_target_label.png'))



        predicted_label = predicted_label[0, 0].cpu().numpy()
        target_label    = target_label[0, 0].cpu().numpy()
        save_numpy(os.path.join(output_dir, 'atlas', folder, p + '_MR_seg.npy.gz'), predicted_label)
        
        overall, classwise = compute_accuracy_brain(predicted_label, target_label)
        overall_accuracy += overall

        print(np.unique(target_label))
        if len(classwise_accuracy) == 0:
            classwise_accuracy = classwise
        else:
            for k in classwise_accuracy:
                classwise_accuracy[k] += classwise[k]

    mean_overall_accuracy, std_overall_accracy = np.mean(overall_accuracy), np.std(overall_accuracy)
    mean_std_classwise_accuracy = {}

    for k in classwise_accuracy.keys():
        mean_std_classwise_accuracy[k] = np.mean(classwise_accuracy[k]), np.std(classwise_accuracy[k])
    
    with open(os.path.join(output_dir, 'brain_stats.txt'), 'a') as outfile:
        outfile.write('Overall dice scores: {} +- {}\n\n'.format(mean_overall_accuracy, std_overall_accracy))
        outfile.write('Overall classwise dice scores:\n')

        for k, v in mean_std_classwise_accuracy.items():
            outfile.write('{} : {} +- {}\n'.format(k, v[0], v[1]))

        outfile.write('\nTest case wise accuracies\n:')
        pprint.pprint(classwise_accuracy, stream=outfile)
    
@torch.no_grad()
def run_brain_evaluation_wo_appearance(input_dir, output_dir, checkpoint_dir, base_name, dataset_name, cuda=True):
    K = Constants(dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(os.path.join(output_dir, 'images', 'train'))
        os.makedirs(os.path.join(output_dir, 'images', 'test'))
    
    flow_model = UnetFlowModel([128, 160, 160])
    flow_model_name = 'UnetFlowModel'

    flow_model = load_network(flow_model, flow_model_name, 'latest', checkpoint_dir)
    if cuda:
        flow_model.cuda()
    flow_model.eval()

    # load the base image and labels
    base_img = load_numpy(os.path.join(input_dir, base_name + '.npy.gz')).astype(np.float32).transpose(2, 1, 0)
    base_label = np.rint(load_numpy(os.path.join(input_dir, base_name + '_seg.npy.gz'))).astype(np.float32).transpose(2, 1, 0)

    base_img = torch.from_numpy(base_img).unsqueeze(0).unsqueeze(0)
    base_label = torch.from_numpy(base_label).unsqueeze(0).unsqueeze(0)

    base_img = 2.0*base_img - 1.0
    if cuda:
        base_img = base_img.cuda()
        base_label = base_label.cuda()

    overall_accuracy = []
    classwise_accuracy = {}

    folder = 'test'
    for p in K.brain_tests:
        target_img = load_numpy(os.path.join(input_dir, p + '_MR.npy.gz')).astype(np.float32).transpose(2, 1, 0)
        target_label = np.rint(load_numpy(os.path.join(input_dir, p + '_MR_seg.npy.gz'))).astype(np.float32).transpose(2, 1, 0)

        target_img = torch.from_numpy(target_img).unsqueeze(0).unsqueeze(0)
        target_label = torch.from_numpy(target_label).unsqueeze(0).unsqueeze(0)
        target_img = 2.0*target_img -1.0
        if cuda:
            target_img = target_img.cuda()
        
        predicted_img, predicted_label, _ = flow_model(base_img, base_label, target_img)

        # save images
        convert_to_image(predicted_img).save(os.path.join(output_dir, 'images', folder, p + '_pred_img.png'))
        convert_to_image(target_img).save(os.path.join(output_dir, 'images', folder, p + '_target_img.png'))
        color_seg(predicted_label).save(os.path.join(output_dir, 'images', folder, p + '_pred_label.png'))
        color_seg(target_label).save(os.path.join(output_dir, 'images', folder, p + '_target_label.png'))



        predicted_label = predicted_label[0, 0].cpu().numpy()
        target_label    = target_label[0, 0].cpu().numpy()

        overall, classwise = compute_accuracy_brain(predicted_label, target_label)
        overall_accuracy += overall

        # save images



        print(np.unique(target_label))
        if len(classwise_accuracy) == 0:
            classwise_accuracy = classwise
        else:
            for k in classwise_accuracy:
                classwise_accuracy[k] += classwise[k]

    mean_overall_accuracy, std_overall_accracy = np.mean(overall_accuracy), np.std(overall_accuracy)
    mean_std_classwise_accuracy = {}

    for k in classwise_accuracy.keys():
        mean_std_classwise_accuracy[k] = np.mean(classwise_accuracy[k]), np.std(classwise_accuracy[k])
    
    with open(os.path.join(output_dir, 'brain_stats.txt'), 'a') as outfile:
        outfile.write('Overall dice scores: {} +- {}\n\n'.format(mean_overall_accuracy, std_overall_accracy))
        outfile.write('Overall classwise dice scores:\n')

        for k, v in mean_std_classwise_accuracy.items():
            outfile.write('{} : {} +- {}\n'.format(k, v[0], v[1]))

        outfile.write('\nTest case wise accuracies\n:')
        pprint.pprint(classwise_accuracy, stream=outfile)