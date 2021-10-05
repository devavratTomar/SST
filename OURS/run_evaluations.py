from evaluations.figures import generate_main_figure
from evaluations.evaluate_segmentation import run_brain_evaluation, run_brain_evaluation_wo_appearance
from evaluations.style_flow_analysis import generate_some_styles_flows, flow_latent_walk
from evaluations.tnse_viz import generate_tsne_viz
from evaluations.helpers import evaluate_seg
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"]='1'

if __name__ == '__main__':
    if sys.argv[1] == 'figures':
        # generate_main_figure('../CANDIShare_clean_gz/', '../figures_paper/', './checkpoints')
        generate_main_figure('../OASIS_clean/', '../figures_paper/', './oasis_checkpoints')
        
    if sys.argv[1] == 'evaluate_tests':
        # run_brain_evaluation('../CANDIShare_clean_gz/', '../seg_evaluations', './checkpoints')
        run_brain_evaluation('../OASIS_clean/', '../seg_evaluations', './oasis_checkpoints',  base_name='1750_MR', base_path='', dataset_name='oasis')

    if sys.argv[1] == 'evaluate_flow':
        # run_brain_evaluation_wo_appearance('../CANDIShare_clean_gz/', '../seg_evaluations_flow', './checkpoints_pretrained')
        run_brain_evaluation_wo_appearance('../OASIS_clean/', '../seg_evaluations_flow', './oasis_checkpoints_pretrained')

    if sys.argv[1] == 'tsne':
        # generate_tsne_viz('../CANDIShare_clean_gz', '../tsne_plots', './checkpoints', False)
        generate_tsne_viz('../OASIS_clean', '../tsne_plots', './oasis_checkpoints_pretrained', True)

    if sys.argv[1] == 'evaluate_registration':
        # evaluate_seg('../CANDIShare_clean_gz', '../registration_visuals', ['../results/VoxelMorph', '../results/MABMIS', '../results/ours_registration'])
        evaluate_seg('../OASIS_clean', '../registration_visuals', ['../results/VoxelMorph', '../results/MABMIS', '../results/ours_registration'])

    if sys.argv[1] == 'style_flow_figs':
        # generate_some_styles_flows('../CANDIShare_clean_gz', '../FlowFields', '../style_flow_figs', './checkpoints')
        generate_some_styles_flows('../OASIS_clean', '../FlowFields', '../style_flow_figs', './oasis_checkpoints')
        
    if sys.argv[1] == 'evaluate_sota':
        # evaluate_seg('../CANDIShare_clean_gz', '../sota_baseline_visuals', ['../results/MABMIS', '../results/VoxelMorph', '../results/bayes', '../results/data_aug_results', '../results/sup_gt_results_unet', '../results/our_results_unet'])
        evaluate_seg('../OASIS_clean', '../sota_baseline_visuals/oasis', ['../results/oasis/MABMIS', '../results/oasis/VoxelMorph', '../results/oasis/bayes',
        '../results/oasis/data_aug_results', '../results/oasis/sup_gt_results_unet', '../results/oasis/our_results_unet'])

        # evaluate_seg('../OASIS_clean', '../sota_baseline_visuals/oasis', ['../results/oasis/sup_gt_results_unet'])
        

    if sys.argv[1] == 'style_flow_walk':
        # flow_latent_walk('../CANDIShare_clean_gz', '../LatentWalk', '../FlowFields', './checkpoints', './checkpoints_pretrained', base_img_path='BPDwoPsy_049_MR', nstyles=6, nflows=10, n_experiments=10)
        flow_latent_walk('../OASIS_clean', '../LatentWalk', '../FlowFields', './oasis_checkpoints', './oasis_checkpoints_pretrained', base_img_path='1750_MR', nstyles=6, nflows=10, n_experiments=10)