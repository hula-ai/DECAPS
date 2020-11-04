import os
import warnings
import torch.backends.cudnn as cudnn
warnings.filterwarnings("ignore")
from torch.utils.data import DataLoader
from decaps import CapsuleNet
from torch.optim import Adam
import numpy as np
from config import options
import torch
import torch.nn.functional as F
from utils.eval_utils import binary_cls_compute_metrics
import torch.nn as nn

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
theta_c = 0.5   # crop region with attention values higher than this
theta_d = 0.5   # drop region with attention values higher than this


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


@torch.no_grad()
def evaluate():

    capsule_net.eval()
    test_loss = np.zeros(4)
    targets, predictions_raw, predictions_crop, predictions_drop, predictions_combined = [], [], [], [], []
    outputs_raw, outputs_crop, outputs_combined = [], [], []

    with torch.no_grad():
        for batch_id, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            target_ohe = F.one_hot(target, options.num_classes)
            y_pred_raw, x_reconst, output, attention_map, _, c_maps, out_vec_raw = capsule_net(data, target_ohe)
            loss = capsule_loss(output, target)
            targets += [target_ohe]
            outputs_raw += [output]
            predictions_raw += [y_pred_raw]
            test_loss[0] += loss

            ##################################
            # Object Localization and Refinement
            ##################################
            bbox_coords = []
            upsampled_attention_map = F.upsample_bilinear(attention_map, size=(data.size(2), data.size(3)))
            crop_mask = upsampled_attention_map > theta_c
            crop_images = []
            for batch_index in range(crop_mask.size(0)):
                nonzero_indices = torch.nonzero(crop_mask[batch_index, 0, ...])

                height_min = nonzero_indices[:, 0].min()
                height_max = nonzero_indices[:, 0].max()
                width_min = nonzero_indices[:, 1].min()
                width_max = nonzero_indices[:, 1].max()

                bbox_coord = np.array([height_min, height_max, width_min, width_max])
                bbox_coords.append(bbox_coord)

                crop_images.append(F.upsample_bilinear(
                    data[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                    size=options.img_h))

            crop_images = torch.cat(crop_images, dim=0)

            y_pred_crop, _, output_crop, _, _, c_maps_crop, out_vec_crop = capsule_net(crop_images, target_ohe)
            loss = capsule_loss(output_crop, target)
            predictions_crop += [y_pred_crop]
            outputs_crop += [output_crop]
            test_loss[1] += loss

            # final prediction
            output_combined = (output + output_crop) / 2
            outputs_combined += [output_combined]
            y_pred_combined = output_combined.argmax(dim=1)
            y_pred_combined_ohe = F.one_hot(y_pred_combined, options.num_classes)

            test_loss[3] += capsule_loss(output_combined, target)
            predictions_combined += [y_pred_combined_ohe]

            ##################################
            # Attention Dropping
            ##################################
            drop_mask = F.upsample_bilinear(attention_map, size=(data.size(2), data.size(3))) <= theta_d
            drop_images = data * drop_mask.float()

            # drop images forward
            y_pred_drop, _, output_drop, _, _, c_maps_drop, out_vec_drop = capsule_net(drop_images.cuda(), target_ohe)
            loss = capsule_loss(output_crop, target)
            predictions_drop += [y_pred_drop]
            test_loss[2] += loss

        test_loss /= (batch_id + 1)
        metrics_raw = binary_cls_compute_metrics(torch.cat(outputs_raw).cpu(), torch.cat(targets).cpu())
        metrics_crop = binary_cls_compute_metrics(torch.cat(outputs_crop).cpu(), torch.cat(targets).cpu())
        metrics_combined = binary_cls_compute_metrics(torch.cat(outputs_combined).cpu(), torch.cat(targets).cpu())

        # display
        log_string(" - (Raw)      loss: {0:.4f}, acc: {1:.02%}, auc: {2:.02%}"
                   .format(test_loss[0], metrics_raw['acc'], metrics_raw['auc']))
        log_string(" - (Crop)     loss: {0:.4f}, acc: {1:.02%}, auc: {2:.02%}"
                   .format(test_loss[1], metrics_crop['acc'], metrics_crop['auc']))
        log_string(" - (Combined) loss: {0:.4f}, acc: {1:.02%}, auc: {2:.02%}"
                   .format(test_loss[2], metrics_combined['acc'], metrics_combined['auc']))


if __name__ == '__main__':
    ##################################
    # Initialize saving directory
    ##################################
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    iter_num = options.load_model_path.split('/')[-1].split('.')[0]

    save_dir = os.path.dirname(os.path.dirname(options.load_model_path))
    img_dir = os.path.join(save_dir, 'imgs')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    viz_dir = os.path.join(img_dir, iter_num+'_crop_{}'.format(theta_c))
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)

    LOG_FOUT = open(os.path.join(save_dir, 'log_inference.txt'), 'w')
    LOG_FOUT.write(str(options) + '\n')

    # bkp of inference
    os.system('cp {}/inference.py {}'.format(BASE_DIR, save_dir))

    ##################################
    # Create the model
    ##################################
    capsule_net = CapsuleNet(options)
    log_string('Model Generated.')
    log_string("Number of trainable parameters: {}".format(sum(param.numel() for param in capsule_net.parameters())))

    ##################################
    # Use cuda
    ##################################
    cudnn.benchmark = True
    capsule_net.cuda()
    capsule_net = nn.DataParallel(capsule_net)

    ##################################
    # Load the trained model
    ##################################
    ckpt = options.load_model_path
    checkpoint = torch.load(ckpt)
    state_dict = checkpoint['state_dict']

    # Load weights
    capsule_net.load_state_dict(state_dict)
    log_string('Model successfully loaded from {}'.format(ckpt))
    if 'feature_center' in checkpoint:
        feature_center = checkpoint['feature_center'].to(torch.device("cuda"))
        log_string('feature_center loaded from {}'.format(ckpt))

    ##################################
    # Loss and Optimizer
    ##################################
    if options.loss_type == 'margin':
        from utils.loss_utils import MarginLoss

        capsule_loss = MarginLoss(options)
    elif options.loss_type == 'spread':
        from utils.loss_utils import SpreadLoss

        capsule_loss = SpreadLoss(options)
    elif options.loss_type == 'cross-entropy':
        capsule_loss = nn.CrossEntropyLoss()

    if options.add_decoder:
        from utils.loss_utils import ReconstructionLoss
        reconst_loss = ReconstructionLoss()

    optimizer = Adam(capsule_net.parameters(), lr=options.lr, betas=(options.beta1, 0.999))
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    ##################################
    # Load dataset
    ##################################
    if options.data_name == 'mnist':
        from dataset.mnist import MNIST as data
        os.system('cp {}/dataset/mnist.py {}'.format(BASE_DIR, save_dir))
    elif options.data_name == 'fashion_mnist':
        from dataset.fashion_mnist import FashionMNIST as data
        os.system('cp {}/dataset/fashion_mnist.py {}'.format(BASE_DIR, save_dir))
    elif options.data_name == 't_mnist':
        from dataset.mnist_translate import MNIST as data
        os.system('cp {}/dataset/mnist_translate.py {}'.format(BASE_DIR, save_dir))
    elif options.data_name == 'c_mnist':
        from dataset.mnist_clutter import MNIST as data
        os.system('cp {}/dataset/mnist_clutter.py {}'.format(BASE_DIR, save_dir))
    elif options.data_name == 'cub':
        from dataset.dataset_CUB import CUB as data
        os.system('cp {}/dataset/dataset_CUB.py {}'.format(BASE_DIR, save_dir))
    elif options.data_name == 'chexpert':
        from dataset.chexpert_dataset import CheXpertDataSet as data
        os.system('cp {}/dataset/chexpert_dataset.py {}'.format(BASE_DIR, save_dir))

    test_dataset = data(mode='test')
    test_loader = DataLoader(test_dataset, batch_size=options.batch_size,
                             shuffle=False, num_workers=options.workers, drop_last=False)
    ##################################
    # TESTING
    ##################################
    log_string('')
    log_string('Start Testing')

    evaluate()
