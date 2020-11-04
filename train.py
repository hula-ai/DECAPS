import os
import warnings
import torch.backends.cudnn as cudnn
warnings.filterwarnings("ignore")
from datetime import datetime
from torch.utils.data import DataLoader
from decaps import CapsuleNet
from torch.optim import Adam
import numpy as np
from config import options
import torch
import torch.nn.functional as F
from utils.eval_utils import compute_accuracy, binary_cls_compute_metrics
from utils.logger_utils import Logger
import torch.nn as nn

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
theta_c = 0.5  # crop region with attention values higher than this
theta_d = 0.5  # drop region with attention values higher than this


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def train():
    global_step = 0
    best_loss = 100
    best_acc = 0
    best_auc = 0

    # Default Parameters
    beta = 1e-4

    for epoch in range(options.epochs):
        log_string('**' * 30)
        log_string('Training Epoch %03d, Learning Rate %g' % (epoch + 1, optimizer.param_groups[0]['lr']))
        capsule_net.train()

        margin_loss_, reg_loss_, total_loss_ = 0, 0, np.zeros(4)
        targets, predictions, predictions_crop, predictions_drop, predictions_combined = [], [], [], [], []

        # increments the margin for spread loss
        if options.loss_type == 'spread' and (epoch + 1) % options.n_eps_for_m == 0 and epoch != 0:
            capsule_loss.margin += options.m_delta
            capsule_loss.margin = min(capsule_loss.margin, options.m_max)
            log_string(' *------- Margin increased to {0:.1f}'.format(capsule_loss.margin))

        for batch_id, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            global_step += 1
            target_ohe = F.one_hot(target, options.num_classes)
            att_reg_loss, rec_loss = 0, 0

            optimizer.zero_grad()
            y_pred, x_reconst, output, attention_map, feats, attention_maps, out_vec_raw = capsule_net(data, target_ohe)
            cls_loss = capsule_loss(output, target)

            if options.add_decoder:
                rec_loss = reconst_loss(data, x_reconst)

            if options.attention_reg:
                feature_matrix = feats[np.arange(len(target)), :, target, :]
                att_reg_loss = l2_loss(feature_matrix, feature_center[target])
                # Update Feature Center
                feature_center[target] += beta * (feature_matrix.detach() - feature_center[target])

            total_loss = cls_loss + options.lambda_one * att_reg_loss + options.alpha * rec_loss
            total_loss.backward()
            optimizer.step()

            # Update Feature Center
            feature_center[target] += beta * (feature_matrix.detach() - feature_center[target])

            targets += [target_ohe]
            predictions += [y_pred]
            margin_loss_ += cls_loss.item()
            reg_loss_ += att_reg_loss.item()
            total_loss_[0] += total_loss.item()

            ##################################
            # Attention Cropping
            ##################################
            empty_map_count = 0
            one_nonzero_count = 0
            width_count = 0
            height_count = 0
            with torch.no_grad():
                crop_mask = F.upsample_bilinear(attention_map, size=(data.size(2), data.size(3))) > theta_c
                crop_images = []
                for batch_index in range(crop_mask.size(0)):
                    if torch.sum(crop_mask[batch_index]) == 0:
                        height_min, width_min = 0, 0
                        height_max, width_max = 200, 200
                        # print('0, batch: {}, map: {}'.format(batch_index, map_index))
                        empty_map_count += 1
                    else:
                        nonzero_indices = torch.nonzero(crop_mask[batch_index, 0, ...])
                        if nonzero_indices.size(0) == 1:
                            height_min, width_min = 0, 0
                            height_max, width_max = 200, 200
                            # print('1, batch: {}, map: {}'.format(batch_index, map_index))
                            one_nonzero_count += 1
                        else:
                            height_min = nonzero_indices[:, 0].min()
                            height_max = nonzero_indices[:, 0].max()
                            width_min = nonzero_indices[:, 1].min()
                            width_max = nonzero_indices[:, 1].max()
                        if width_min == width_max:
                            if width_min == 0:
                                width_max += 1
                            else:
                                width_min -= 1
                            width_count += 1
                        if height_min == height_max:
                            if height_min == 0:
                                height_max += 1
                            else:
                                height_min -= 1
                            height_count += 1
                    crop_images.append(F.upsample_bilinear(
                        data[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                        size=options.img_h))
                # print('Batch {} :  empty map: {},  one nonzero idx: {}, width_issue: {}, height_issue: {}'
                #       .format(i, empty_map_count, one_nonzero_count, width_count, height_count))

            crop_images = torch.cat(crop_images, dim=0).cuda()

            # crop images forward
            y_pred_crop, _, output_crop, _, _, _, _ = capsule_net(crop_images, target_ohe)
            loss = capsule_loss(output_crop, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            predictions_crop += [y_pred_crop.float()]
            total_loss_[1] += loss.item()

            ##################################
            # Attention Combining
            ##################################
            # final prediction
            output_combined = (output + output_crop) / 2
            _, y_pred_combined = output_combined.max(dim=1)
            y_pred_combined_ohe = F.one_hot(y_pred_combined, options.num_classes)

            loss = capsule_loss(output_combined, target)
            predictions_combined += [y_pred_combined_ohe]
            total_loss_[3] += loss.item()

            ##################################
            # Attention Dropping
            ##################################
            with torch.no_grad():
                drop_mask = F.upsample_bilinear(attention_map, size=(data.size(2), data.size(3))) <= theta_d
                drop_images = data * drop_mask.float()

            # drop images forward
            y_pred_drop, _, output_drop, _, _, _, _ = capsule_net(drop_images.cuda(), target_ohe)
            loss = capsule_loss(output_drop, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            predictions_drop += [y_pred_drop.float()]
            total_loss_[2] += loss.item()

            if (batch_id + 1) % options.disp_freq == 0:
                margin_loss_ /= options.disp_freq
                reg_loss_ /= options.disp_freq
                total_loss_ = total_loss_ / options.disp_freq
                train_acc = compute_accuracy(torch.cat(targets), torch.cat(predictions))
                train_acc_crop = compute_accuracy(torch.cat(predictions_crop), torch.cat(targets))
                train_acc_drop = compute_accuracy(torch.cat(predictions_drop), torch.cat(targets))
                train_acc_comb = compute_accuracy(torch.cat(predictions_combined), torch.cat(targets))

                log_string("epoch: {0}, step: {1}, (Raw): loss: {2:.4f} acc: {3:.02%}, "
                           "(Crop): loss: {4:.4f} acc: {5:.02%}, (Drop): loss: {6:.4f} acc: {7:.02%}, "
                           "(Comb): loss: {8:.4f} acc: {9:.02%}"
                           .format(epoch + 1, batch_id + 1,
                                   total_loss_[0], train_acc,
                                   total_loss_[1], train_acc_crop,
                                   total_loss_[2], train_acc_drop,
                                   total_loss_[3], train_acc_comb))
                info = {'loss/raw': total_loss_[0],
                        'loss/crop': total_loss_[1],
                        'loss/drop': total_loss_[2],
                        'loss/combined': total_loss_[3],
                        'accuracy/raw': train_acc,
                        'accuracy/crop': train_acc_crop,
                        'accuracy/drop': train_acc_drop,
                        'accuracy/comb': train_acc_comb}

                for tag, value in info.items():
                    train_logger.scalar_summary(tag, value, global_step)
                margin_loss_, reg_loss_, total_loss_ = 0, 0, np.zeros(4)
                targets, predictions, predictions_crop, predictions_drop, predictions_combined = [], [], [], [], []

            if (batch_id + 1) % options.val_freq == 0:
                log_string('--' * 30)
                log_string('Evaluating at step #{}'.format(global_step))
                best_loss, best_acc, best_auc = evaluate(best_loss=best_loss,
                                                         best_acc=best_acc,
                                                         best_auc=best_auc,
                                                         global_step=global_step)
                capsule_net.train()


@torch.no_grad()
def evaluate(**kwargs):
    best_loss = kwargs['best_loss']
    best_acc = kwargs['best_acc']
    best_auc = kwargs['best_auc']
    global_step = kwargs['global_step']

    capsule_net.eval()
    test_loss = np.zeros(3)
    targets, predictions_raw, predictions_crop, predictions_combined = [], [], [], []
    outputs_raw, outputs_crop, outputs_combined = [], [], []
    empty_map_count = 0
    one_nonzero_count = 0
    width_count = 0
    height_count = 0

    with torch.no_grad():
        for batch_id, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            target_ohe = F.one_hot(target, options.num_classes)
            y_pred_raw, x_reconst, output, attention_map, _, _, _ = capsule_net(data, target_ohe)
            loss = capsule_loss(output, target)
            targets += [target_ohe]
            outputs_raw += [output]
            predictions_raw += [y_pred_raw]
            test_loss[0] += loss

            ##################################
            # Object Localization and Refinement
            ##################################
            crop_mask = F.upsample_bilinear(attention_map, size=(data.size(2), data.size(3))) > theta_c
            crop_images = []
            for batch_index in range(crop_mask.size(0)):
                if torch.sum(crop_mask[batch_index]) == 0:
                    height_min, width_min = 0, 0
                    height_max, width_max = 200, 200
                    # print('0, batch: {}, map: {}'.format(batch_index, map_index))
                    empty_map_count += 1
                else:
                    nonzero_indices = torch.nonzero(crop_mask[batch_index, 0, ...])
                    if nonzero_indices.size(0) == 1:
                        height_min, width_min = 0, 0
                        height_max, width_max = 200, 200
                        # print('1, batch: {}, map: {}'.format(batch_index, map_index))
                        one_nonzero_count += 1
                    else:
                        height_min = nonzero_indices[:, 0].min()
                        height_max = nonzero_indices[:, 0].max()
                        width_min = nonzero_indices[:, 1].min()
                        width_max = nonzero_indices[:, 1].max()
                    if width_min == width_max:
                        if width_min == 0:
                            width_max += 1
                        else:
                            width_min -= 1
                        width_count += 1
                    if height_min == height_max:
                        if height_min == 0:
                            height_max += 1
                        else:
                            height_min -= 1
                        height_count += 1
                crop_images.append(F.upsample_bilinear(
                    data[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                    size=options.img_h))
            crop_images = torch.cat(crop_images, dim=0)

            y_pred_crop, _, output_crop, c_maps, _, _, out_vec_crop = capsule_net(crop_images, target_ohe)
            loss = capsule_loss(output_crop, target)
            predictions_crop += [y_pred_crop]
            outputs_crop += [output_crop]
            test_loss[1] += loss

            # final prediction
            # out_vec = (out_vec_raw + out_vec_crop) / 2
            # output_combined = (out_vec ** 2).sum(dim=-1) ** 0.5

            output_combined = (output + output_crop) / 2
            outputs_combined += [output_combined]
            _, y_pred_combined = output_combined.max(dim=1)
            y_pred_combined_ohe = F.one_hot(y_pred_combined, options.num_classes)

            test_loss[2] += capsule_loss(output_combined, target)
            predictions_combined += [y_pred_combined_ohe]

        test_loss /= (batch_id + 1)
        metrics_raw = binary_cls_compute_metrics(torch.cat(outputs_raw).cpu(), torch.cat(targets).cpu())
        metrics_crop = binary_cls_compute_metrics(torch.cat(outputs_crop).cpu(), torch.cat(targets).cpu())
        metrics_combined = binary_cls_compute_metrics(torch.cat(outputs_combined).cpu(), torch.cat(targets).cpu())

        # check for improvement
        loss_str, acc_str, auc_str = '', '', ''
        if test_loss[0] <= best_loss:
            loss_str, best_loss = '(improved)', test_loss[0]
        if metrics_combined['acc'] >= best_acc:
            acc_str, best_acc = '(improved)', metrics_combined['acc']
        if metrics_combined['auc'] >= best_auc:
            auc_str, best_auc = '(improved)', metrics_combined['auc']

        # display
        log_string(" - (Raw)      loss: {0:.4f}, acc: {1:.02%}, auc: {2:.02%}"
                   .format(test_loss[0], metrics_raw['acc'], metrics_raw['auc']))
        log_string(" - (Crop)     loss: {0:.4f}, acc: {1:.02%}, auc: {2:.02%}"
                   .format(test_loss[1], metrics_crop['acc'], metrics_crop['auc']))
        log_string(" - (Combined) loss: {0:.4f} {1}, acc: {2:.02%}{3}, auc: {4:.02%}{5}"
                   .format(test_loss[2], loss_str, metrics_combined['acc'], acc_str, metrics_combined['auc'], auc_str))
        # write to TensorBoard
        info = {'loss/raw': test_loss[0],
                'loss/crop': test_loss[0],
                'accuracy/raw': metrics_raw['acc'],
                'accuracy/crop': metrics_crop['acc'],
                'accuracy/combined': metrics_combined['acc'],
                'AUC/raw': metrics_raw['auc'],
                'AUC/crop': metrics_crop['auc'],
                'AUC/combined': metrics_combined['auc']}
        for tag, value in info.items():
            test_logger.scalar_summary(tag, value, global_step)

        # save checkpoint model
        state_dict = capsule_net.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        save_path = os.path.join(model_dir, '{}.ckpt'.format(global_step))
        torch.save({
            'global_step': global_step,
            'feature_center': feature_center.cpu(),
            'acc': metrics_combined['acc'],
            'auc': metrics_combined['auc'],
            'save_dir': model_dir,
            'state_dict': state_dict},
            save_path)
        log_string('Model saved at: {}'.format(save_path))
        log_string('--' * 30)
        return best_loss, best_acc, best_auc


if __name__ == '__main__':
    ##################################
    # Initialize saving directory
    ##################################
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    save_dir = options.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = os.path.join(save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(save_dir)

    LOG_FOUT = open(os.path.join(save_dir, 'log_train.txt'), 'w')
    LOG_FOUT.write(str(options) + '\n')

    model_dir = os.path.join(save_dir, 'models')
    logs_dir = os.path.join(save_dir, 'tf_logs')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # bkp of model def
    os.system('cp {}/decaps.py {}'.format(BASE_DIR, save_dir))
    # bkp of train procedure
    os.system('cp {}/train.py {}'.format(BASE_DIR, save_dir))
    os.system('cp {}/config.py {}'.format(BASE_DIR, save_dir))

    ##################################
    # Create the model
    ##################################
    capsule_net = CapsuleNet(options)
    log_string('Model Generated.')
    log_string("Number of trainable parameters: {}".format(sum(param.numel() for param in capsule_net.parameters())))

    # feature center of shape [200, 4, 64]
    feature_center = torch.zeros(options.num_classes, capsule_net.num_prim_map, options.digit_cap_dim).cuda()

    ##################################
    # Use cuda
    ##################################
    cudnn.benchmark = True
    capsule_net.cuda()
    # capsule_net = nn.DataParallel(capsule_net)

    ##################################
    # Load the trained model
    ##################################
    # ckpt = options.load_model_path
    # checkpoint = torch.load(ckpt)
    # state_dict = checkpoint['state_dict']
    #
    # # Load weights
    # capsule_net.load_state_dict(state_dict)
    # log_string('Model successfully loaded from {}'.format(ckpt))
    # if 'feature_center' in checkpoint:
    #     feature_center = checkpoint['feature_center'].to(torch.device("cuda"))
    #     log_string('feature_center loaded from {}'.format(ckpt))
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

    # Attention Regularization
    l2_loss = nn.MSELoss()

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
    elif options.data_name == 'rsna':
        from dataset.rsna_data import RSNADataSet as data
        os.system('cp {}/dataset/rsna_data.py {}'.format(BASE_DIR, save_dir))
    elif options.data_name == 'mias':
        from dataset.mias_data import MIASDataSet as data
        os.system('cp {}/dataset/mias_data.py {}'.format(BASE_DIR, save_dir))

    train_dataset = data(mode='train')
    train_loader = DataLoader(train_dataset, batch_size=options.batch_size,
                              shuffle=True, num_workers=options.workers, drop_last=False)
    test_dataset = data(mode='test')
    test_loader = DataLoader(test_dataset, batch_size=options.batch_size,
                             shuffle=False, num_workers=options.workers, drop_last=False)

    ##################################
    # TRAINING
    ##################################
    log_string('')
    log_string('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.
               format(options.epochs, options.batch_size, len(train_dataset), len(test_dataset)))
    train_logger = Logger(os.path.join(logs_dir, 'train'))
    test_logger = Logger(os.path.join(logs_dir, 'test'))

    train()
