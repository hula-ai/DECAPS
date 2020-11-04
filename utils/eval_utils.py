import torch
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def binary_cls_compute_metrics(outputs, targets):

    fpr_, tpr_, _ = roc_curve(targets[:, -1], outputs[:, -1])
    auc_ = auc(fpr_, tpr_)
    precision_, recall_, _ = precision_recall_curve(targets[:, -1], outputs[:, -1])

    metrics = {'fpr': fpr_,
               'tpr': tpr_,
               'auc': auc_,
               'precision': precision_,
               'recall': recall_,
               'acc': compute_accuracy(outputs, targets)}
    return metrics


def compute_accuracy(output, target):
    """
     Calculates the classification accuracy.
    :param target: Tensor of correct labels of size [batch_size, numClasses]
    :param output: Tensor of model predictions.
            It should have the same dimensions as target
    :return: prediction accuracy
    """
    num_samples = target.size(0)
    num_correct = torch.sum(torch.argmax(target, dim=1) == torch.argmax(output, dim=1))
    accuracy = num_correct.float() / num_samples
    return accuracy


def compute_metrics(outputs, targets):
    n_classes = outputs.shape[1]
    fpr, tpr, aucs, precision, recall = {}, {}, {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(targets[:, i], outputs[:, i])
        aucs[i] = auc(fpr[i], tpr[i])
        precision[i], recall[i], _ = precision_recall_curve(targets[:, i], outputs[:, i])
        fpr[i], tpr[i], precision[i], recall[i] = fpr[i].tolist(), tpr[i].tolist(), precision[i].tolist(), recall[
            i].tolist()

    metrics = {'fpr': fpr,
               'tpr': tpr,
               'aucs': aucs,
               'precision': precision,
               'recall': recall,
               'acc': compute_accuracy_multitask(outputs, targets)}

    return metrics


def compute_accuracy_multitask(output, target):
    """
     Calculates the classification accuracy.
    :param target: Tensor of correct labels of size [batch_size, numClasses]
    :param output: Predicted scores (logits) by the model.
            It should have the same dimensions as target
    :return: accuracy: average accuracy over the samples of the current batch for each condition
    """
    num_samples = target.size(0)
    correct_pred = target.eq(output.round().long())
    accuracy = torch.sum(correct_pred, dim=0)
    return accuracy.cpu().numpy() * (100. / num_samples)
