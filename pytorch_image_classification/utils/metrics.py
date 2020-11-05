import torch
import torch.nn as nn
import torch.nn.functional as F

# accuracy for model with data augmentation, use another function to calculate the pure accuracy
def compute_accuracy(config, outputs, targets, augmentation, topk=(1, )):
    if augmentation:
        if config.augmentation.use_mixup or config.augmentation.use_cutmix:
            targets1, targets2, lam = targets
            accs1 = accuracy(outputs, targets1, topk)
            accs2 = accuracy(outputs, targets2, topk)
            accs = tuple([
                lam * acc1 + (1 - lam) * acc2
                for acc1, acc2 in zip(accs1, accs2)
            ])
        elif config.augmentation.use_ricap:
            weights = []
            accs_all = []
            for labels, weight in zip(*targets):
                weights.append(weight)
                accs_all.append(accuracy(outputs, labels, topk))
            accs = []
            for i in range(len(accs_all[0])):
                acc = 0
                for weight, accs_list in zip(weights, accs_all):
                    acc += weight * accs_list[i]
                accs.append(acc)
            accs = tuple(accs)
        elif config.augmentation.use_dual_cutout:
            outputs1, outputs2 = outputs[:, 0], outputs[:, 1]
            accs = accuracy((outputs1 + outputs2) / 2, targets, topk)
        else:
            accs = accuracy(outputs, targets, topk)
    else:
        accs = accuracy(outputs, targets, topk)
    return accs


# pure accuracy without any extra setting(e.g data augmentation)
# But careful, here it could be top-1, but also could be top 5, if top1, top5, topk=(1,5)
def accuracy(outputs, targets, topk=(1, )):
    with torch.no_grad():
        # Here max(topk) will decide how many probabilities per sample will be kept.
        maxk = max(topk)
        batch_size = targets.size(0)

        # return the top-k values(dropped) and indices
        # torch.topk(input, k, dim=None, largest=True, sorted=True, out=None)
        # TOPK operation doesn't change the dimension of the Tensor
        # Here maxk is top-k predictions will be kept, dimension=1
        _, pred = outputs.topk(maxk, 1, True, True)
        # If k=5, after transpose, Tensor.size() -> [5, batch_size]
        pred = pred.t()
        # Tensor.eq() returns equality
        # expand_as will broadcast the current Tensor to the shape of pred. It looks like tf.broadcast_to
        # So targets original [1, batch_size] expand to [5, batch_size]
        # comparing pred and targets, we have our correctness Tensor with size [5, batch_size],
        # along the first dimension, only 1 of 5 is correct.
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # if among the top-5 predictions(batch_size * 5), for all samples top-5 predictions are correct, summing up,
            # there will batch_size correct positions. batch_size * (1 / batch_size) = 1, top-5 accuracy will be 100%.
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1 / batch_size))
    return res


class DTopECE(nn.Module):
    """
    classical ECE with equal size
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super().__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        # softmax, not log_softmax
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        # default value will be 0
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            # calculate confidences > bin_lower element-wise
            # calculate confidences < bin_upper element-wise
            # multiplication --> confidence in intervals could be kept
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            # fraction of samples in current bin
            prop_in_bin = in_bin.float().mean()
            # if there is any samples in this bin
            if prop_in_bin.item() > 0:
                # indexing: use binary in_bin_samples to fetch corresponding samples in the bin
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                # not distinguish between underconfident and overconfident, just sum up the total absolute difference
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


class DClasswiseECE(nn.Module):
    """
    Compute Classwise ECE
    """

    def __init__(self, n_bins=15):
        super().__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        num_classes = int((torch.max(labels) + 1).item())
        softmaxes = F.softmax(logits, dim=1)
        per_class_sce = None

        for i in range(num_classes):
            class_confidences = softmaxes[:, i]
            class_sce = torch.zeros(1, device=logits.device)
            labels_in_class = labels.eq(i) # one-hot vector of all positions where the label belongs to the class i

            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                in_bin = class_confidences.gt(bin_lower.item()) * class_confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = labels_in_class[in_bin].float().mean()
                    avg_confidence_in_bin = class_confidences[in_bin].mean()
                    class_sce += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            if (i == 0):
                per_class_sce = class_sce
            else:
                per_class_sce = torch.cat((per_class_sce, class_sce), dim=0)

        sce = torch.mean(per_class_sce)
        return sce

class MTopECE(nn.Module):
    """
    classical ECE with equal mass
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super().__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        # calculate the index lower bound and upper bound for each bin
        num_samples = logits.size()[0]
        self.bin_lowers *= num_samples
        self.bin_uppers *= num_samples
        self.bin_lowers.round_()
        self.bin_uppers.round_()

        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        # default value will be 0
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            # calculate confidences > bin_lower element-wise
            # calculate confidences < bin_upper element-wise
            # multiplication --> confidence in intervals could be kept
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            # fraction of samples in current bin
            prop_in_bin = in_bin.float().mean()
            # if there is any samples in this bin
            if prop_in_bin.item() > 0:
                # indexing: use binary in_bin_samples to fetch corresponding samples in the bin
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                # not distinguish between underconfident and overconfident, just sum up the total absolute difference
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece