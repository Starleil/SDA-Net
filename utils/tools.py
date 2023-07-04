import math
import torch
import numpy as np
import shutil
from .config import *
from collections import OrderedDict

def adjust_learning_rate(optimizer, epoch, epoch_num, initial_lr, reduce_epoch, decay=0.1, print_lr=True):
    epoch -= 1
    if reduce_epoch == 'dynamic':
        lr = initial_lr * (1 - math.pow(float(epoch)/float(epoch_num), power))
    else:
        lr = initial_lr * (decay ** (epoch // reduce_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if print_lr:
        print('lr={:.10f}'.format(lr))

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, sum_flag=True):
        if sum_flag:
            self.val = val
            self.sum += val * n
        else:
            self.val = val / n
            self.sum += val
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(net, arch, name, epoch, _bestauc=False, _bestacc=False, _bestloss=False, log=False, only_log=False, best=0, train_log=None, valid_log=None):
    savepath = os.path.join(model_savepath, arch, name) # ./checkpoint/arch/model_name
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    file_name = os.path.join(savepath, "{}_epoch_{:0>4}".format(arch, epoch)+ '.pth')
    torch.save(net.state_dict(), file_name)
    remove_flag = False
    if _bestloss:
        best_name = os.path.join(savepath, "{}_best_loss".format(arch)+ '.pth') # ./checkpoint/arch/model_name/arch_best_loss.pth
        shutil.copy(file_name, best_name)
        remove_flag = True
        file = open(os.path.join(savepath, "{}_best_loss".format(arch)+ '.txt'), 'w')
        file.write('arch: {}'.format(arch)+'\n')
        file.write('epoch: {}'.format(epoch)+'\n')
        file.write('best loss: {}'.format(best)+'\n')
        file.close()
    if _bestacc:
        best_name = os.path.join(savepath, "{}_best_acc".format(arch)+ '.pth')
        shutil.copy(file_name, best_name)
        remove_flag = True
        file = open(os.path.join(savepath, "{}_best_acc".format(arch)+ '.txt'), 'w')
        file.write('arch: {}'.format(arch)+'\n')
        file.write('epoch: {}'.format(epoch)+'\n')
        file.write('best mean acc: {}'.format(best)+'\n')
        file.close()
    if _bestauc:
        best_name = os.path.join(savepath, "{}_best_auc".format(arch)+ '.pth')
        shutil.copy(file_name, best_name)
        remove_flag = True
        file = open(os.path.join(savepath, "{}_best_auc".format(arch)+ '.txt'), 'w')
        file.write('arch: {}'.format(arch)+'\n')
        file.write('epoch: {}'.format(epoch)+'\n')
        file.write('best mean auc: {}'.format(best)+'\n')
        file.close()
    if only_log:
        remove_flag = True
        log = True
    if log:
        file = open(os.path.join(savepath, "{}_logging".format(arch) + '.txt'), 'a')
        file.write(train_log + '\n')
        file.write(valid_log + '\n')
        file.close()
    if remove_flag:
        os.remove(file_name)

def load_checkpoint(net, model_path, _sgpu=True):
    state_dict = torch.load(model_path)
    if _sgpu:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # print(k)
            head = k[:7]
            if head == 'module.':
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
    else:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head != 'module.':
                name = 'module.' + k
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
    print('Load resume network...')

# classification metrics
def get_num_correct(preds,labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


# segmentation metrics
def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    # print('Test One Hot: shape:',shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)
    # result = result.scatter_(1, input.cpu(), 1)

    return result


def get_accuracy(SR, GT, threshold=0.5):
    corr = 0.
    GT = make_one_hot(GT.unsqueeze(1), SR.size(1))
    GT = GT.cuda()
    for i in range(SR.size(1)):
        SR[:, i, :, :] = SR[:, i, :, :] > threshold
        GT[:, i, :, :] = GT[:, i, :, :] == torch.max(GT)
        corr += torch.sum(SR[:, i, :, :] == GT[:, i, :, :])
    tensor_size = SR.size(0) * SR.size(1) * SR.size(2) * SR.size(3)
    acc = float(corr) / float(tensor_size)

    return acc


def get_sensitivity(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    # TP : True Positive
    # FN : False Negative
    TP = 0.
    FN = 0.
    SE = 0.
    # SR = make_one_hot(SR.unsqueeze(1), 4)
    GT = make_one_hot(GT.unsqueeze(1), SR.size(1))
    GT = GT.cuda()
    for i in range(SR.size(1)):
        SR[:, i, :, :] = SR[:, i, :, :] > threshold
        GT[:, i, :, :] = GT[:, i, :, :] == torch.max(GT)
        tp = ((SR[:, i, :, :] == 1) & (GT[:, i, :, :] == 1))
        fn = ((SR[:, i, :, :] == 0) & (GT[:, i, :, :] == 1))
        SE += float(tp.sum()) / (float(tp.sum()) + float(fn.sum()) + 1e-6)
    SE = float(SE / SR.size(1))
    return SE


def get_specificity(SR, GT, threshold=0.5):
    # TN : True Negative
    # FP : False Positive
    TN = 0.
    FP = 0.
    SP = 0.
    GT = make_one_hot(GT.unsqueeze(1), SR.size(1))
    GT = GT.cuda()
    for i in range(SR.size(1)):
        SR[:, i, :, :] = SR[:, i, :, :] > threshold
        GT[:, i, :, :] = GT[:, i, :, :] == torch.max(GT)
        tn = ((SR[:, i, :, :] == 0) & (GT[:, i, :, :] == 0))
        fp = ((SR[:, i, :, :] == 1) & (GT[:, i, :, :] == 0))
        SP += float(tn.sum()) / (float(tn.sum()) + float(fp.sum()) + 1e-6)
    SP = float(SP / SR.size(1))
    return SP


def get_precision(SR, GT, threshold=0.5):
    # TP : True Positive
    # FP : False Positive
    TP = 0.
    FP = 0.
    PC = 0.
    GT = make_one_hot(GT.unsqueeze(1), SR.size(1))
    GT = GT.cuda()
    for i in range(SR.size(1)):
        SR[:, i, :, :] = SR[:, i, :, :] > threshold
        GT[:, i, :, :] = GT[:, i, :, :] == torch.max(GT)
        tp = ((SR[:, i, :, :] == 1) & (GT[:, i, :, :] == 1))
        fp = ((SR[:, i, :, :] == 1) & (GT[:, i, :, :] == 0))
        PC += float(tp.sum()) / (float(tp.sum()) + float(fp.sum()) + 1e-6)
    PC = float(PC / SR.size(1))
    return PC


def get_F1(SR, GT, threshold=0.5):
    PC = []
    SE = []
    F1 = []
    GT = make_one_hot(GT.unsqueeze(1), SR.size(1))
    GT = GT.cuda()
    for i in range(SR.size(1)):
        SR[:, i, :, :] = SR[:, i, :, :] > threshold
        GT[:, i, :, :] = GT[:, i, :, :] == torch.max(GT)
        tp = ((SR[:, i, :, :] == 1) & (GT[:, i, :, :] == 1))
        fp = ((SR[:, i, :, :] == 1) & (GT[:, i, :, :] == 0))
        fn = ((SR[:, i, :, :] == 0) & (GT[:, i, :, :] == 1))
        SE.append(float(tp.sum()) / (float(tp.sum()) + float(fn.sum()) + 1e-6))
        PC.append(float(tp.sum()) / (float(tp.sum()) + float(fp.sum()) + 1e-6))

    assert len(SE) == len(PC)
    for e in range(len(SE)):
        F1.append(2 * SE[e] * PC[e] / (SE[e] + PC[e] + 1e-6))

    F1 = np.mean(F1)
    return F1


def get_DC(SR, GT, threshold=0.5):
    # DC : Dice Coefficient
    # TP : True Positive
    # FP : False Positive
    # FN : False Negative
    TP = 0.
    FP = 0.
    FN = 0.
    dice = 0.
    GT = make_one_hot(GT.unsqueeze(1), SR.size(1))
    GT = GT.cuda()
    for i in range(SR.size(1)):
        SR[:, i, :, :] = SR[:, i, :, :] > threshold
        GT[:, i, :, :] = GT[:, i, :, :] == torch.max(GT)
        tp = ((SR[:, i, :, :] == 1) & (GT[:, i, :, :] == 1))
        fp = ((SR[:, i, :, :] == 1) & (GT[:, i, :, :] == 0))
        fn = ((SR[:, i, :, :] == 0) & (GT[:, i, :, :] == 1))
        dice += float(2 * tp.sum()) / (float(2 * tp.sum()) + float(fp.sum()) + float(fn.sum()) + 1e-6)

    return float(dice / SR.size(1))


def get_MIoU(SR, GT, threshold=0.5):

    IoU = []
    GT = make_one_hot(GT.unsqueeze(1), SR.size(1))
    GT = GT.cuda()
    for i in range(SR.size(1)):
        SR[:, i, :, :] = SR[:, i, :, :] > threshold
        GT[:, i, :, :] = GT[:, i, :, :] == torch.max(GT)
        tp = ((SR[:, i, :, :] == 1) & (GT[:, i, :, :] == 1))
        fp = ((SR[:, i, :, :] == 1) & (GT[:, i, :, :] == 0))
        fn = ((SR[:, i, :, :] == 0) & (GT[:, i, :, :] == 1))
        tn = ((SR[:, i, :, :] == 0) & (GT[:, i, :, :] == 0))
        IoU.append(float(2 * tp.sum()) / (float(2 * tp.sum()) + float(fp.sum()) + float(fn.sum()) + 1e-6))

    return float(np.nanmean(IoU))
