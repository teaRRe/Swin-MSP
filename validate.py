import os
from operator import truediv

import numpy as np
import pandas as pd
import torch
import torchvision.transforms
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, cohen_kappa_score
from tqdm import tqdm

from extra_util import gpu, cpu


def test(device, net, dataloader):
    count = 0
    # 模型测试
    net.eval()
    y_pred_test = 0
    y_test = 0
    for inputs, labels in tqdm(dataloader,postfix='Testing...'):
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))

    y_test = y_test.flatten()
    y_pred_test = y_pred_test.flatten()

    return y_pred_test, y_test


def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def acc_reports(y_test, y_pred_test):
    target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture', 'Grass-trees',
                    'Grass-pasture-mowed', 'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                    'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                    'Stone-Steel-Towers']
    # classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names)
    classification = classification_report(y_test, y_pred_test, digits=4)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, oa * 100, confusion, each_acc * 100, aa * 100, kappa * 100


def _get_classification_map(y_pred, y):
    """
    :param y_pred: 全部可分类像素(n,)
    :param y: gt (h,w)
    """
    height = y.shape[0]
    width = y.shape[1]
    k = 0
    cls_labels = np.zeros((height, width))
    for i in range(height):
        for j in range(width):

            target = int(y[i, j])
            if target == 0: # if gt为0，则让它继续为0
                continue
            else:       # 否则分配类别：pre + 1，就是从0-15 -> 1-16
                cls_labels[i][j] = y_pred[k] + 1
                k += 1

    return cls_labels


def list_to_colormap(x_list):   # IP:21025
    y = np.zeros((x_list.shape[0], 3))  # IP:(21025,3)
    label_color = {
        0: np.array([0, 0, 0]),
        1: np.array([147, 67, 46]),
        2: np.array([0, 0, 255]),
        3: np.array([255, 100, 0]),
        4: np.array([0, 255, 123]),
        5: np.array([164, 75, 155]),
        6: np.array([101, 174, 255]),
        7: np.array([118, 254, 172]),
        8: np.array([60, 91, 112]),
        9: np.array([255, 255, 0]),
        10: np.array([255, 255, 125]),
        11: np.array([255, 0, 255]),
        12: np.array([100, 0, 255]),
        13: np.array([0, 172, 254]),
        14: np.array([0, 255, 0]),
        15: np.array([171, 175, 80]),
        16: np.array([101, 193, 60])
    }
    _label_color = {
        0: np.array([0, 0, 0]),  # 黑色
        1: np.array([240, 240, 240]),  # 浅灰色
        2: np.array([255, 192, 203]),  # 浅粉红
        3: np.array([152, 251, 152]),  # 淡绿
        4: np.array([135, 206, 250]),  # 天蓝
        5: np.array([255, 255, 224]),  # 米黄
        6: np.array([224, 255, 255]),  # 明青
        7: np.array([238, 130, 238]),  # 紫罗兰
        8: np.array([211, 211, 211]),  # 银白
        9: np.array([191, 191, 191]),  # 轻灰
        10: np.array([255, 228, 181]),  # 浅橙
        11: np.array([255, 182, 193]),  # 桃粉
        12: np.array([205, 133, 63]),  # 栗色
        13: np.array([255, 215, 0]),  # 金黄
        14: np.array([199, 97, 20]),  # 栗棕
        15: np.array([152, 251, 152]),  # 淡绿
        16: np.array([255, 255, 255])  # 白色
    }
    for index, item in enumerate(x_list):
        if item != 0:
            y[index] = label_color[item] / 255.0

    return y

def _classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2. / dpi, ground_truth.shape[0] * 2. / dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)

    return 0


def get_cls_map(net, device, all_data_loader, y, file_time, **kwargs):
    """
    :param net: 网络
    :param device: 设备
    :param all_data_loader: 全部可分类数据集
    :param y: h×w的gt:(h,w) 9类：0-9
    """
    y_pred, y_new = test(device, net, all_data_loader)  # y_pred:(n,),y_new:(n,)
    cls_labels = _get_classification_map(y_pred, y) # cls_labels:(h,w)[0-9]
    x = np.ravel(cls_labels)
    gt = y.flatten()

    y_list = list_to_colormap(x)    # 全部像素点的 RGB值
    y_gt = list_to_colormap(gt) # 全部像素点的 RGB值

    y_re = np.reshape(y_list, (y.shape[0], y.shape[1], 3))  # h, w, 3
    gt_re = np.reshape(y_gt, (y.shape[0], y.shape[1], 3))   # h, w, 3

    train_type = 'swin_transformer' if not kwargs['train_type'] else kwargs['train_type']

    path = r'./cls_map/' + train_type + '/'
    if not os.path.exists(path):
        os.makedirs(path)

    _classification_map(y_re, y, 50, path + file_time + '_predictions.eps')
    pre_path = path + file_time + '_predictions.png'
    _classification_map(y_re, y, 50, pre_path)
    gt_path = path + file_time + '_gt.png'
    _classification_map(gt_re, y, 50, gt_path)
    return pre_path, gt_path


@torch.no_grad()
def hsi_validate(config, logger, model, dataloader_test, file_time, dataloader_all, labels, **kwargs):
    """
    验证方法，包含了 cls_report, cls_confusion, cls_map，swin_transformer和微调都可以使用，参数 train_type=swin_transformer|swin_mae
    :param logger:
    :param model:
    :param dataloader_test: 用于测试集
    :param file_time:
    :param dataloader_all: 全部数据用于作图
    :param labels:
    :return:
    """
    num_classes = config.MODEL.NUM_CLASSES
    if config.DATA.DATASET == 'SalinasA': num_classes = 6
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # ****************** 测试数据集精度 ******************
    y_pred_test, y_test = test(device, model, dataloader_test)  # pred:(42326,) y:(42326)
    classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)
    classification = str(classification)
    # 打印
    logger.info(f'Test finished.')
    logger.info(f'\nKa: {format(kappa,".2f")} OA: {format(oa,".2f")} AA:{format(aa,".2f")}')
    # ****************** end ******************
    train_type = 'swin_transformer' if not kwargs['train_type'] else kwargs['train_type']

    cls_file_path = r'./cls_result/' + train_type + '/'
    file_name = file_time + "_classification_report.txt"
    if not os.path.exists(cls_file_path):
        os.makedirs(cls_file_path)

    # hsi_confusion(file_time, confusion, train_type=train_type, num_classes=num_classes)
    # logger.info(r'混淆矩阵生成完毕')
    # ****************** 全部数据画分类图 ******************
    logger.info(f'Start writing classification report to {cls_file_path}')
    pre_path, gt_path = get_cls_map(model, device, dataloader_all, labels, file_time, train_type=train_type)    # all数据集

    cls_report = ""
    cls_report += '\n'
    cls_report += '{} Kappa accuracy (%)'.format(kappa)
    cls_report += '\n'
    cls_report += '{} Overall accuracy (%)'.format(oa)
    cls_report += '\n'
    cls_report += '{} Average accuracy (%)'.format(aa)
    cls_report += '\n'
    cls_report += '\n{} Each accuracy (%)'.format(each_acc)
    cls_report += '\n'
    cls_report += '\n{}'.format(classification)
    cls_report += '\n'
    confusion = str(confusion)
    cls_report += '\n{:^}'.format(confusion)

    with open(cls_file_path + file_name, 'w') as x_file:
        x_file.write(cls_report)
    logger.info('Classification report saved in: {}'.format(cls_file_path + file_name))

    gpu()
    cpu()
    pre_data = np.array(Image.open(pre_path))
    gt_data = np.array(Image.open(gt_path))
    totensor = torchvision.transforms.ToTensor()
    pre_data = totensor(pre_data)
    gt_data = totensor(gt_data)

    return cls_report, pre_data, gt_data


def hsi_confusion(file_time, confusion, **kwargs):
    train_type = 'swin_transformer' if not kwargs['train_type'] else kwargs['train_type']
    num_classes = kwargs['num_classes']

    cls_confusion_path = r'./cls_confusion/' + train_type + '/'
    file_name = file_time + '_confusion.png'
    if not os.path.exists(cls_confusion_path):
        os.makedirs(cls_confusion_path)
    # IP 数据集详细
    # target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture', 'Grass-trees',
    #                 'Grass-pasture-mowed', 'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
    #                 'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
    #                 'Stone-Steel-Towers']
    target_names = np.arange(num_classes)
    df_cm = pd.DataFrame(confusion, index=target_names, columns=target_names)
    cmap = 'Oranges'
    # pp_matrix(df_cm, cmap=cmap, fz=11, file=cls_confusion_path + file_name, figsize=[20, 20])


def metrics(prediction, target, n_classes=None):
    """Compute and print metrics (accuracy, confusion matrix and F1 scores).

    Args:
        prediction: list of predicted labels
        target: list of target labels
        n_classes (optional): number of classes, max(target) by default
    Returns:
        accuracy, accuracy by class, confusion matrix
    """
    ignored_mask = np.zeros(target.shape[:2], dtype=np.bool)
    ignored_mask[target < 0] = True
    ignored_mask = ~ignored_mask
    target = target[ignored_mask]
    prediction = prediction[ignored_mask]
    results = {}

    n_classes = np.max(target) + 1 if n_classes is None else n_classes

    cm = confusion_matrix(
        target,
        prediction,
        labels=range(n_classes))

    results["Confusion matrix"] = cm

    # Compute global accuracy
    total = np.sum(cm)
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy /= float(total)

    results["Accuracy"] = accuracy * 100.0

    # Compute accuracy of each class
    class_acc = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            acc = cm[i, i] / np.sum(cm[i, :])
        except ZeroDivisionError:
            acc = 0.
        class_acc[i] = acc

    results["class acc"] = class_acc * 100.0
    results['AA'] = np.mean(class_acc) * 100.0
    # Compute kappa coefficient
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / \
         float(total * total)
    kappa = (pa - pe) / (1 - pe)
    results["Kappa"] = kappa * 100.0

    return results


def show_results(results, label_values=None, agregated=False):
    text = ""

    if agregated:
        accuracies = [r["Accuracy"] for r in results]
        aa = [r['AA'] for r in results]
        kappas = [r["Kappa"] for r in results]
        class_acc = [r["class acc"] for r in results]

        class_acc_mean = np.mean(class_acc, axis=0)
        class_acc_std = np.std(class_acc, axis=0)
        cm = np.mean([r["Confusion matrix"] for r in results], axis=0)
        text += "Agregated results :\n"
    else:
        cm = results["Confusion matrix"]
        accuracy = results["Accuracy"]
        aa = results['AA']
        classacc = results["class acc"]
        kappa = results["Kappa"]

    text += "Confusion matrix :\n"
    text += str(cm)
    text += "---\n"

    if agregated:
        text += ("Accuracy: {:.02f} +- {:.02f}\n".format(np.mean(accuracies),
                                                         np.std(accuracies)))
    else:
        text += "Accuracy : {:.02f}%\n".format(accuracy)
    text += "---\n"

    text += "class acc :\n"
    if agregated:
        for label, score, std in zip(label_values, class_acc_mean,
                                     class_acc_std):
            text += "\t{}: {:.02f} +- {:.02f}\n".format(label, score, std)
    else:
        for label, score in zip(label_values, classacc):
            text += "\t{}: {:.02f}\n".format(label, score)
    text += "---\n"

    if agregated:
        text += ("AA: {:.02f} +- {:.02f}\n".format(np.mean(aa),
                                                   np.std(aa)))
        text += ("Kappa: {:.02f} +- {:.02f}\n".format(np.mean(kappas),
                                                      np.std(kappas)))
    else:
        text += "AA: {:.02f}%\n".format(aa)
        text += "Kappa: {:.02f}\n".format(kappa)

    print(text)
    return text


def test4metrics(net, dataloader):
    """
    :param gt: (h,w)
    """
    net.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    y_pred, y_new = test(device, net, dataloader)  # y_pred:(n,),y_new:(n,)
    classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_new, y_pred)
    run_results = {'Confusion matrix': confusion, 'class acc': each_acc, 'Accuracy': oa, 'AA': aa, 'Kappa': kappa }
    return run_results


def save2file(runs, text, ds, file_time, logger):
    cls_file_path = os.path.join('cls_result', ds, file_time)
    file_name = str(runs) + ".txt"
    if not os.path.exists(cls_file_path):
        os.makedirs(cls_file_path)
    with open(cls_file_path + os.sep + file_name, 'w') as x_file:
        x_file.write(text)
    logger.info("save results to {}".format(cls_file_path + os.sep + file_name))

def get_cls_map2(runs, net, ds, all_data_loader, y, file_time, logger):
    """
    :param net: 网络
    :param device: 设备
    :param all_data_loader: 全部可分类数据集
    :param y: h×w的gt:(h,w) 9类：0-9
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    y_pred, y_new = test(device, net, all_data_loader)  # y_pred:(n,),y_new:(n,)
    cls_labels = _get_classification_map(y_pred, y) # cls_labels:(h,w)[0-9]
    x = np.ravel(cls_labels)
    gt = y.flatten()

    y_list = list_to_colormap(x)    # 全部像素点的 RGB值
    y_gt = list_to_colormap(gt)     # 全部像素点的 RGB值

    y_re = np.reshape(y_list, (y.shape[0], y.shape[1], 3))  # h, w, 3
    gt_re = np.reshape(y_gt, (y.shape[0], y.shape[1], 3))   # h, w, 3

    path = os.path.join('cls_map', ds, file_time)
    if not os.path.exists(path):
        os.makedirs(path)

    pre_path = path + os.sep + str(runs) + '_predictions.png'
    _classification_map(y_re, y, 50, pre_path)
    gt_path = path + os.sep + str(runs) + '_gt.png'
    _classification_map(gt_re, y, 50, gt_path)
    logger.info(f'Saved predictions to {pre_path}')
    logger.info(f'Saved gt to {gt_path}')
    return pre_path, gt_path
