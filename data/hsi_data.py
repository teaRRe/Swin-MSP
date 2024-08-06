# --------------------------------------------------------
# HSI 数据处理模块
# --------------------------------------------------------
from typing import Union

import numpy as np
import scipy.io as sio
import torch.utils.data
from sklearn.model_selection import train_test_split

from DATA_PATH import (SA, SA_GT, PA, PA_GT, IN_PATH, IN_GT_PATH, SA_A, SA_A_GT,
                       __IN, __IN_gt, __SA, __SA_gt, __PU, __PU_gt, PU, PU_GT,
                       XiongAn, XIongAn_GT, Xuzhou, Xuzhou_GT, KSC, KSC_GT, Botswana, Botswana_GT, HOU, HOU_GT,
                       HOU2018_GT, HOU2018, WHU_Hi_HanChuan, WHU_Hi_HanChuan_GT, WHU_Hi_HongHu,
                       WHU_Hi_HongHu_GT, WHU_Hi_LongKou, WHU_Hi_LongKou_GT)


def load_data(ds):
    print(f"---------------------------load_data:{ds}---------------------------")
    if ds == 'Indian_Pines':  # 200
        data_hsi = sio.loadmat(IN_PATH)['indian_pines_corrected']  # [:,:,:192]
        gt_hsi = sio.loadmat(IN_GT_PATH)['indian_pines_gt']
    elif ds == 'Salinas':  # 204 *
        data_hsi = sio.loadmat(SA)['salinas_corrected'][:, :, 0:200]
        gt_hsi = sio.loadmat(SA_GT)['salinas_gt']
    elif ds == 'Pavia':  # 102
        data_hsi = sio.loadmat(PA)['pavia'][:, :, 0:100]
        gt_hsi = sio.loadmat(PA_GT)['pavia_gt']
    elif ds == 'PaviaU':  # 103 *
        data_hsi = sio.loadmat(PU)['paviaU'][:, :, 0:100]
        gt_hsi = sio.loadmat(PU_GT)['paviaU_gt']
    elif ds == 'SalinasA':
        data_hsi = sio.loadmat(SA_A)['salinasA_corrected']
        gt_hsi = sio.loadmat(SA_A_GT)['salinasA_gt']
    elif ds == 'Indian_Pines_c':
        data_hsi = sio.loadmat(__IN)['img']
        gt_hsi = sio.loadmat(__IN_gt)['GroundT']
    elif ds == 'Salinas_c':
        data_hsi = sio.loadmat(__SA)['img'][:, :, 0:200]
        gt_hsi = sio.loadmat(__SA_gt)['GroundT']
    elif ds == 'PaviaU_c':
        data_hsi = sio.loadmat(__PU)['img'][:, :, 0:100]
        gt_hsi = sio.loadmat(__PU_gt)['GroundT']
    elif ds == 'Houston2013':  # 取后140维
        data_hsi = sio.loadmat(HOU)['HSI'][:, :, 4:144]
        # data_hsi = sio.loadmat(HOU)['HSI']
        gt_hsi = sio.loadmat(HOU_GT)['gt']
    elif ds == 'Houston2018':
        import h5py
        import numpy as np
        fgt = h5py.File(HOU2018_GT, mode='r')
        gt_hsi = np.array(fgt['/houstonU_gt'][:])
        f = h5py.File(HOU2018, mode='r')
        data_hsi = np.array(f['/houstonU'][:])
        data_hsi = data_hsi.transpose(1, 2, 0)[:, :, :40]
        del f, fgt
    elif ds == 'XiongAn':
        import h5py
        import numpy as np
        fgt = h5py.File(XIongAn_GT, mode='r')
        gt_hsi = np.array(fgt['/xiongan_gt'][:])
        f = h5py.File(XiongAn, mode='r')
        data_hsi = np.array(f['/XiongAn'][:])
        data_hsi = data_hsi.transpose(1, 2, 0)[:, :, :200]
        # print(data_hsi.shape)
    elif ds == 'Xuzhou':
        data_hsi = sio.loadmat(Xuzhou)['xuzhou'][:, :, :400]
        gt_hsi = sio.loadmat(Xuzhou_GT)['xuzhou_gt']
    elif ds == 'KSC':
        data_hsi = sio.loadmat(KSC)['KSC']
        gt_hsi = sio.loadmat(KSC_GT)['KSC_gt']
    elif ds == 'Botswana':
        data_hsi = sio.loadmat(Botswana)['Botswana']
        gt_hsi = sio.loadmat(Botswana_GT)['Botswana_gt']
    elif ds == 'WHU_Hi_HanChuan':
        data_hsi = sio.loadmat(WHU_Hi_HanChuan)['WHU_Hi_HanChuan'][:, :, 50:250]
        gt_hsi = sio.loadmat(WHU_Hi_HanChuan_GT)['WHU_Hi_HanChuan_gt']
    elif ds == 'WHU_Hi_HongHu':
        data_hsi = sio.loadmat(WHU_Hi_HongHu)['WHU_Hi_HongHu'][:, :, :240]
        gt_hsi = sio.loadmat(WHU_Hi_HongHu_GT)['WHU_Hi_HongHu_gt']
    elif ds == 'WHU_Hi_LongKou':
        data_hsi = sio.loadmat(WHU_Hi_LongKou)['WHU_Hi_LongKou'][:, :, :240]
        gt_hsi = sio.loadmat(WHU_Hi_LongKou_GT)['WHU_Hi_LongKou_gt']
    else:
        raise ValueError("dataset doesn't exist or wrong name")

    print('data_hsi.shape: ', data_hsi.shape)
    print('gt_hsi.shape: ', gt_hsi.shape)

    return data_hsi, gt_hsi


def pad_with_zeros(x, margin=2):
    print("------------------------------pad_with_zeros------------------------------")
    new_x = np.zeros((x.shape[0] + 2 * margin, x.shape[1] + 2 * margin, x.shape[2]))
    x_offset = margin
    y_offset = margin
    new_x[x_offset:x.shape[0] + x_offset, y_offset:x.shape[1] + y_offset, :] = x
    return new_x


def create_samples(x, y, margin=2, remove_zero_labels=False):
    """
    将所有像素点创建成为样本
    """
    print("------------------------------create_samples------------------------------")
    samples_data = np.zeros((x.shape[0] * x.shape[1], 2 * margin + 1, 2 * margin + 1, x.shape[2]), dtype=np.float32)
    samples_label = np.zeros((x.shape[0] * x.shape[1]), dtype=np.int8)
    sample_index = 0
    for r in range(margin, x.shape[0] - margin):
        for c in range(margin, x.shape[1] - margin):
            patch = x[r - margin:r + margin + 1, c - margin:c + margin + 1]
            samples_data[sample_index, :, :, :] = patch
            samples_label[sample_index] = y[r - margin, c - margin]
            sample_index = sample_index + 1
    if remove_zero_labels:
        samples_data = samples_data[samples_label > 0, :, :, :]
        samples_label = samples_label[samples_label > 0]
        samples_label -= 1

    return samples_data, samples_label


class AllDataSet(torch.utils.data.Dataset):
    """
    创建数据集，类型：DataSet
    """

    def __init__(self, x, y):
        self.len = x.shape[0]
        self.x_data = torch.FloatTensor(x)
        self.y_data = torch.LongTensor(y)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


def split_train_test_set(x, y, test_ratio, random_state=345, **kwargs):
    """
    :param x: 全部数据
    :param y: 全部 labels
    :param test_ratio: 测试数据集的比例
    :param random_state:
    :return: X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=test_ratio,
                                                        random_state=random_state,
                                                        stratify=y)

    return X_train, X_test, y_train, y_test


def arr2poi(poi: np.ndarray, h: int, w: int):
    """
    ndarray格式的位置转为fatten过后的位置
    :return:
    """
    list_index = []
    for i in range(len(poi)):
        list_index.append(poi[i][0] * w + poi[i][1])
    return list_index


def choose_sample(ds: str, num: int, per: Union[int, float], **kwargs):
    """
    选取固定样本。20个，小于40个的选取其50%，实验结果表明已经达到模型极限
    每个类选取num个样本，少于num个样本的类选取其per比例
    weights: deprecated
    :param ds: 数据集
    :param num: 每个类选取的样本数量
    :param per: float:少于2*num个样本的类选取per个百分比;int:少于2*num个样本的类选取per个;\n
    list:根据列表里的数量选取
    :return: flatten过后的训练坐标，测试坐标
    """
    remove_zero = kwargs['remove_zero']
    _, gt = load_data(ds)
    h, w = gt.shape
    _class = np.bincount(gt.flatten())  # 第一列是0的数目，属于不参与分类的类别
    train_poi = []
    zero_poi = []
    weights = torch.HalfTensor(len(_class) - 1)

    for i in range(1, len(_class)):
        all_indexes = np.argwhere(gt == i)  # 类别为i的所有索引
        _rand_poi = torch.argsort(torch.rand(len(all_indexes)))
        if type(per) == float:  # per是float，选取百分比
            p = int(len(all_indexes) * (1 - per))
            poi = all_indexes[_rand_poi[:p]]
            print('class: {}, sample: {}'.format(i, p))
        elif type(per) == int:  # per是int，选取指定数量
            if len(all_indexes) > num:  # 如果某类大于num个，则选择num个
                poi = all_indexes[_rand_poi[:num]]
                print('class: {}, sample: {}'.format(i, num))
            else:  # 否则，选择per个
                poi = all_indexes[_rand_poi[:per]]
                print('class: {}, sample: {}'.format(i, per))
        train_poi.extend(arr2poi(poi, h, w))

    # 为0的索引
    zero_indexes = np.argwhere(gt == 0)
    zero_poi.extend(arr2poi(zero_indexes, h, w))

    all = np.arange(h * w)
    # 非0索引，可认为是所有的可分类数据，也就是去除未分类像素
    nonzero = np.setdiff1d(all, zero_poi)
    # 所有可分类的索引除去训练索引，剩下的就是测试索引
    test_poi = np.setdiff1d(nonzero, train_poi)
    if not remove_zero:
        nonzero = np.concatenate([nonzero, np.array(zero_poi)])
        # nonzero = np.array(zero_poi)    # for DEBUG
        # train_poi.extend(zero_poi)
        # test_poi = np.concatenate([test_poi, np.array(zero_poi)])
    return np.array(train_poi), nonzero, test_poi, weights


if __name__ == "__main__":
    # test

    KSC = r'../hsi_data/KSC/KSC.mat'
    KSC_GT = r'../hsi_data/KSC/KSC_gt.mat'
    XuZhou = r'../hsi_data/Xuzhou/xuzhou.mat'
    XuZhou_GT = r'../hsi_data/Xuzhou/xuzhou_gt.mat'

    # data = sio.loadmat(PU)['paviaU'][:, :, 0:100]
    # gt = sio.loadmat(PU_GT)['paviaU_gt']

    # data = sio.loadmat(IN_PATH)['indian_pines_corrected']
    # gt = sio.loadmat(IN_GT_PATH)['indian_pines_gt']

    # data = sio.loadmat(HOU)['HSI'][:, :, 4:144]
    # gt = sio.loadmat(HOU_GT)['gt']

    data = sio.loadmat(KSC)['KSC']
    gt = sio.loadmat(KSC_GT)['KSC_gt']

    # import h5py
    # import numpy as np
    # fgt = h5py.File(HOU2018_GT, mode='r')
    # gt = np.array(fgt['/houstonU_gt'][:])
    # f = h5py.File(HOU2018, mode='r')
    # data = np.array(f['/houstonU'][:])
    # data = data.transpose(1, 2, 0)[:, :, :48]
    # del f, fgt

    # _class = np.bincount(gt.flatten())
    # sum = 0
    # for i in range(len(_class) - 1):
    #     sum = sum + _class[i + 1]
    # train_poi = []
    # zero_poi = []
    # h, w = gt.shape
    # for i in range(len(_class) -1):
    #     all_indexes = np.argwhere(gt == i)  # 类别为i的所有索引
    #     _rand_poi = torch.argsort(torch.rand(len(all_indexes)))
    #     poi = all_indexes[_rand_poi[:HOU2018_SAMPLE[i-1]]]
    #     train_poi.extend(arr2poi(poi, h, w))

    # np.bincount 统计每个数字出现的次数，从0开始，gt数据刚好是从0开始，可以用这个统计每个类的总数 Indian_Pines
    all_poi, nonzero_poi, test, _ = choose_sample("KSC", 50, 0.9, remove_zero=True)
    # print(all_poi, nonzero_poi, test)
