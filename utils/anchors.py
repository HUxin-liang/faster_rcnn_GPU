import numpy as np

def generate_anchor_base(base_size=16, rations=[0.5, 1, 2],
                         anchor_scales=[8, 16, 32]):
    '''
    :param base_size: 16是对应原图16*16个像素点区域
    :param rations:
    :param anchor_scales: [8,16,32]是针对特征图的
    :return:每个特征点的anchor_base，[9,4]
    '''
    anchor_base = np.zeros((len(rations)*len(anchor_scales),4),
                           dtype=np.float32)

    for i in range(len(rations)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(rations[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / rations[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = - h / 2.
            anchor_base[index, 1] = - w / 2.
            anchor_base[index, 2] = h / 2.
            anchor_base[index, 3] = w / 2.
        return anchor_base

def _enmuerate_shifted_anchor(anchor_base, feat_stride, height, width):
    '''
    生成所有特征点的anchor
    :param anchor_base:
    :param feat_stride:
    :param height:
    :param width:
    :return: anchor:[feature, 9, 4]
    '''
    # 计算网格中心点
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_x.ravel(), shift_y.ravel(),
                      shift_x.ravel(), shift_y.ravel(),), axis=1)

    # 每个网格点上9个anchor
    # anchor_base.shape=[9,4]
    A = anchor_base.shape[0]
    # shift.shape = [feature,2,2]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((K, 1, 4))
    # 所有先验框
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor
