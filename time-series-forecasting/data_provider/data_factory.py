from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, \
    Dataset_Custom2
from torch.utils.data import DataLoader
import numpy as np

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'custom2': Dataset_Custom2,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
    # 针对钢桁架浮桥公开数据集
    # window_stride 3640 train 62 vali test
    # print(data_set[61][0])
    # print(data_set[62][0])
    # print(data_set[63][0])
    # 总共234个样本
    # if flag=="train":
    #     filtered_data_set = tuple(data_set[i] for i in range(0, 187))
    # elif flag=="val":
    #     filtered_data_set = tuple(data_set[i] for i in range(0, 23))
    # else:
    #     filtered_data_set = tuple(data_set[i] for i in range(0, 23))

    # 输出处理后的样本数量
    # print(flag, len(filtered_data_set))
    print(flag, len(data_set))
    # data_loader 根据batch_size生成一个batch的数据，实现多线程
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag, #是否打乱数据的顺序
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
