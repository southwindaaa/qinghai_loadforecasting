from data_provider.data_loader import QinghaiLoadData
from torch.utils.data import DataLoader

data_dict = {
    'qinghaidata':QinghaiLoadData,
}


def data_provider(args, flag):
    Data = data_dict[args.data]

    if flag in ['test','zeroshot']:
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    
    data_set = Data(
            flag=flag,
            root_path = args.root_path,
            size=[args.seq_len, args.label_len, args.pred_len],
            data_path=args.data_path,  
            other_id=args.other_id,
            scale=args.scale,
            args=args,
        )
    
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
