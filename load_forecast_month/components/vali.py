import pickle

import numpy as np
import torch
from sklearn.metrics import r2_score
from tqdm import tqdm


def validating(args, epoch, vali_loader, device, model, criterion, mae_metric, save_path):
    total_loss = []
    total_mae_loss = []
    model.eval()
    model.to(device)
    data_x = []
    data_y = []
    pred_y = []
    with torch.no_grad():
        with tqdm(total=args.train_epochs, desc=f"Testing", unit="batch") as pbar:
            for i, batch in enumerate(vali_loader):
                batch_x = batch[0].float().to(device)
                batch_y = batch[1].float().to(device)
                batch_x_mark = batch[2].float().to(device)
                batch_y_mark = batch[3].float().to(device)
                net_id = batch[4].float().to(device)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(
                    device)
                dec_inp = torch.cat([batch_x[:, :args.pred_len, :], dec_inp], dim=1).float().to(
                    device)

                batch_x_mark = None
                batch_y_mark = None
                if args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -args.pred_len:, :]
                batch_y = batch_y[:, -args.pred_len:, :].to(device)

                pred = outputs.detach()
                true = batch_y.detach()
                data_x.append(batch_x.detach().cpu().numpy())
                data_y.append(true.detach().cpu().numpy())
                pred_y.append(pred.detach().cpu().numpy())

                loss = criterion(pred, true)
                mae_loss = mae_metric(pred, true)
                total_loss.append(loss.item())
                total_mae_loss.append(mae_loss.item())

                # 更新 tqdm 进度条，并显示自定义指标
                pbar.set_postfix({"Loss": f"{loss:.4f}", "Mean Loss": f"{np.average(total_loss):.4f}",
                                  "MAE Loss": f"{mae_loss}", "Mean MAE Loss": f"{np.average(total_mae_loss):.4f}"})
                pbar.update(1)

    data_x = np.concatenate(data_x, axis=0)
    data_y = np.concatenate(data_y, axis=0)
    pred_y = np.concatenate(pred_y, axis=0)
    r2 = r2_score(data_y.reshape(-1, 1), pred_y.reshape(-1, 1))

    validation_data = {}
    validation_data['data_x'] = data_x
    validation_data['data_y'] = data_y
    validation_data['pred_y'] = pred_y
    validation_data['total_loss'] = total_loss
    validation_data['total_mae_loss'] = total_mae_loss
    validation_data['total_r2'] = r2
    pickle.dump(validation_data, open(save_path + '/checkpoint-{}'.format(epoch) + '/vali_data.pkl', 'wb'))
    return np.average(total_loss), np.average(total_mae_loss),r2

