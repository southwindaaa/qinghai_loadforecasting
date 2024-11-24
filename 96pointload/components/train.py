import time

import torch
from tqdm import tqdm

import utils.configure as cfg
from utils.tools import adjust_learning_rate

config = cfg.load_config()['params'] #获取配置信息


def train(train_loader,model_optim,device, model,criterion,scaler):
    for epoch in range(config['train_epochs']):
        iter_count = 0
        train_steps = len(train_loader)
        train_loss = []

        model.train()
        epoch_time = time.time()
        for i, batch in tqdm(enumerate(train_loader)):
            iter_count += 1
            model_optim.zero_grad()
            batch_x = batch[0].float().to(device)
            batch_y = batch[1].float().to(device)
            batch_x_mark = batch[2].float().to(device)
            batch_y_mark = batch[3].float().to(device)
            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -config['pred_len']:, :]).float().to(device)
            dec_inp = torch.cat([batch_x[:, :config['pred_len'], :], dec_inp], dim=1).float().to(device)

            if config['use_amp']:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
            else:
                outputs = model(batch_x, None, dec_inp, None)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                print(
                    "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((config['train_epochs'] - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            if config['use_amp']:
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                loss.backward()
                model_optim.step()

            if config['lradj'] == 'TST':
                adjust_learning_rate(None, model_optim, scheduler, epoch + 1, args, printout=False)
                scheduler.step()

        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        # accelerator.save_state(output_dir=path+'/checkpoint-{}'.format(epoch))
        checkpoint_path = path + '/checkpoint-{}'.format(epoch)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        # print(model.state_dict().keys())
        torch.save(model.state_dict(), checkpoint_path + '/model_weights.pth')
        train_loss = np.average(train_loss)
        validation_data = vali_evaluation(args, device, model, vali_data, vali_loader, criterion, mae_metric)
        vali_loss = validation_data['total_loss']
        vali_mae_loss = validation_data['total_mae_loss']
        vali_r2 = validation_data['total_r2']

        # testation_data = vali_social_evaluation(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
        # test_loss = testation_data['total_loss']
        # test_mae_loss = testation_data['total_mae_loss']
        # test_r2 = testation_data['total_r2']
        if vali_loss < min_mse:
            min_mse = vali_loss
        if vali_mae_loss < min_mae:
            min_mae = vali_mae_loss
        if vali_r2 > max_r2:
            max_r2 = vali_r2
            torch.save(model.state_dict(), path + '/model_weights.pth')
        pickle.dump(validation_data, open(checkpoint_path + '/test_data.pkl', 'wb'))
        print(
            "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f} MAE Loss: {4:.7f}, r2: {5:.7f}".format(
                epoch + 1, train_loss, vali_loss, vali_loss, vali_mae_loss, vali_r2))
        # accelerator.print(
        #     "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f}".format(
        #         epoch + 1, train_loss, vali_loss))
        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        if args.lradj != 'TST':
            if args.lradj == 'COS':
                scheduler.step()
                print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                if epoch == 0:
                    args.learning_rate = model_optim.param_groups[0]['lr']
                    print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                adjust_learning_rate(None, model_optim, scheduler, epoch + 1, args, printout=True)

        else:
            print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
