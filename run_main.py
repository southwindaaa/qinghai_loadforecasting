import argparse
import torch
# from accelerate import Accelerator, DeepSpeedPlugin
# from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

# from models import Autoformer, DLinear, TimeLLM,FEDformer, TSMixer, PatchTST, iTransformer, LSTM
from models import DLinear

from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import os
import pickle
from utils.painting import draw_comparision

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content, vali_evaluation

parser = argparse.ArgumentParser(description='Qinghai')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--checkpoint', type=int, default=0, help='checkpoint')
parser.add_argument('--model_id', type=str, required=True, default='qinghai', help='model id')
parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, required=True, default='DLinear',
                    help='model name, options: [Autoformer, DLinear]')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/', help='root path')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--scale', type=int, default=0, help='whether to scale data')

# model define
parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--prompt_domain', type=int, default=1, help='')
parser.add_argument('--llm_model', type=str, default='GPT2', help='LLM model') # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default='768', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768


# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=4, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--content', type=str, default="")
parser.add_argument('--other_id', type=str, default=None)
parser.add_argument('--train_date', type=str, default='20220120')

args = parser.parse_args()
if args.is_training == 0 and args.checkpoint <= 0:
    raise ValueError("When --is_training is 0, --checkpoint must be greater than 0.")


# ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
#deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
# accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])#, deepspeed_plugin=deepspeed_plugin)

for ii in range(args.itr):
    # setting record of experiments
    setting = f'{args.task_name}_{args.model}_{args.data}'
    setting += f'_{args.des}_{args.llm_model}_d_ff_{args.d_ff}'

    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    # test_data, test_loader = data_provider(args, 'test')
    device = torch.device('cuda:0')
    
    if args.model == 'DLinear':
        model = DLinear.Model(args).float().to(device)
    # elif args.model == 'Autoformer':
    #     model = Autoformer.Model(args).float()
    # elif args.model == 'PatchTST':
    #     model = PatchTST.Model(args).float()
    # elif args.model == 'FEDformer':
    #     model = FEDformer.Model(args).float()
    # elif args.model == 'TSMixer':
    #     model = TSMixer.Model(args).float()
    # elif args.model == 'iTransformer':
    #     model = iTransformer.Model(args).float()
    # # elif args.model == 'TimeLLM':
    # #     if args.add_social:
    # #         model = TimeLLM_wSocial.Model(args).float()
    # #     else:
    # #         model = TimeLLM.Model(args).float()
    # elif args.model == 'LSTM':
    #     model = LSTM.Model(args).float()
    #     # model = TimeLLM.Model(args).float()
    else:
        raise NotImplementedError

    path = os.path.join(args.checkpoints,
                        setting + '-' + args.model_comment)  # unique checkpoint saving path
    
    if not os.path.exists(path):
        os.makedirs(path)
    time_now = time.time()

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(accelerator=None, patience=args.patience)

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    trained_parameters_show = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trained_parameters_show.append((name, param))

    # 打印每个训练参数的名字
    for name, param in trained_parameters_show:
        print(f"Parameter name: {name}")

    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=args.pct_start,
                                            epochs=args.train_epochs,
                                            max_lr=args.learning_rate)

    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()

    # train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
    #     train_loader, vali_loader, test_loader, model, model_optim, scheduler)
    # train_loader, model, model_optim, scheduler = accelerator.prepare(
    #     train_loader, model, model_optim, scheduler)
    print ("train_loader size",len(train_loader))
    # print ("vali_loader size",len(vali_loader))
    # print ("test_loader size",len(test_loader))
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    min_mse = 10000
    min_mae = 10000
    max_r2 = -10000
    if args.is_training:
        for epoch in range(args.train_epochs):
            iter_count = 0
            train_loss = []

            model.train()
            epoch_time = time.time()
            #for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
            print(len(train_loader))
            for i, batch in tqdm(enumerate(train_loader)):
                # print ("batch",batch)
                # print ("batch[0].shape",batch[0].shape)
                # print ("batch[1].shape",batch[1].shape)
                # print ("batch[2].shape",batch[2].shape)
                # print ("batch[3].shape",batch[3].shape)
                # print ("batch[4].shape",batch[4].shape)
                # if args.other_id == 1:
                #     print ("batch[5].shape",batch[5].shape)
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch[0].float().to(device)
                batch_y = batch[1].float().to(device)
                batch_x_mark = batch[2].float().to(device)
                batch_y_mark = batch[3].float().to(device)
                # print(batch_x.device,batch_x_mark.device,batch_y.device,batch_y_mark.device)
                net_id = batch[4].float().to(device)
                if args.other_id == 1:
                    other_id = batch[5].float().to(device)
                else:
                    other_id = None
                
                # print ("batch_x.shape",batch_x.shape)
                # print ("batch_y.shape",batch_y.shape)
                # print (social_prompt)
                # exit()
                
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(
                    device)
                # print ('dec_inp',dec_inp)
                dec_inp = torch.cat([batch_x[:, :args.pred_len, :], dec_inp], dim=1).float().to(
                    device)
                
                if args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = model(batch_x, None, dec_inp, None)
                    #f_dim = -1 if args.features == 'MS' else 0
                    #outputs = outputs[:, -args.pred_len:, f_dim:]
                    #batch_y = batch_y[:, -args.pred_len:, f_dim:]
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
                    #print ('outputs',outputs.shape,'batch_y',batch_y.shape)
                    
                

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if args.lradj == 'TST':
                    adjust_learning_rate(None, model_optim, scheduler, epoch + 1, args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            #accelerator.save_state(output_dir=path+'/checkpoint-{}'.format(epoch))
            checkpoint_path = path + '/checkpoint-{}'.format(epoch)
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            #print(model.state_dict().keys())
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
                    epoch + 1, train_loss, vali_loss,vali_loss, vali_mae_loss, vali_r2))
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
    # else:
    #     print("direct to test")
    #     model.load_state_dict(torch.load(path + f'/checkpoint-{args.checkpoint}/model_weights.pth'),strict=False)
    #     model.eval()
    #     if args.zeroshot:
    #         testation_data = vali_social_evaluation(args, accelerator, model, zeroshot_data, zeroshot_loader, criterion, mae_metric)
    #     elif args.plot:
    #         testation_data = vali_social_evaluation(args, accelerator, model, plot_data, plot_loader, criterion, mae_metric)
    #     else:
    #         testation_data = vali_social_evaluation(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
    #     test_loss = testation_data['total_loss']
    #     test_mae_loss = testation_data['total_mae_loss']
    #     test_r2 = testation_data['total_r2']
    #     if test_loss < min_mse:
    #         min_mse = test_loss
    #     if test_mae_loss < min_mae:
    #         min_mae = test_mae_loss
    #     if test_r2 > max_r2:
    #         max_r2 = test_r2
    #     if args.eval_no_social:
    #         pickle.dump(testation_data, open(path + '/evaluation_data_no_social.pkl', 'wb'))
    #     elif args.plot:
    #         pickle.dump(testation_data, open(path + '/plot_data.pkl', 'wb'))
    #     else: 
    #         pickle.dump(testation_data, open(path +'/evaluation_data.pkl', 'wb'))
    draw_comparision(checkpoint_path + '/test_data.pkl',ii)

    print ("min_mse: ",min_mse,", min_mae: ",min_mae," max_r2: ",max_r2)
# accelerator.wait_for_everyone()
# if accelerator.is_local_main_process:
#     path = './checkpoints'  # unique checkpoint saving path
#     del_files(path)  # delete checkpoint files
#     accelerator.print('success delete checkpoints')