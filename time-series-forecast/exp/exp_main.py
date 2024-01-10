from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import PatchMixer, SegRNN, iTransformer, TSMixer, CNN_MLP_MLP, TIDE, CNN_GRU_RES, \
    CNN_GRU_RES_V2, CNN_GRU_GRU, CNN_MLP_MLP_V2, Transformer, Informer, DLinear, Linear, \
    NLinear, LSTM, CNN_LINEAR_LINEAR, CNN_LINEAR_LINEAR_V2, LBCNN_LINEAR, CNN_LINEAR, Reformer, GRU, BlockRNN, \
    TCN, TCN_LINEAR, TCN_LSTM_LINEAR
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules import Module


class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        return 0.5 * F.mse_loss(x, y) + 0.5 * F.l1_loss(x, y)

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'PatchMixer': PatchMixer,
            'SegRNN': SegRNN,
            'iTransformer': iTransformer,
            'TSMixer': TSMixer,
            'CNN_MLP_MLP': CNN_MLP_MLP,
            'TIDE': TIDE,
            'CNN_GRU_RES': CNN_GRU_RES,
            'TCN_LINEAR': TCN_LINEAR,
            'TCN_LSTM_LINEAR': TCN_LSTM_LINEAR,
            'CNN_GRU_RES_V2': CNN_GRU_RES_V2,
            'CNN_GRU_GRU': CNN_GRU_GRU,
            'CNN_MLP_MLP_V2': CNN_MLP_MLP_V2,
            'Transformer': Transformer,
            # 'Autoformer': Autoformer,
            # 'FEDformer': FEDformer,
            'Informer': Informer,
            'DLinear': DLinear,
            # 'FiLM': FiLM,
            'Linear': Linear,
            'NLinear': NLinear,
            'LSTM': LSTM,
            'GRU': GRU,
            'BlockRNN': BlockRNN,
            'TCN': TCN,
            # 'NHITS': NHITS,
            'CNN_LINEAR_LINEAR': CNN_LINEAR_LINEAR,
            'CNN_LINEAR_LINEAR_V2': CNN_LINEAR_LINEAR_V2,
            'LBCNN_LINEAR': LBCNN_LINEAR,
            'CNN_LINEAR': CNN_LINEAR,
            'Reformer': Reformer
        }
        model = model_dict[self.args.model].Model(self.args).float()

        # 计算模型占用的空间大小
        total_params = sum(p.numel() for p in model.parameters())
        total_size_MB = total_params * 4 / (1024 ** 2)  # 将参数数量乘以每个浮点数所占的字节数（通常是4字节），然后转换为MB
        print(f"Model: {self.args.model}; Total parameters: {total_params}; Total size: {total_size_MB:.2f} MB")

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate) # for PatchMixer
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate) # origin one
        return model_optim

    def _select_criterion(self):
        criterion = CustomLoss() # for PatchMixer
        # criterion = nn.MSELoss() # origin one
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        # 评估模式
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            
                        else:
                            # print("---exp_main---")
                            # print(batch_x.shape)
                            # print(batch_x_mark.shape)
                            # print(dec_inp.shape)
                            # print(batch_y.shape)
                            # print(batch_y_mark.shape)
                            # print("---end---")
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        # best_model_path = path + '/' + 'checkpoint.pt'
        # # self.model.load_state_dict(torch.load(best_model_path))
        # torch.save(self.model, best_model_path) #保存模型的整体

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # from torchviz import make_dot
        # g = make_dot(outputs)
        # g.render('TCN_LINEAR_model', view=True)  # 会自动保存为一个 espnet.pdf，第二个参数为True,则会自动打开该PDF文件，为False则不打开
        # import tensorwatch as tw
        # img = tw.draw_model(self.model, [32, 384, 78])
        # img.save(r'TCN_LINEAR.png')


        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        # print("test_data:", test_data)
        # print("test_loader:",test_loader)
        
        if test:
            print('loading model') #checkpoints\checkpoint_denormalization_384_288.pth
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/checkpoint_denormalization_384_288.pth')))
            # self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            # print(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                # print("exp test i:", i)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # (3424, 288, 78) 获取子数组 (1, 288, 78)
        import plotly.graph_objects as go

        #预测结果图
        i = [500, 1000, 3000, 3100]
        j = [9, 24, 27, 28, 70, 71]
        for row in i:
            for col in j:
                if row == 3100 and col == 24 or row == 3000 and col == 27 or row == 500 and col == 28 or row == 1000 and col == 9 or row == 1000 and col == 70 or row == 3000 and col == 71:
                    import plotly.graph_objects as go
                    sub_preds = preds[row]
                    sub_pred = sub_preds[:, col:col + 1]
                    sub_trues = trues[row]
                    sub_true = sub_trues[:, col:col + 1]
                    # 计算误差
                    errors = sub_true - sub_pred
                    x = np.arange(1, 289)
                    # 创建预测值和真实值线的Scatter对象
                    trace_pred = go.Scatter(x=x, y=sub_pred.ravel(), name='Predicted',
                                            line=dict(color='#4169E1', width=2))
                    trace_actual = go.Scatter(x=x, y=sub_true.ravel(), name='Actual',
                                              line=dict(color='#ff7f0e', width=2))
                    # 创建误差线的Scatter对象
                    trace_error = go.Bar(x=x, y=errors.ravel(), name='Error', marker=dict(color='#828282'))
                    # 创建误差阴影的填充区域
                    trace_error_shade = go.Scatter(
                        x=np.concatenate([x, x[::-1]]),
                        y=np.concatenate([sub_pred.ravel() - (sub_pred.max() - sub_pred.min()) / 10,
                                          (sub_pred.ravel() + (sub_pred.max() - sub_pred.min()) / 10)[::-1]]),
                        fill='toself',
                        fillcolor='#4169E1',  # 将fillcolor的值更改为橙色的色值
                        opacity=0.3,
                        showlegend=False
                    )

                    # 创建y=0的虚线
                    trace_zero = go.Scatter(x=[1, 288], y=[0, 0], name='Zero',
                                            line=dict(color='red', width=2, dash='dash'),showlegend=False)
                    # 将所有的Scatter对象添加到图表中
                    fig = go.Figure()
                    fig.add_trace(trace_pred)
                    fig.add_trace(trace_actual)
                    fig.add_trace(trace_error)
                    fig.add_trace(trace_error_shade)
                    fig.add_trace(trace_zero)
                    # 更新布局
                    fig.update_layout(
                        xaxis=dict(showgrid=False, showline=True, linewidth=2, linecolor='black', showticklabels=True,
                                   ticks='outside', tickwidth=1, tickcolor='black', tickfont=dict(size=18)),
                        yaxis=dict(title_text=str(row) + "_" + str(col), showgrid=False, showline=True, linewidth=2,
                                   linecolor='black', showticklabels=True, ticks='outside', tickwidth=1,
                                   tickcolor='black',
                                   tickfont=dict(size=18),
                                   range=[min(sub_pred.min(), sub_true.min(), errors.min()),
                                          max(sub_pred.max(), sub_true.max(), errors.max())]),
                        plot_bgcolor="white",
                        width=900,
                        height=600,
                        legend=dict(xanchor='right', yanchor='top', orientation='h', x=0.99, y=1.1,font=dict(size=18)),
                    )
                    # 显示图表
                    fig.show()

                else:
                    continue

        mae, mse, rmse, mape, mspe, rse, corr, r2 = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}, r2:{}'.format(mse, mae, rse, r2))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}, r2:{}'.format(mse, mae, rse, r2))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
