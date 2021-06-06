from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from config import cfg
from misc.utils import *
from models.CC_NewNet import CrowdCounter


class Trainer(nn.Module):
    def __init__(self, dataloader, cfg_data, pwd):
        super(Trainer, self).__init__()

        self.cfg_data = cfg_data

        self.data_mode = cfg.DATASET
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.exception = cfg.EXCEPTION
        self.pwd = pwd

        self.net_name = cfg.NET
        self.net = CrowdCounter(cfg.GPU_ID, self.net_name).cuda()
        self.optimizer = optim.Adam(self.net.CCN.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
        # self.optimizer = optim.SGD(self.net.CCN.parameters(), lr=cfg.LR, momentum=0.95, weight_decay=cfg.WEIGHT_DECAY)
        self.scheduler = StepLR(self.optimizer, step_size=cfg.NUM_EPOCH_LR_DECAY, gamma=cfg.LR_DECAY)
        # self.scheduler = CosineAnnealingLR(optimizer=self.optimizer, T_max=cfg.MAX_EPOCH, eta_min=0)

        self.train_record = {'best_mae': 1e20, 'best_mse': 1e20, 'best_model_name': ''}
        self.timer = {'iter time': Timer(), 'train time': Timer(), 'val time': Timer()}
        self.state_saver = Save_Handle(max_num=1)

        self.epoch = 0
        self.i_tb = 0
        self.scores = None

        self.train_loader, self.val_loader, self.restore_transform = dataloader()

        if cfg.RESUME:
            latest_state = torch.load(cfg.RESUME_PATH)
            self.net.load_state_dict(latest_state['net'])
            self.optimizer.load_state_dict(latest_state['optimizer'])
            self.scheduler.load_state_dict(latest_state['scheduler'])
            self.epoch = latest_state['epoch'] + 1
            self.i_tb = latest_state['i_tb']
            self.train_record = latest_state['train_record']
            self.exp_path = latest_state['exp_path']
            self.exp_name = latest_state['exp_name']

        self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd,
                                           self.exception, resume=cfg.RESUME,
                                           config=cfg, dataset_config=self.cfg_data)

    def forward(self):

        # self.validate_V3()
        for epoch in range(self.epoch, cfg.MAX_EPOCH):
            self.epoch = epoch
            logging.info('> Epoch {:4d}/{:4d} '.format(self.epoch + 1, cfg.MAX_EPOCH))

            # training
            self.timer['train time'].tic()
            self.train()

            if epoch > cfg.LR_DECAY_START:
                self.scheduler.step()

            # validation
            if epoch % cfg.VAL_FREQ == 0 or epoch + 1 > cfg.VAL_DENSE_START:
                self.timer['val time'].tic()
                if self.data_mode == 'WE':
                    self.validate_V2()
                elif self.data_mode == 'GCC':
                    self.validate_V3()
                else:
                    self.validate()
                self.timer['val time'].toc(average=False)
                logging.info('\t[Val]   MAE: {:6.2f}, MSE: {:6.2f}, Val loss: {:7.4f}, Cost time: {:.2f}s {:}'.
                             format(self.scores[0], self.scores[1], self.scores[2], self.timer['val time'].diff,
                                    '(save best model!)' if self.train_record['save_best'] else ''))

    def train(self, mode=True):  # training for all datasets
        self.net.train()
        epoch_losses = AverageMeter()
        epoch_mse_loss = AverageMeter()
        epoch_count_loss = AverageMeter()
        epoch_lcm_loss = AverageMeter()

        for i, data in enumerate(self.train_loader, 0):

            self.timer['iter time'].tic()
            img, gt_map = data
            img = Variable(img).cuda()
            gt_map = Variable(gt_map).cuda()

            self.optimizer.zero_grad()
            pred_map = self.net(img, gt_map)
            loss = self.net.loss
            loss.backward()
            self.optimizer.step()

            epoch_losses.update(self.net.losses.item())
            # epoch_mse_loss.update(self.net.loss_mse.item())
            # epoch_count_loss.update(self.net.count_loss.item())
            epoch_count_loss.update(abs(gt_map[0].sum().data - pred_map[0].sum().data))
            # epoch_lcm_loss.update(self.net.loss_lcm.item())

            if (i + 1) % cfg.PRINT_FREQ == 0:
                self.i_tb += 1
                self.writer.add_scalar('train_loss', loss.item(), self.i_tb)
                self.timer['iter time'].toc(average=False)

                logging.info('\tit: {:3d} | avg. loss: {:9.4f} | avg. count diff: {:9.4f} '
                             '| cnt (gt/pred): {:7.2f}/{:7.2f}'.
                             format(i + 1, float(epoch_losses.avg),
                                    float(epoch_count_loss.avg / self.cfg_data.LOG_PARA),
                                    float(gt_map[0].sum().data / self.cfg_data.LOG_PARA),
                                    float(pred_map[0].sum().data / self.cfg_data.LOG_PARA)))

                # logging.info('\tit: {:3d} | avg. mse loss: {:9.4f} | avg. lcm loss: {:9.4f} '
                #              '| cnt (gt/pred): {:7.2f}/{:7.2f}'.
                #              format(i + 1, float(epoch_mse_loss.avg), float(epoch_lcm_loss.avg),
                #                     float(gt_map[0].sum().data / self.cfg_data.LOG_PARA),
                #                     float(pred_map[0].sum().data / self.cfg_data.LOG_PARA)))

                # print('\t[it: {:4d}] [loss: {:5.4f}[ [lr: {:5.4e}] '
                #       '[time: {:.2f}s] [cnt (gt/pred): {:6.2f}/{:6.2f}'.
                #       format(i + 1, loss.item(), self.optimizer.param_groups[0]['lr'],
                #              self.timer['iter time'].diff,
                #              float(gt_map[0].sum().data / self.cfg_data.LOG_PARA),
                #              float(pred_map[0].sum().data / self.cfg_data.LOG_PARA)))
                # logging.info('\t[it: {:3d}] [loss: {:5.4f}[ [lr: {:5.4e}] '
                #              '[time: {:.2f}s] [cnt (gt/pred): {:6.2f}/{:6.2f}'.
                #              format(i + 1, loss.item(), self.optimizer.param_groups[0]['lr'],
                #                     self.timer['iter time'].diff,
                #                     float(gt_map[0].sum().data / self.cfg_data.LOG_PARA),
                #                     float(pred_map[0].sum().data / self.cfg_data.LOG_PARA)))

                epoch_losses.reset()
                epoch_mse_loss.reset()
                epoch_count_loss.reset()
                epoch_lcm_loss.reset()

        self.timer['train time'].toc(average=False)
        logging.info('\t[Train] LR: {:6.4e}, Cost time: {:.2f}s'.
                     format(self.optimizer.param_groups[0]['lr'], self.timer['train time'].diff))

    def validate(self):  # validate_V1 for SHHA, SHHB, UCF-QNRF, UCF50

        self.net.eval()

        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()

        for vi, data in enumerate(self.val_loader, 0):
            img, gt_map = data

            with torch.no_grad():
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()

                # pred_map = self.net.forward(img, gt_map)
                pred_map = self.net.test_forward(img)

                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()

                for i_img in range(pred_map.shape[0]):
                    pred_cnt = np.sum(pred_map[i_img]) / self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img]) / self.cfg_data.LOG_PARA

                    losses.update(self.net.loss.item())
                    maes.update(abs(gt_count - pred_cnt))
                    mses.update((gt_count - pred_cnt) * (gt_count - pred_cnt))
                # if vi == 0:
                #     vis_results(self.exp_name, self.epoch, self.writer,
                #                 self.restore_transform, img, pred_map, gt_map)

        mae = maes.avg
        mse = np.sqrt(mses.avg)
        loss = losses.avg

        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        self.writer.add_scalar('mae', mae, self.epoch + 1)
        self.writer.add_scalar('mse', mse, self.epoch + 1)

        self.train_record = update_model(self.net, self.optimizer, self.scheduler, self.epoch,
                                         self.i_tb, self.exp_path, self.exp_name, [mae, mse, loss],
                                         self.train_record, self.state_saver, self.log_txt)

        self.scores = [mae, mse, loss]

    def validate_V2(self):  # validate_V2 for WE

        self.net.eval()

        losses = AverageCategoryMeter(5)
        maes = AverageCategoryMeter(5)

        roi_mask = []
        from datasets.WE.setting import cfg_data
        from scipy import io as sio
        for val_folder in cfg_data.VAL_FOLDER:
            roi_mask.append(sio.loadmat(os.path.join(cfg_data.DATA_PATH, 'test', val_folder + '_roi.mat'))['BW'])

        for i_sub, i_loader in enumerate(self.val_loader, 0):

            mask = roi_mask[i_sub]
            for vi, data in enumerate(i_loader, 0):
                img, gt_map = data

                with torch.no_grad():
                    img = Variable(img).cuda()
                    gt_map = Variable(gt_map).cuda()

                    pred_map = self.net.forward(img, gt_map)

                    pred_map = pred_map.data.cpu().numpy()
                    gt_map = gt_map.data.cpu().numpy()

                    for i_img in range(pred_map.shape[0]):
                        pred_cnt = np.sum(pred_map[i_img]) / self.cfg_data.LOG_PARA
                        gt_count = np.sum(gt_map[i_img]) / self.cfg_data.LOG_PARA

                        losses.update(self.net.loss.item(), i_sub)
                        maes.update(abs(gt_count - pred_cnt), i_sub)
                    if vi == 0:
                        vis_results(self.exp_name, self.epoch, self.writer, self.restore_transform, img, pred_map,
                                    gt_map)

        mae = np.average(maes.avg)
        loss = np.average(losses.avg)

        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        self.writer.add_scalar('mae', mae, self.epoch + 1)
        self.writer.add_scalar('mae_s1', maes.avg[0], self.epoch + 1)
        self.writer.add_scalar('mae_s2', maes.avg[1], self.epoch + 1)
        self.writer.add_scalar('mae_s3', maes.avg[2], self.epoch + 1)
        self.writer.add_scalar('mae_s4', maes.avg[3], self.epoch + 1)
        self.writer.add_scalar('mae_s5', maes.avg[4], self.epoch + 1)

        self.train_record = update_model(self.net, self.optimizer, self.scheduler, self.epoch, self.i_tb, self.exp_path,
                                         self.exp_name, [mae, 0, loss], self.train_record, self.log_txt)
        print_WE_summary(self.log_txt, self.epoch, [mae, 0, loss], self.train_record, maes)

    def validate_V3(self):  # validate_V3 for GCC

        self.net.eval()

        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()

        c_maes = {'level': AverageCategoryMeter(9), 'time': AverageCategoryMeter(8), 'weather': AverageCategoryMeter(7)}
        c_mses = {'level': AverageCategoryMeter(9), 'time': AverageCategoryMeter(8), 'weather': AverageCategoryMeter(7)}

        for vi, data in enumerate(self.val_loader, 0):
            img, gt_map, attributes_pt = data

            with torch.no_grad():
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()

                pred_map = self.net.forward(img, gt_map)

                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()

                for i_img in range(pred_map.shape[0]):
                    pred_cnt = np.sum(pred_map[i_img]) / self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img]) / self.cfg_data.LOG_PARA

                    s_mae = abs(gt_count - pred_cnt)
                    s_mse = (gt_count - pred_cnt) * (gt_count - pred_cnt)

                    losses.update(self.net.loss.item())
                    maes.update(s_mae)
                    mses.update(s_mse)
                    attributes_pt = attributes_pt.squeeze()
                    c_maes['level'].update(s_mae, attributes_pt[i_img][0])
                    c_mses['level'].update(s_mse, attributes_pt[i_img][0])
                    c_maes['time'].update(s_mae, attributes_pt[i_img][1] / 3)
                    c_mses['time'].update(s_mse, attributes_pt[i_img][1] / 3)
                    c_maes['weather'].update(s_mae, attributes_pt[i_img][2])
                    c_mses['weather'].update(s_mse, attributes_pt[i_img][2])

                if vi == 0:
                    vis_results(self.exp_name, self.epoch, self.writer, self.restore_transform, img, pred_map, gt_map)

        loss = losses.avg
        mae = maes.avg
        mse = np.sqrt(mses.avg)

        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        self.writer.add_scalar('mae', mae, self.epoch + 1)
        self.writer.add_scalar('mse', mse, self.epoch + 1)

        self.train_record = update_model(self.net, self.optimizer, self.scheduler, self.epoch, self.i_tb, self.exp_path,
                                         self.exp_name, [mae, mse, loss], self.train_record, self.log_txt)

        print_GCC_summary(self.log_txt, self.epoch, [mae, mse, loss], self.train_record, c_maes, c_mses)

    def _forward_unimplemented(self, x) -> None:
        raise NotImplementedError
