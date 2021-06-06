import pandas as pd
import scipy.io as sio
from PIL import Image
from matplotlib import pyplot as plt
from torch.autograd import Variable

import misc.transforms as own_transforms
from config import cfg
from misc.utils import *
from models.CC import CrowdCounter

torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

dataRoot = '../ProcessedData/SHHA/test'
test_model_path = './exp/NewNet-SHHA-20201228-150552/epoch_183_mae_73.47_mse_114.67.pth'

exp_name = './test_results/{:s}'.format(test_model_path.split('/')[2])
if not os.path.exists(exp_name):
    os.makedirs(exp_name)

if not os.path.exists(exp_name + '/pred'):
    os.makedirs(exp_name + '/pred')

if not os.path.exists(exp_name + '/gt'):
    os.makedirs(exp_name + '/gt')

mean_std = ([0.410824894905, 0.370634973049, 0.359682112932], [0.278580576181, 0.26925137639, 0.27156367898])
img_transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*mean_std)
])
restore = standard_transforms.Compose([
    own_transforms.DeNormalize(*mean_std),
    standard_transforms.ToPILImage()
])
pil_to_tensor = standard_transforms.ToTensor()


def main():
    file_list = [filename for root, dirs, filename in os.walk(dataRoot + '/img/')]

    test(file_list[0], test_model_path)


def test(file_list, model_path):
    net = CrowdCounter(cfg.GPU_ID, cfg.NET)
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()

    f1 = plt.figure(1)

    N = len(file_list)
    gts = []
    preds = []

    mae = AverageMeter()
    mse = AverageMeter()

    for i, filename in enumerate(file_list):
        img_name = dataRoot + '/img/' + filename
        filename_no_ext = filename.split('.')[0]

        den_name = dataRoot + '/den/' + filename_no_ext + '.csv'

        den = pd.read_csv(den_name, sep=',', header=None).values
        den = den.astype(np.float32, copy=False)

        img = Image.open(img_name)

        if img.mode == 'L':
            img = img.convert('RGB')

        img = img_transform(img)

        gt_count = np.sum(den)
        with torch.no_grad():
            img = Variable(img[None, :, :, :]).cuda()
            pred_map = net.test_forward(img)

        sio.savemat(exp_name + '/pred/' + filename_no_ext + '.mat', {'data': pred_map.squeeze().cpu().numpy() / 100.})
        sio.savemat(exp_name + '/gt/' + filename_no_ext + '.mat', {'data': den})

        pred_map = pred_map.cpu().data.numpy()[0, 0, :, :]

        pred_count = np.sum(pred_map) / 100.0
        pred_map = pred_map / np.max(pred_map + 1e-20)

        den = den / np.max(den + 1e-20)

        den_frame = plt.gca()
        plt.imshow(den, 'jet')
        den_frame.axes.get_yaxis().set_visible(False)
        den_frame.axes.get_xaxis().set_visible(False)
        den_frame.spines['top'].set_visible(False)
        den_frame.spines['bottom'].set_visible(False)
        den_frame.spines['left'].set_visible(False)
        den_frame.spines['right'].set_visible(False)
        plt.savefig('{:}/IMG_{:}_gt_{:d}.png'.format(exp_name, filename_no_ext, int(gt_count)),
                    bbox_inches='tight', pad_inches=0, dpi=150)

        plt.close()

        # sio.savemat(exp_name+'/'+filename_no_ext+'_gt_'+str(int(gt))+'.mat',{'data':den})

        pred_frame = plt.gca()
        plt.imshow(pred_map, 'jet')
        pred_frame.axes.get_yaxis().set_visible(False)
        pred_frame.axes.get_xaxis().set_visible(False)
        pred_frame.spines['top'].set_visible(False)
        pred_frame.spines['bottom'].set_visible(False)
        pred_frame.spines['left'].set_visible(False)
        pred_frame.spines['right'].set_visible(False)
        plt.savefig('{:}/IMG_{:}_pred_{:.2f}.png'.format(exp_name, filename_no_ext, pred_count),
                    bbox_inches='tight', pad_inches=0, dpi=150)

        plt.close()

        # sio.savemat(exp_name+'/'+filename_no_ext+'_pred_'+str(float(pred))+'.mat',{'data':pred_map})

        diff = den - pred_map

        diff_frame = plt.gca()
        plt.imshow(diff, 'jet')
        plt.colorbar()
        diff_frame.axes.get_yaxis().set_visible(False)
        diff_frame.axes.get_xaxis().set_visible(False)
        diff_frame.spines['top'].set_visible(False)
        diff_frame.spines['bottom'].set_visible(False)
        diff_frame.spines['left'].set_visible(False)
        diff_frame.spines['right'].set_visible(False)
        plt.savefig('{:}/IMG_{:}_diff.png'.format(exp_name, filename_no_ext),
                    bbox_inches='tight', pad_inches=0, dpi=150)

        plt.close()

        mae.update(np.abs(gt_count - pred_count))
        mse.update((gt_count - pred_count) * (gt_count - pred_count))

        print('({:4d}/{:4d}) IMG_{:8s}, gt: {:4d}, pred: {:7.2f}'.format(
            i + 1, N, filename, int(gt_count), pred_count))

        # sio.savemat(exp_name+'/'+filename_no_ext+'_diff.mat',{'data':diff})

    print('\n{:}\nMAE: {:.2f}, MSE: {:.2f}'.format(
        '-' * 80, mae.avg, np.sqrt(mse.avg)))


if __name__ == '__main__':
    main()
