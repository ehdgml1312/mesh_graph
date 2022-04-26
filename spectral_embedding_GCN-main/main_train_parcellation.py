import time
import argparse
import json
import os
import numpy as np
import torch
import torch.optim as optm
import random

from os.path import join
from tqdm import tqdm
from shutil import copyfile
from parcellation.input.graphLoader_flipped import GeometricDataset
from torch_geometric.data import DataLoader
from parcellation.custom_callbacks.Loss_plotter import LossPlotter
from parcellation.custom_callbacks.Logger import Logger
from parcellation.models.GCN_parcellation import GCN
# from parcellation.nn.criterions_dice import CompDice
# from parcellation.utils.utils import dice_coeff, compute_acc_metrics
import torch.nn.functional as F


def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# parsing the parameters
def _get_config():
    parser = argparse.ArgumentParser(description="Main handler for training",
                                     usage="python ./train.py -j config.json -g 0")
    # parser.add_argument("-j", "--json", help="configuration json file", required=True)
    # parser.add_argument('-g', '--gpu', help='Cuda Visible Devices', required=True)
    # parser.add_argument('-he', '--hemi', help='Left or right hemishere', required=True)
    # parser.add_argument('-cn', '--name', help='Name of data', required=True)
    # parser.add_argument('-exp', '--exp', help='Name of experiment', required=True)
    # parser.add_argument('-lr', '--lr', help='Learning rate', required=True)
    args = parser.parse_args()

    with open(args.json, 'r') as f:
        config = json.loads(f.read())

    config['directories']['ConfigName'] = args.name + '_' + args.exp + '_' + args.hemi
    config['directories']['hemi'] = args.hemi
    config['directories']['exp'] = args.exp
    config['directories']['name'] = args.name
    config['optimizer_wh']['LR_seg'] = args.lr

    initial_weights = config['generator']['initial_epoch']
    directory = os.path.join(config['directories']['out_dir'],
                             config['directories']['ConfigName'],
                             'config', str(initial_weights))
    if not os.path.exists(directory):
        os.makedirs(directory)

    copyfile(args.json, os.path.join(config['directories']['out_dir'],
                                     config['directories']['ConfigName'],
                                     'config', str(initial_weights), 'config.json'))
    text_file = open(os.path.join(config['directories']['out_dir'],
                                     config['directories']['ConfigName'],
                                     'config', str(initial_weights), 'parameters.txt'), "wt")
    text_file.write(config['directories']['ConfigName'] + '\n')
    text_file.write(config['directories']['hemi'] + '\n')
    text_file.write(config['directories']['exp'] + '\n')
    text_file.write(config['directories']['name'] + '\n')
    text_file.write(config['optimizer_wh']['LR_seg'] + '\n')
    text_file.close()

    # Set the GPU flag to run the code
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    set_seed(config["seed"]["seed"])

    return config


def main(config):
    device = torch.device("cuda")

    generator_config = config['generator']  # Model experiments total epochs and beginning epoch
    initial_epoch = generator_config['initial_epoch']  # O by default and otherwise 'N' if loading
    num_epochs = generator_config['num_epochs']  # Total Number of Epochs
    plt_sep = generator_config['plot_separate']  # Plot the train, valid and test separately: 0 or 1

    model_params = config['model_params']
    feat = model_params['feat']
    hid1 = model_params['hid1']
    hid2 = model_params['hid2']
    hid3 = model_params['hid3']

    par = model_params['par']
    emb_siz = model_params['emb_siz']
    ker_siz = model_params['ker_siz']

    optm_config = config['optimizer_wh']
    b1 = optm_config['B1']  # B1 for Adam Optimizer: Ex. 0.9
    b2 = optm_config['B2']  # B2 for Adam Optimizer: Ex. 0.999
    lr_wh_seg = float(optm_config['LR_seg'])  # Learning Rate: Ex. 0.001
    optm_con_mu = config['optimizer_mu']
    lr_mu = optm_con_mu['LR']  # Learning Rate: Ex. 0.001
    optm_con_si = config['optimizer_si']
    lr_si = optm_con_si['LR']  # Learning Rate: Ex. 0.001

    directory_config = config['directories']
    out_dir = directory_config['out_dir']  # Path to save the outputs of the experiments
    config_name = directory_config['ConfigName']  # Configuration Name to Uniquely Identify this Experiment
    hemi = directory_config['hemi']
    log_path = join(out_dir, config_name, 'log')  # Path to save the training log files
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(join(log_path, 'weights')):  # Path to save the weights of training
        os.makedirs(join(log_path, 'weights'))
    main_path = os.path.join(directory_config['datafile'],
                             directory_config['name']) # Full Path of the dataset

    # Initialize the model, optimizer and data loader
    model_seg = GCN(feat=feat, hid1=hid1, hid2=hid2, hid3=hid3, par=par, emb_size=emb_siz, ker_size=ker_siz)  # Create the model
    model_seg = model_seg.to(device)
    # compute_loss = CompDice()  # Loss function: Here dice
    compute_loss = torch.nn.CrossEntropyLoss()

    # optimizer_mu : updates 'mu' parameters of the network
    # optimizer_si: updates 'sigma' parameters of the network
    # optimizer_wh: updates 'weights' and bias parameters of the network

    optimizer_seg_mu = optm.SGD([model_seg.gc1_1.mu, model_seg.gc2_1.mu,
                                 model_seg.gc3_1.mu, model_seg.gc4_1.mu], lr=lr_mu)
    optimizer_seg_si = optm.SGD([model_seg.gc1_1.sig, model_seg.gc2_1.sig,
                                 model_seg.gc3_1.sig, model_seg.gc4_1.sig], lr=lr_si)
    optimizer_seg_wh = optm.Adam([model_seg.gc1_1.weight, model_seg.gc2_1.weight,
                                  model_seg.gc3_1.weight, model_seg.gc4_1.weight,
                                  model_seg.gc1_1.bias, model_seg.gc2_1.bias,
                                  model_seg.gc3_1.bias, model_seg.gc4_1.bias], lr=lr_wh_seg, betas=(b1, b2))

    data = torch.load('spectral')
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    random.shuffle(data)

    train_set = data[0:71]
    valid_set = data[71:81]
    test_set = data[81:]

    train_loader = DataLoader(train_set,
                              batch_size=generator_config['batch_size'],
                              num_workers=4,
                              shuffle=True)

    valid_loader = DataLoader(valid_set,
                              batch_size=generator_config['batch_size'],
                              num_workers=4,
                              shuffle=False)

    test_loader = DataLoader(test_set,
                             batch_size=generator_config['batch_size'],
                             num_workers=4,
                             shuffle=False)

    if initial_epoch > 0:
        print("===> Loading pre-trained weight {}".format(initial_epoch - 1))
        weight_path = 'weights/model-{:04d}.pt'.format(initial_epoch - 1)
        checkpoint = torch.load(join(log_path, weight_path))
        model_seg.load_state_dict(checkpoint['model_state_dict'])
        # optm.load_state_dict(checkpoint['optimizer_state_dict'])

    def checkpoint(epc):
        w_path = 'weights/model-{:04d}.pt'.format(epc)
        torch.save(
            {'epoch': epc, 'model_state_dict': model_seg.state_dict(),
             'optimizer_state_dict': [optimizer_seg_wh.state_dict(),
                                      optimizer_seg_mu.state_dict(),
                                      optimizer_seg_si.state_dict()]}, join(log_path, w_path))

    # setup our callbacks
    my_metric = ['Dice', 'Segmentation_Accuracy']
    my_loss = ['Loss']

    logger = Logger(mylog_path=log_path, mylog_name="training.log", myloss_names=my_loss, mymetric_names=my_metric)
    ls_plt = LossPlotter(mylog_path=log_path, mylog_name="training.log",
                         myloss_names=my_loss, mymetric_names=my_metric, cmb_plot=plt_sep)

    def train(loader):
        lss_seg = dic_all = acc_all = 0

        for data_dm1 in tqdm(loader):

            # Train Segmentator
            for param in model_seg.parameters():
                param.requires_grad = True
                model_seg.zero_grad()
                model_seg.train()

            data_dm1.to(device)

            optimizer_seg_wh.zero_grad()
            optimizer_seg_mu.zero_grad()
            optimizer_seg_si.zero_grad()

            data_dm1.gt = F.one_hot(data_dm1.y,32)

            pred_dm1 = model_seg(data_dm1)
            loss_seg = compute_loss(pred_dm1, data_dm1.gt)
            # dic = dice_coeff(pred_dm1, data_dm1.gt)
            # acc = compute_acc_metrics(pred_dm1, data_dm1.gt)

            loss_seg.backward()

            optimizer_seg_wh.step()
            optimizer_seg_mu.step()
            optimizer_seg_si.step()

            lss_seg += loss_seg.item()
            # dic_all += dic.item()
            # acc_all += acc.item()

        metric = np.array([lss_seg / len(loader), dic_all / len(loader), acc_all / len(loader)])

        return metric

    def test(loader):
        lss_seg = dic_all = acc_all = 0
        model_seg.eval()

        with torch.no_grad():
            for data in tqdm(loader):
                data.to(device)

                pred_dm1 = model_seg(data)
                data.gt = F.one_hot(data.y, 32)
                loss_seg_dm1 = compute_loss(pred_dm1, data.gt)
                # dic = dice_coeff(pred_dm1, data.gt)
                # acc = compute_acc_metrics(pred_dm1, data.gt)

                lss_seg += loss_seg_dm1.item()
                # dic_all += dic.item()
                # acc_all += acc.item()

            metric = np.array([lss_seg / len(loader), dic_all / len(loader), acc_all / len(loader)])
        return metric

    print("===> Starting Model Training at Epoch: {}".format(initial_epoch))

    for epoch in range(initial_epoch, num_epochs):
        start = time.time()

        print("\n\n")
        print("Epoch:{}".format(epoch))

        train_metric = train(train_loader)
        print("===> Train   Epoch {}: Loss = {:.4f}, Dice_Accuracy = {:.4f}, Accuracy = {:.4f}".format(
            epoch,
            train_metric[0],
            train_metric[1],
            train_metric[2]))

        val_metric = test(valid_loader)
        print("===> Validation Epoch {}: Loss = {:.4f}, Dice_Accuracy = {:.4f}, Accuracy = {:.4f}".format(
            epoch,
            val_metric[0],
            val_metric[1],
            val_metric[2]))

        test_metric = test(test_loader)
        print("===> Testing_d1 Epoch {}: Loss = {:.4f}, Dice_Accuracy = {:.4f}, Accuracy = {:.4f}".format(
            epoch,
            test_metric[0],
            test_metric[1],
            test_metric[2]))

        logger.to_csv(
            np.concatenate((train_metric, val_metric, test_metric)), epoch)
        print("===> Logged All Metrics")
        ls_plt.plotter()
        checkpoint(epoch)

        end = time.time()
        print("===> Epoch:{} Completed in {:.4f} seconds".format(epoch, end - start))

    print("===> Done Training for Total {:.4f} Epochs".format(num_epochs))


if __name__ == "__main__":
    main(_get_config())
