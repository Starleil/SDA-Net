import os
import argparse
import yaml
import torch
import torch.backends.cudnn as cudnn
from utils.tools import load_checkpoint
from BasicNet import BasicNet
from utils import net_utils

def arg_parse():
    parser = argparse.ArgumentParser(
        description='BasicNet_for_TN')
    parser.add_argument('-cfg', '--config', default='configs/settings.yaml',
                        type=str, help='load the config file')
    parser.add_argument('--cuda', default=True,
                        type=bool, help='Use cuda to train model')
    args = parser.parse_args()
    return args

def main():
    args = arg_parse()
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)

    gpus = ','.join([str(i) for i in config['GPUs']])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    print('Using the network: {}'.format(config['arch']))

    # set net
    BasicModel = BasicNet.build_model(config['arch'], config['Downsampling'], config['Using_pooling'],
                                      config['Using_aggregation'], config['Using_cam'], config['Using_dilation'], len(config['Data_CLASSES']))

    if config['Using_pretrained_weights']:
        BasicModel.load_pretrained_weights()
    if config['Basic']['resume'] != None:
        load_checkpoint(BasicModel, config['Basic']['resume'])

    if args.cuda:
        BasicModel.cuda()
        cudnn.benchmark = True

    BasicModel = torch.nn.DataParallel(BasicModel)

    optimizer, train_loader, val_loader = net_utils.prepare_net(config, BasicModel)

    net_utils.train_net(optimizer, train_loader, val_loader, BasicModel, config)

if __name__ == '__main__':

    main()