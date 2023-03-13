import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import math
import argparse
import _init_paths
from config import cfg
from utils.utils import fix_random_seed
from config import update_config
import pprint
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from dataset.TALDataset import TALDataset
from models.a2net import A2Net
from core.function import train, evaluation
from core.post_process import final_result_process
from core.utils_ab import weight_init
from utils.utils import save_model, backup_codes

def main(subject, config):
    update_config(config)
    # create output directory
    if cfg.BASIC.CREATE_OUTPUT_DIR:
        out_dir = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.MODEL_DIR, subject)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    # copy config file
    if cfg.BASIC.BACKUP_CODES:
        backup_dir = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.MODEL_DIR, subject, 'code')
        backup_codes(cfg.BASIC.ROOT_DIR, backup_dir, cfg.BASIC.BACKUP_LISTS)
    fix_random_seed(cfg.BASIC.SEED)
    if cfg.BASIC.SHOW_CFG:
        pprint.pprint(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLE

    # model
    model = A2Net(cfg)
    model.apply(weight_init)
    model.cuda()
    
    # optimizer
    # warm_up_with_cosine_lr
    # optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, momentum=0.9, nesterov=True)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    warm_up_with_cosine_lr = lambda epoch: epoch / cfg.TRAIN.WARM_UP_EPOCH if epoch <= cfg.TRAIN.WARM_UP_EPOCH else 0.5 * ( math.cos((epoch - cfg.TRAIN.WARM_UP_EPOCH) /(cfg.TRAIN.END_EPOCH - cfg.TRAIN.WARM_UP_EPOCH) * math.pi) + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
    
    # data loader
    train_dset = TALDataset(cfg, cfg.DATASET.TRAIN_SPLIT, subject)
    train_loader = DataLoader(train_dset, batch_size=cfg.TRAIN.BATCH_SIZE,
                              shuffle=True, drop_last=False, num_workers=cfg.BASIC.WORKERS, pin_memory=cfg.DATASET.PIN_MEMORY)
    val_dset = TALDataset(cfg, cfg.DATASET.VAL_SPLIT, subject)
    val_loader = DataLoader(val_dset, batch_size=cfg.TEST.BATCH_SIZE,
                            shuffle=False, drop_last=False, num_workers=cfg.BASIC.WORKERS, pin_memory=cfg.DATASET.PIN_MEMORY)
    # creating log path
    log_path = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.MODEL_DIR, subject, str(cfg.TRAIN.BATCH_SIZE) + '_' + cfg.TRAIN.LOG_FILE)
    if os.path.exists(log_path):
        os.remove(log_path)
    
    for epoch in range(cfg.TRAIN.END_EPOCH):
        # loss_train, cls_loss_af, reg_loss_af, cls_loss_ab, reg_loss_ab, rep_loss_ab, rep_loss_af= train(cfg, train_loader, model, optimizer)
        # print('Epoch %d: loss: %.4f AF cls loss: %.4f, reg loss: %.4f, rep loss: %.4f AB cls loss: %.4f, reg loss: %.4f, rep loss: %.4f ' % (
        #     epoch, loss_train, cls_loss_af, reg_loss_af, rep_loss_af, cls_loss_ab, reg_loss_ab, rep_loss_ab))
        
        loss_train, cls_loss_af, reg_loss_af, cls_loss_ab, reg_loss_ab = train(cfg, train_loader, model, optimizer)
        print('Epoch %d: loss: %.4f AF cls loss: %.4f, reg loss: %.4f AB cls loss: %.4f, reg loss: %.4f' % (
            epoch, loss_train, cls_loss_af, reg_loss_af, cls_loss_ab, reg_loss_ab))
        
        with open(log_path, 'a') as f:
            f.write('Epoch %d: loss: %.4f AF cls loss: %.4f, reg loss: %.4f AB cls loss: %.4f, reg loss: %.4f\n' % (
                epoch, loss_train, cls_loss_af, reg_loss_af, cls_loss_ab, reg_loss_ab))

        # decay lr
        scheduler.step()
        lr = scheduler.get_last_lr()

        if (epoch+1) % cfg.TEST.EVAL_INTERVAL == 0:
            # save_model(cfg, epoch=epoch, model=model, optimizer=optimizer, subject=subject)
            out_df_ab, out_df_af = evaluation(val_loader, model, epoch, cfg)
            out_df_list = [out_df_ab, out_df_af]
            final_result_process(out_df_list, epoch, subject, cfg, flag=0)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MER SPOT')
    # parser.add_argument('--dataset', type=str, default='cas(me)^2')
    # parser.add_argument('--cfg', type=str, help='experiment config file', default='/home/yww/1_spot/MSA-Net/experiments/cas.yaml')

    # parser.add_argument('--cfg', type=str, help='experiment config file', default='/home/yww/1_spot/MSA-Net/experiments/samm7.yaml')
    # parser.add_argument('--dataset', type=str, default='samm')
    
    parser.add_argument('--cfg', type=str, help='experiment config file', default='/home/yww/1_spot/MSA-Net/experiments/cas32.yaml')
    parser.add_argument('--dataset', type=str, default='cas(me)^3')

    parser.add_argument('--subject', nargs='+', default=[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,25,26,30,32,33,34,35,36,37,99], required=False)
    args = parser.parse_args()

    dataset = args.dataset
    new_cfg = args.cfg
    print(args.cfg)
    if dataset == 'cas(me)^2':
        ca_subject = [16,15,19,20,21,22,23,24,25,26,27,29,30,31,32,33,34,35,36,37,38,40]
        ca_subject = ['s'+ str(i)  for i in ca_subject]
        for i in ca_subject:
            subject = 'subject_' + i 
            main(subject, new_cfg)
    elif dataset == 'cas(me)^3':
        ca3_subject = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,39,40,41,42,77,138,139,140,142,143,144,145,146,147,148,\
            149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,165,166,167,168,169,170,171,172,173,\
            174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,192,193,194,195,196,197,198,\
            200,201,202,203,204,206,207,208,209,210,212,213,214,215,216,217,218]
        ca3_subject = [str(i).zfill(3) for i in ca3_subject]
        for i in ca3_subject:
            main(i, new_cfg)
    else:
        sa_subject = args.subject
        # sa_subject = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,25,26,30,32,33,34,35,36,37,99]
        sa_subject = [str(i).zfill(3)  for i in sa_subject]
        for i in sa_subject:
            subject = 'subject_' + i 
            main(subject, new_cfg)

