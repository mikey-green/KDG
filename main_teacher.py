import os
import random

import torch
import numpy as np

from time import time
from tqdm import tqdm
from copy import deepcopy
import logging
from prettytable import PrettyTable

from Utils.parser import parse_args
from Utils.data_loader import load_data
from Utils.evaluate import test
from Utils.helper import early_stopping

n_users = 0
n_items = 0


def get_feed_dict(train_entity_pairs, train_pos_set, start, end, n_negs=1):

    def sampling(user_item, train_set, n):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            negitems = []
            for i in range(n):  # sample n times
                while True:
                    negitem = random.choice(range(n_items))
                    if negitem not in train_set[user]:
                        break
                negitems.append(negitem)
            neg_items.append(negitems)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end]
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(sampling(entity_pairs,
                                                       train_pos_set,
                                                       n_negs*K)).to(device)
    return feed_dict


if __name__ == '__main__':
##############################################################################


    # 日志系统（自动保存训练信息）
    def init_logger(args):
        log_dir = "Log"
        os.makedirs(log_dir, exist_ok=True)

        # 文件名：teacher_数据集名称_dim**.txt
        log_file = f"teacher_{args.gnn}_{args.dataset}_dim{args.dim}.txt"
        log_path = os.path.join(log_dir, log_file)

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # 清除旧 handler（避免重复打印）
        if logger.handlers:
            logger.handlers.clear()

        # 写入文件
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)

        # 同时输出到终端
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        print(f"日志将保存到: {log_path}")

        return logger
    #新增代码
    def get_save_path(args, model_type="teacher"):
        save_dir = "Checkpoints"
        os.makedirs(save_dir, exist_ok=True)

        filename = f"{model_type}_{args.gnn}_{args.dataset}_dim{args.dim}_hop{args.context_hops}.pth"
        return os.path.join(save_dir, filename)
##############################################################################
    """fix the random seed"""
    # 固定随机种子，保证可复现性
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    # 解析命令行参数 + 设备配置
    global args, device
    # 加载 parser.py 中定义的超参数
    args = parse_args()
    logger = init_logger(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    ##################################################
    import torch

    if args.gpu_id < 0 or not torch.cuda.is_available():
        device = torch.device('cpu')
        print("⚙️  Using CPU for training (no CUDA detected or gpu_id < 0).")
    else:
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"⚙️  Using GPU: cuda:{args.gpu_id}")
    ##################################################
    """build dataset"""
    # 从 data_loader.py 加载数据集
    # train_cf：用户 - 物品交互对（训练集）；
    # user_dict：包含训练 / 验证 / 测试集的用户 - 物品映射；
    # n_params：用户 / 物品数量等统计信息；
    # norm_mat：图归一化邻接矩阵（用于 GNN 传播）
    train_cf, user_dict, n_params, norm_mat = load_data(args)
    train_cf_size = len(train_cf)
    train_cf = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_negs = args.n_negs
    K = args.K

    """define model"""
    # 选择 LightGCN/NGCF 模型,初始化时传入数据维度、超参数、归一化矩阵
    from Modules.LightGCN import LightGCN
    from Modules.NGCF import NGCF
    if args.gnn == 'lightgcn':
        model = LightGCN(n_params, args, norm_mat).to(device)
    else:
        model = NGCF(n_params, args, norm_mat).to(device)

    """define optimizer"""
    # 优化器（Adam）
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False

    print("start training ...")
    for epoch in range(args.epoch):
        # shuffle training data 打乱训练数据
        train_cf_ = train_cf
        index = np.arange(len(train_cf_))
        np.random.shuffle(index)
        train_cf_ = train_cf_[index].to(device)

        """training"""
        # 批量训练
        model.train()
        loss, s = 0, 0
        hits = 0
        train_s_t = time()
        while s + args.batch_size <= len(train_cf):
            #构建批次（包含负样本）
            batch = get_feed_dict(train_cf_,
                                  user_dict['train_user_set'],
                                  s, s + args.batch_size,
                                  n_negs)

            batch_loss, _, _ = model(batch)

            # 反向传播
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            s += args.batch_size

        train_e_t = time()

        if epoch % 5 == 0:
            """testing"""

            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time(s)", "testing time(s)", "Loss", "recall", "ndcg", "precision", "hit_ratio"]

            model.eval()
            test_s_t = time()
            test_ret = test(model, user_dict, n_params, mode='test')
            test_e_t = time()
            train_res.add_row(
                [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), test_ret['recall'], test_ret['ndcg'],
                 test_ret['precision'], test_ret['hit_ratio']])

            if user_dict['valid_user_set'] is None:
                valid_ret = test_ret
            else:
                test_s_t = time()
                valid_ret = test(model, user_dict, n_params, mode='valid')
                test_e_t = time()
                train_res.add_row(
                    [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), valid_ret['recall'], valid_ret['ndcg'],
                     valid_ret['precision'], valid_ret['hit_ratio']])
            logger.info("\n" + str(train_res))

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for 10 successive steps.早停逻辑
            cur_best_pre_0, stopping_step, should_stop = early_stopping(valid_ret['recall'][0], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=10)
            if should_stop:
                break

            """save weight"""
                
            if valid_ret['recall'][0] == cur_best_pre_0:
                save_path = get_save_path(args, model_type="teacher")
                torch.save(model.state_dict(), save_path)
                logger.info(f"Best Teacher saved to {save_path}")

        else:
            logger.info('using time %.4fs, training loss at epoch %d: %.4f' % (train_e_t - train_s_t, epoch, loss.item()))
    logger.info('early stopping at %d, recall@20:%.4f' % (epoch, cur_best_pre_0))
