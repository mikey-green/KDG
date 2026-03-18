import os
import random

import torch
import numpy as np

from time import time
from tqdm import tqdm
from copy import deepcopy
import logging
from prettytable import PrettyTable
import math

from Utils.parser import parse_args
from Utils.data_loader import load_data
from Utils.evaluate import test
from Utils.helper import early_stopping

######################################################################################################
# student模型
from Modules.Student import StudentLightGCN
import torch.nn.functional as F

'''
def kl_div_with_logits(s_logits, t_logits, temperature=4.0):
    """学生和教师的 logits 之间的 KL 散度"""
    s_prob = F.log_softmax(s_logits / temperature, dim=-1)
    t_prob = F.softmax(t_logits / temperature, dim=-1)
    return F.kl_div(s_prob, t_prob, reduction='batchmean') * (temperature ** 2)
'''
def kl_div_with_logits(s_logits, t_logits, temperature=2.0):
    s_log_prob = F.log_softmax(s_logits / temperature, dim=1)
    t_prob = F.softmax(t_logits / temperature, dim=1)

    return F.kl_div(s_log_prob, t_prob, reduction='batchmean') * (temperature ** 2)

def wasserstein_distance(s_logits, t_logits):
    s_prob = F.softmax(s_logits, dim=-1)
    t_prob = F.softmax(t_logits, dim=-1)
    return torch.mean(torch.abs(s_prob - t_prob))


######################################################################################################

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
                                                       n_negs * K)).to(device)
    return feed_dict


if __name__ == '__main__':
    ##############################################################################
    # 日志系统（自动保存训练信息）
    def init_logger(args):
        log_dir = "Log"
        os.makedirs(log_dir, exist_ok=True)

        # 文件名：student_数据集名称_dim**.txt
        log_file = f"student_{args.gnn}_{args.dataset}_dim{args.dim}.txt"
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


    ##############################################################################
    def get_save_path(args, model_type="student"):
        save_dir = "Checkpoints"
        os.makedirs(save_dir, exist_ok=True)

        filename = f"{model_type}_{args.gnn}_{args.dataset}_dim{args.dim}_hop{args.context_hops}_kd.pth"
        return os.path.join(save_dir, filename)
    ##############################################################################

    """fix the random seed"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
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
    train_cf, user_dict, n_params, norm_mat = load_data(args)
    train_cf_size = len(train_cf)
    train_cf = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_negs = args.n_negs
    K = args.K

    ####################################################################################################
    # 新增代码
    """define Teacher model"""
    teacher_args = deepcopy(args)
    teacher_args.dim = 64
    teacher_args.context_hops = 3

    if args.gnn == 'lightgcn':
        from modules.LightGCN import LightGCN

        teacher = LightGCN(n_params, teacher_args, norm_mat).to(device)
    else:
        from Modules.NGCF import NGCF

        teacher = NGCF(n_params, teacher_args, norm_mat).to(device)

    # ===== 加载预训练 Teacher 权重 =====
    teacher_save_path = get_save_path(teacher_args, model_type="teacher").replace("_kd.pth", ".pth")
    try:
        teacher.load_state_dict(torch.load(teacher_save_path, map_location=device))
        teacher.eval()
        print(f"成功加载Teacher模型：{teacher_save_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"未找到Teacher模型文件：{teacher_save_path}")

    # ===== 冻结 Teacher =====
    for param in teacher.parameters():
        param.requires_grad = False

    print("Teacher model loaded and frozen.")

    for name, param in teacher.named_parameters():
        print(name, param.requires_grad)

    # ========= Student =========
    student = StudentLightGCN(n_params, args, norm_mat).to(device)

    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    ####################################################################################################

    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False

    print("start training ...")
    for epoch in range(args.epoch):

        # =========================
        # 时间衰减（epoch级）
        # =========================
        progress = epoch / args.epoch

        alpha = 2.0
        time_weight = 1.0 + 0.5 * progress

        # ===== 统计 KD 各部分 loss =====
        score_sum = 0
        rep_sum = 0
        struct_sum = 0
        batch_count = 0

        # shuffle training data
        train_cf_ = train_cf
        index = np.arange(len(train_cf_))
        np.random.shuffle(index)
        train_cf_ = train_cf_[index].to(device)

        """training"""
        student.train()
        loss, s = 0, 0
        hits = 0
        train_s_t = time()
        while s + args.batch_size <= len(train_cf):
            batch = get_feed_dict(train_cf_,
                                  user_dict['train_user_set'],
                                  s, s + args.batch_size,
                                  n_negs)
            ####################################################################################################
            # 新增代码
            users = batch['users']
            pos_items = batch['pos_items']
            # neg_items = batch['neg_items'][:, 0]  # 只取一个负样本

            # 安全获取单个负样本
            neg_items = batch['neg_items']
            if neg_items.dim() > 1:
                neg_items = neg_items[:, 0]  # 多维时取第一列
            else:
                neg_items = neg_items  # 一维时直接使用

            # ===== Student 多层 embedding =====
            # s_user_layers, s_item_layers = student.generate(return_layers=True)
            try:
                s_user_layers, s_item_layers = student.generate(return_layers=True)
            except TypeError:
                s_users_emb, s_items_emb = student.generate()
                s_user_layers = s_users_emb.unsqueeze(1)
                s_item_layers = s_items_emb.unsqueeze(1)

            # 最终embedding
            s_users_emb = student.pooling(s_user_layers)
            s_items_emb = student.pooling(s_item_layers)

            s_u = s_users_emb[users]
            s_pos = s_items_emb[pos_items]
            s_neg = s_items_emb[neg_items]

            s_pos_scores = torch.sum(s_u * s_pos, dim=1)
            s_neg_scores = torch.sum(s_u * s_neg, dim=1)

            bpr_loss = -torch.mean(F.logsigmoid(s_pos_scores - s_neg_scores))

            # ===== Teacher forward =====
            with torch.no_grad():
                # ===== Teacher embedding（兼容 LightGCN / NGCF）=====
                try:
                    t_user_layers, t_item_layers = teacher.generate(return_layers=True)
                    t_users_emb = teacher.pooling(t_user_layers)
                    t_items_emb = teacher.pooling(t_item_layers)

                except TypeError:
                    # NGCF fallback
                    t_users_emb, t_items_emb = teacher.generate()

                    # 转换成统一格式 [N,1,dim]
                    t_user_layers = t_users_emb.unsqueeze(1)
                    t_item_layers = t_items_emb.unsqueeze(1)

                t_u = t_users_emb[users]
                t_pos = t_items_emb[pos_items]
                t_neg = t_items_emb[neg_items]

                t_pos_scores = torch.sum(t_u * t_pos, dim=1)
                t_neg_scores = torch.sum(t_u * t_neg, dim=1)

            # =========================
            # Score KD
            # =========================
            # 拼接正负样本分数构成 logits（形状 [batch, 2]）
            s_logits = torch.stack([s_pos_scores, s_neg_scores], dim=1)
            t_logits = torch.stack([t_pos_scores, t_neg_scores], dim=1)
            #score_kd = wasserstein_distance(s_logits, t_logits)
            score_kd = kl_div_with_logits(s_logits, t_logits, temperature=2.0)

            # =========================
            # Representation KD（中间层蒸馏 + Projection）
            # =========================
            rep_kd = 0
            min_layers = min(s_user_layers.shape[1], t_user_layers.shape[1])

            for l in range(min_layers):
                # Student 映射到 Teacher 维度
                s_user_proj = student.kd_proj(s_user_layers[:, l, :])
                s_item_proj = student.kd_proj(s_item_layers[:, l, :])

                rep_kd += F.mse_loss(
                    s_user_proj,
                    t_user_layers[:, l, :]
                )

                rep_kd += F.mse_loss(
                    s_item_proj,
                    t_item_layers[:, l, :]
                )

            rep_kd = rep_kd / min_layers

            # =========================
            # Structure KD（Batch级结构蒸馏，防OOM）
            # =========================

            # 取 batch 内 embedding
            s_users_batch = s_users_emb[users]
            s_items_batch = s_items_emb[pos_items]

            t_users_batch = t_users_emb[users]
            t_items_batch = t_items_emb[pos_items]

            # Student 投影到 Teacher 维度
            s_users_proj = student.kd_proj(s_users_batch)
            s_items_proj = student.kd_proj(s_items_batch)

            sim_s_ui = F.cosine_similarity(s_users_proj, s_items_proj, dim=1)
            sim_t_ui = F.cosine_similarity(t_users_batch, t_items_batch, dim=1)

            struct_kd = F.mse_loss(sim_s_ui, sim_t_ui)
            '''
            struct_kd = F.mse_loss(
                F.normalize(sim_s_ui, dim=0),
                F.normalize(sim_t_ui, dim=0)
            )
            '''
            kd_weight = 0.3 * (1 / (1 + math.exp(-(epoch - 0.3 * args.epoch) / 10)))
            # =========================
            # 总Loss
            # =========================

            # 0.3 0.3 0.1->0.4 0.4 0.03->0.5 0.4 0.002
            batch_loss = time_weight * bpr_loss \
                         + time_weight * kd_weight * (
                                 0.4 * score_kd +
                                 0.3 * rep_kd +
                                 0.3 * struct_kd
                         )

            # ===== 统计 KD =====
            score_sum += score_kd.item()
            rep_sum += rep_kd.item()
            struct_sum += struct_kd.item()
            batch_count += 1

            # 添加 L2 正则化
            l2_lambda = 1e-5
            l2_norm = sum(p.norm(2) for p in student.parameters())  # 或 p.pow(2).sum()
            batch_loss = batch_loss + l2_lambda * l2_norm
            ####################################################################################################

            # 反向传播
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            s += args.batch_size

        train_e_t = time()

        if batch_count > 0:
            logger.info(
                "KD avg: score=%.6f rep=%.6f struct=%.6f"
                % (score_sum / batch_count,
                   rep_sum / batch_count,
                   struct_sum / batch_count)
            )

        if epoch % 5 == 0:
            """testing"""
            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time(s)", "testing time(s)", "Loss", "recall", "ndcg",
                                     "precision", "hit_ratio"]

            student.eval()
            test_s_t = time()
            ####################################################################################################
            test_ret = test(student, user_dict, n_params, mode='test')
            ####################################################################################################
            test_e_t = time()

            train_res.add_row(
                [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), test_ret['recall'], test_ret['ndcg'],
                 test_ret['precision'], test_ret['hit_ratio']])

            if user_dict['valid_user_set'] is None:
                valid_ret = test_ret
            else:
                test_s_t = time()
                # valid_ret = test(model, user_dict, n_params, mode='valid')
                ####################################################################################################
                valid_ret = test(student, user_dict, n_params, mode='valid')
                ####################################################################################################
                test_e_t = time()
                train_res.add_row(
                    [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), valid_ret['recall'],
                     valid_ret['ndcg'],
                     valid_ret['precision'], valid_ret['hit_ratio']])

            logger.info("\n" + str(train_res))

            # 学习率调整
            if 'scheduler' in locals():
                scheduler.step(valid_ret['recall'][0])

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for 10 successive steps.
            cur_best_pre_0, stopping_step, should_stop = early_stopping(valid_ret['recall'][0], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=6)
            if should_stop:
                break

            """save weight"""
            if valid_ret['recall'][0] == cur_best_pre_0:
                save_path = get_save_path(args, model_type="student")
                torch.save(student.state_dict(), save_path)
                logger.info(f"Best Student saved to {save_path}")
        else:
            logger.info(
                'using time %.4fs, training loss at epoch %d: %.4f'
                % (train_e_t - train_s_t, epoch, loss.item())
            )
    logger.info('early stopping at %d, recall@20:%.4f' % (epoch, cur_best_pre_0))
