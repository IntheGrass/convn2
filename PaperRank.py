import os.path

import numpy as np
# from scipy.sparse import csc_matrix
import tqdm
from scipy.stats import rankdata
from random import seed, randint
from operator import itemgetter
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from metrics import Metrics
from scipy.sparse import csr_matrix
# import pandas as pd
import pickle

np.random.seed(1234)
print('start')

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--data", default="data/", help="Data sources.")
parser.add_argument("--name", default="aan_pw", help="Name of the dataset.")
parser.add_argument("--alpha", default=0.85, type=float, help="alpha")
parser.add_argument("--papers_num", default=12390, type=int, help="论文数量，指定实际待排序节点的id范围")

args = parser.parse_args()
print(args)


class Paper:
    def __init__(self, id):
        self.id = id
        self.test_cited_paper = []
        self.train_cited_paper = []

    def add_test_cited_paper(self, cited_paper_id):
        self.test_cited_paper.append(cited_paper_id)

    def add_train_cited_paper(self, cited_paper_id):
        self.train_cited_paper.append(cited_paper_id)


def train_eval_PaperRank(data_dir, dataset):
    train_dir = []
    test_dir = []
    dataset_dir = os.path.join(data_dir, dataset)

    pr5fold = {}
    rec5fold = {}
    f15fold = {}

    n_iter = 50
    alpha = args.alpha
    print(f"alpha = {alpha}")
    fold = 1
    mrr5fold = 0
    bpref5fold = 0
    k = 20  # for precision@k, recall@k, ...@k

    # 确定使用的数据数量
    for i in range(1, 2):
        train_dir.append(dataset_dir + '/train_fold' + str(i) + '.txt')
        test_dir.append(dataset_dir + '/test_fold' + str(i) + '.txt')

    for i in range(1, k + 1):
        pr5fold[i] = 0
        rec5fold[i] = 0
        f15fold[i] = 0

    for s1, s2 in zip(train_dir, test_dir):
        print(s1, s2)
        train_data = open(s1, 'r')
        test_data = open(s2, 'r')
        doi2id_file = open(dataset_dir + '/id_morethan1cite.txt', 'r')

        lines = train_data.readlines()
        lines2 = test_data.readlines()
        doi2id = doi2id_file.readlines()

        total_node = len(doi2id)
        cite_graph = np.zeros((total_node, total_node))
        test_papers = {}
        m = np.zeros((total_node, total_node))  # 列归一化后的引用矩阵，初始化为0
        result = ''  # store recommendation list
        all_sorted_ir_dict = {}  # 存储每篇测试论文的

        PrecAtK = {}
        RecAtK = {}
        F1AtK = {}
        mrr_allpaper = 0
        bpref_allpaper = 0

        for i in range(1, k + 1):
            PrecAtK[i] = 0
            RecAtK[i] = 0
            F1AtK[i] = 0

        for line in lines2:
            edge_arr = line.split('\t')
            citing_id = int(edge_arr[0])
            cited_id = int(edge_arr[2].replace('\n', ''))
            #             d_vec[cited_id] = 1
            #     test_cited_paper.append(cited_id)

            if citing_id not in test_papers:
                test_papers[citing_id] = Paper(citing_id)
            test_papers[citing_id].add_test_cited_paper(cited_id)

        for line in tqdm.tqdm(lines, total=len(lines), desc="build graph"):
            # 构建引用无向图
            edge_arr = line.split('\t')
            citing_id = int(edge_arr[0])
            cited_id = int(edge_arr[2].replace('\n', ''))
            cite_graph[citing_id, cited_id] = 1
            cite_graph[cited_id, citing_id] = 1
            # 测试论文的输入引用论文
            if citing_id in test_papers:
                test_papers[citing_id].add_train_cited_paper(cited_id)
        # 列平均归一化
        for a in tqdm.tqdm(range(0, total_node), total=total_node, desc="normalize"):
            col_sum = np.sum(cite_graph[:, a])
            if col_sum == 0:
                m[:, a] = 0
            else:
                m[:, a] = cite_graph[:, a] / col_sum
        # modify: 将m转为稀疏矩阵格式提高速度
        m = csr_matrix(m)

        metric = Metrics(n_list=[25, 50, 75, 100])
        # start random walk
        total = len(test_papers)
        # total = 10
        for p in tqdm.tqdm(test_papers, total=total):
            ir = np.random.rand(total_node, 1)  # 评分向量随机初始化
            d_vec = np.zeros((total_node, 1))  # 起点向量
            ir_dict = {}  # store pair of paper id and score
            sorted_ir_dict = {}
            min_ir = 9999999
            ranked_paper = []
            bpref = 0
            mrr = 0

            # 起点向量由测试论文的初始引用确定
            for c in test_papers[p].train_cited_paper:
                d_vec[c] = 1

            # 迭代90次
            d_vec = d_vec / total_node
            for r in range(50):
                # ir = alpha * np.dot(m, ir) + (1 - alpha) * (d_vec / total_node)
                # modify 稀疏矩阵向量乘法
                ir = alpha * m.dot(ir) + (1 - alpha) * d_vec

            # TODO
            ir = ir[:args.papers_num]  # 只保留paper节点的分数
            scores = ir.reshape(-1)[:args.papers_num]
            recommend_order = np.argsort(scores)[::-1]
            metric.add(recommend_order, test_papers[p].test_cited_paper)

            # ir_rank = rankdata(ir)
            # ir_rank = len(ir_rank) - ir_rank + 1  # 将升序改为倒序 评分高的靠前
            for i in range(0, len(ir)):
                ir_dict[i] = ir[i][0]  # 所有论文的评分，格式为{ id: score, id2: s2, ...}
            #
            # sorted_ir = sorted(ir_dict.items(), key=itemgetter(1), reverse=True)  # 降序排列
            all_sorted_ir_dict[test_papers[p].id] = ir_dict  # 实际未排序
            #
            # for i in range(0, len(sorted_ir)):
            #     ranked_paper.append(sorted_ir[i][0])

            # 源码中各个指标的计算
            # for pc in test_papers[p].test_cited_paper:
            #     if ir_rank[pc] < min_ir:
            #         min_ir = ir_rank[pc]
            #
            # if min_ir <= k:
            #     mrr = mrr + (1 / min_ir)
            #
            # mrr_allpaper = mrr_allpaper + mrr
            #
            # for pc in test_papers[p].test_cited_paper:
            #     r_higher = 0
            #     for rec in ranked_paper[:k]:
            #         if pc == rec:
            #             break
            #         if pc != rec and rec not in test_papers[p].test_cited_paper:
            #             r_higher = r_higher + 1
            #     bpref = bpref + (1 - (r_higher / k))
            #
            # bpref = bpref / len(test_papers[p].test_cited_paper)
            # bpref_allpaper = bpref_allpaper + bpref
            #
            # # loop to find precision,recall,F1 @k
            # for i in range(1, k + 1):
            #     pr = len(set(test_papers[p].test_cited_paper) & set(ranked_paper[:i])) / i
            #     rec = len(set(test_papers[p].test_cited_paper) & set(ranked_paper[:i])) / len(
            #         test_papers[p].test_cited_paper)
            #     F1 = (2 * pr * rec) / (pr + rec) if pr + rec > 0 else 0
            #     PrecAtK[i] = PrecAtK[i] + pr
            #     RecAtK[i] = RecAtK[i] + rec
            #     F1AtK[i] = F1AtK[i] + F1

        # mrr_allpaper = mrr_allpaper / total  # divide mrr by total citations
        # mrr5fold = mrr5fold + mrr_allpaper
        #
        # bpref_allpaper = bpref_allpaper / total
        # bpref5fold = bpref5fold + bpref_allpaper
        #
        # print(f"mrr\tbpref\t\n{mrr_allpaper}\t{bpref_allpaper}")
        #
        # for i in range(1, k + 1):
        #     PrecAtK[i] = PrecAtK[i] / total
        #     RecAtK[i] = RecAtK[i] / total
        #     F1AtK[i] = F1AtK[i] / total
        # print(f"prec@k: {PrecAtK}")
        # print(f"recall@k: {RecAtK}")
        # print(f"f1@k: {F1AtK}")

        metric.printf()

        with open('PaperRank_score/' + dataset + "_fold" + str(fold) + f'alpha={alpha}' + '.pkl', 'wb') as f:
            pickle.dump(all_sorted_ir_dict, f, pickle.HIGHEST_PROTOCOL)
        fold = fold + 1
        # print(f"219[ 0: {all_sorted_ir_dict[219][0]}, 1 {all_sorted_ir_dict[219][1]} ]")

        # for i in range(1, k + 1):
        #     pr5fold[i] = pr5fold[i] + PrecAtK[i]
        #     rec5fold[i] = rec5fold[i] + RecAtK[i]
        #     f15fold[i] = f15fold[i] + F1AtK[i]

        # print(ranked_paper)

    # for i in range(1, k + 1):
    #     pr5fold[i] = pr5fold[i] / 5.0
    #     rec5fold[i] = rec5fold[i] / 5.0
    #     f15fold[i] = f15fold[i] / 5.0
    #
    # mrr5fold = mrr5fold / 5.0
    # bpref5fold = bpref5fold / 5.0
    # #         result_file = open('Paperrank_result/'+'recommendation result_'+dataset_dir+'_fold'+str(fold)+'.txt','w')
    # #         result_file.write(result)
    # #         fold = fold+1
    # # print precision
    # print(pr5fold.values())
    # # print recall
    # print(rec5fold.values())
    # # print f1
    # print(f15fold.values())
    # # print MRR, Bpref
    # print(mrr5fold, '\n', bpref5fold)


# for dataset in ['algo_citation', 'topic_citation', 'CiteSeer_umd' , 'CiteULike']:
#     print(dataset)
# dataset = 'CiteSeer_umd' # or CiteULike, topic_citation
# train_eval_PaperRank(dataset)
# train_eval_PaperRank("./data/CiteSeer_umd_5_fold")

train_eval_PaperRank(args.data, args.name)
