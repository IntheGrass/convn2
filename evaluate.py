# 读取PaperRank_score与convcn_score的评分文件评估最终结果
import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from metrics import Metrics
from Paper import Paper
import pickle
import numpy as np

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--data", default="data/", help="Data sources.")
parser.add_argument("--name", default="aan_pw", help="Name of the dataset.")
parser.add_argument('--fold', default='1', help="用于确定读取的分数文件")
parser.add_argument("--alpha", default=0.9, type=float, help="PaperRank分数的占比，在[0.0, 1.0]之间")
parser.add_argument("--alpha-pr", default=0.65, type=float, help="alpha in PaperRank，用于确定读取的文件")
parser.add_argument("--papers_num", default=12390, type=int, help="论文数量，指定实际待排序节点的id范围")

args = parser.parse_args()
print(args)

# 初始化参数
alpha = float(args.alpha)

# 1. 读取测试集
test_papers = {}  # 每一项为 pid: Paper对象

with open(args.data+'/'+args.name+'/'+'test_fold'+args.fold+'.txt') as f:
    lines = f.readlines()

for line in lines:
    xt = line.replace('\n','').split(sep='\t')
    citing_id = int(xt[0])
    cited_id = int(xt[2])
    if citing_id not in test_papers:
        test_papers[citing_id] = Paper(citing_id)
    test_papers[citing_id].add_test_cited_paper(int(cited_id))

# 2. 读取训练集
with open(args.data+'/'+args.name+'/'+'train_fold'+args.fold+'.txt') as f:
    lines = f.readlines()

for line in lines:
    xt = line.replace('\n','').split(sep='\t')
    citing_id = int(xt[0])
    cited_id = int(xt[2])
    if citing_id in test_papers:
        test_papers[citing_id].add_train_cited_paper(int(cited_id))

# 3. 分别读取PaperRank与ConvCN的候选论文分数
with open('PaperRank_score/' + args.name + "_fold" + args.fold + f'alpha={args.alpha_pr}' + '.pkl', 'rb') as f:
    all_PaperRank_score = pickle.load(f)  # 一个字典结构，存储格式为 {test_paper: {id: score, ...}...}

with open('convcn_score/' + args.name + "_fold" + args.fold + '.pkl', 'rb') as f:
    all_ConvCN_score = pickle.load(f)

metrics = Metrics()
# 4. 评估融合结果
for p in tqdm.tqdm(test_papers.keys(), total=len(test_papers)):
    citing_id = test_papers[p].id
    cited_ids = test_papers[p].test_cited_paper
    combined_score = {}
    ConvCN_score = all_ConvCN_score[p]  # 归一化后的ConvCN_score

    PaperRank_score = all_PaperRank_score[p]
    for cited_paper in ConvCN_score.keys():
        if str(cited_paper) not in PaperRank_score:
            PaperRank_score[str(cited_paper)] = 0.0

    PaperRank_score_val = list(np.float_(list(PaperRank_score.values())))  # list(np.float_(list_name))

    PaperRank_score_key = list(PaperRank_score.keys())
    PaperRank_score_normalized_val = (PaperRank_score_val - np.min(PaperRank_score_val)) / np.ptp(
        PaperRank_score_val) if np.ptp(PaperRank_score_val) != 0 else [0] * len(
        PaperRank_score_val)  # score of PaperRank is normalized here...

    for i in range(0, len(PaperRank_score)):
        # 将归一化结果赋值给原始的PaperRank_score
        PaperRank_score[PaperRank_score_key[i]] = PaperRank_score_normalized_val[i]

    # 计算融合分数
    for cited_paper in ConvCN_score.keys():
        ConvCN_scr = ConvCN_score[cited_paper]
        PaperRank_scr = PaperRank_score[cited_paper]
        combined_score[cited_paper] = alpha * PaperRank_scr + (1 - alpha) * ConvCN_scr  # weighted average

    scores = np.zeros(len(args.papers_num))
    for pid in combined_score:
        scores[pid] = combined_score[pid]
    paper_order = np.argsort(scores)[::-1]  # Index of papers in descending order
    metrics.add(paper_order, cited_ids)

metrics.printf()  # 打印各项指标







