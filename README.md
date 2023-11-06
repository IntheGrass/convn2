# ConvCN

This Github repo stores code used in the paper: ConvCN: Enhancing Graph Based Citation Recommendation Using Citation Network Embedding

Step to reproduce experiment result

For Citation Network Embedding Experiment

   1. run train_ConvCN_Citation_Network_Embedding.py
   2. run eval_Citation_Network_embedding.py

For Citation Recommendation Experiment

   1. run train_ConvCN_Citation_Recommendation.py
   2. run code in PaperRank.ipynb to get recommendation result from PaperRank
   3. run code in CF method directory (just run Collaborative Filtering.ipynb by using user-based method) to get recommendation  result from CF
   3. run eval_ConvCN_CF_weighted_avg.py (for ConvCN-CF model) and eval_ConvCN_PaperRank.py (for ConvCN-PR model)

Note that data can be found in Dataset directory.

I recommend to run these code in GPU.


# 训练自定义数据集
前置条件:
需要将数据集放入./data/{dataset}目录下，其中dataset表示数据集名称
1. 训练ConvCN模型
```shell
python train_ConvCN_Citation_Recommendation.py --name=aan_pw --batch_size=4096
```
2. 获取PaperRank分数
```shell
python PaperRank.py --name=aan_pw --alpha=0.65
```
3. 评估ConvCN-PR混合结果
```shell
python eval_ConvCN_PaperRank_weighted_avg.py --name=dblp_pw --alpha=0.8 --alpha-pr=0.65 --papers_num=60407
```