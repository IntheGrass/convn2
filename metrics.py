# 计算各个推荐系统性能指标

import numpy as np


class Metrics:
    # Calculate the mean metrics of recommend results
    def __init__(self, n_list: list = [25, 50, 75, 100]):
        self.n_list = n_list
        self.aps = []  # the list of ap
        self.recalls_list = []  # the list of recalls based on n_list
        self.rrs = []  # the list of rr

    def add(self, paper_order: list, target_idxes: list):
        metrics = MetricsItem(paper_order, target_idxes)
        self.aps.append(metrics.ap)
        self.recalls_list.append(metrics.recalls(self.n_list))
        self.rrs.append(metrics.rr)

    def mean_recalls(self) -> list:
        # Mean Recall basen on n_list
        return np.mean(self.recalls_list, axis=0)

    def map(self) -> float:
        # MAP: Mean Average Precision
        return np.mean(self.aps)

    def mrr(self) -> float:
        # MRR: Mean Reciprocal Rank
        return np.mean(self.rrs)

    def clear(self):
        self.aps = []
        self.recalls_list = []
        self.rrs = []

    def __str__(self):
        map_line = f'MAP:\t{self.map()}'
        mrr_line = f'MRR:\t{self.mrr()}'
        n_list_line = '\t\t'.join([str(n) for n in self.n_list])
        recalls_head = f'Recall:\t{n_list_line}'
        # recall keep four decimal places
        recalls_line = '\t\t' + '\t'.join([str(round(recall, 4)) for recall in self.mean_recalls()])
        return '\n'.join([map_line, mrr_line, recalls_head, recalls_line])

    def printf(self):
        print(self.info())

    def info(self):
        head = ['MAP', 'MRR']
        values = [str(self.map()), str(self.mrr())]
        for n in self.n_list:
            head.append(f'R{str(n)}')
        for recall in self.mean_recalls():
            values.append(str(round(recall, 6)))
        head_str = '\t\t'.join(head)
        values_str = '\t'.join(values)
        return '\n'.join([head_str, values_str])


class MetricsItem:
    # Calculate the metrics of a single recommendation result
    def __init__(self, recommend_order: list, positive_idxes: list):
        # recommend result. paper index sorted by recommended priority
        self.recommend_order = recommend_order
        # positive sample set
        self.positive_set = set(positive_idxes)

    @property
    def ap(self):
        # Average precision
        AP = 0
        num_target = len(self.positive_set)
        num_success = 0
        for i, paper in enumerate(self.recommend_order):
            if paper in self.positive_set:
                num_success += 1
                AP += num_success / (i + 1)
        AP = AP / num_target
        return AP

    def recall(self, n: int) -> float:
        # 召回率
        success_num = 0
        for paper in self.recommend_order[0: n]:
            if paper in self.positive_set:
                success_num += 1
        recall = success_num / len(self.positive_set)
        return recall

    def recalls(self, n_list: list) -> list:
        return [self.recall(n) for n in n_list]

    @property
    def rr(self):
        # Reciprocal Rank
        rank_first = 0
        for recommend_id in self.recommend_order:
            rank_first += 1
            if recommend_id in self.positive_set:
                return 1 / rank_first
        return 0


if __name__ == '__main__':
    # test metrics
    targets = [5, 2, 1, 3, 4, 8]
    recommend_result = [11, 12, 90, 5, 4]
    recommend_result2 = [11, 12, 5, 4, 1]
    N = [4, 5]
    test_metrics = Metrics(n_list=N)
    test_metrics.add(recommend_result, targets)
    test_metrics.add(recommend_result2, targets)

    test_metrics.printf()
