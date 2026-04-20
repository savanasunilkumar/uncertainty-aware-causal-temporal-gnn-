import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm


class Evaluator:

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = torch.device(device)
    
    def evaluate(self, eval_data, user_interactions, k_values=[5, 10, 20], batch_size=1024):
        self.model.eval()

        metrics = {
            'precision': {k: 0.0 for k in k_values},
            'recall': {k: 0.0 for k in k_values},
            'ndcg': {k: 0.0 for k in k_values},
            'hit_ratio': {k: 0.0 for k in k_values}
        }

        # Vectorized user -> list(items). Avoids per-row iterrows().
        user_test_items = defaultdict(list)
        grouped = eval_data.groupby('user_idx')['item_idx'].apply(
            lambda s: s.astype(int).tolist()
        )
        user_test_items.update(grouped.to_dict())

        test_users = list(user_test_items.keys())
        n_batches = (len(test_users) + batch_size - 1) // batch_size
        user_batches = np.array_split(test_users, n_batches)

        with torch.no_grad():
            for user_batch in tqdm(user_batches, desc="Evaluating", leave=False):
                user_indices = torch.tensor(user_batch, dtype=torch.long, device=self.device)

                excluded_items = {}
                for user_idx in user_batch:
                    excluded_items[user_idx] = list(user_interactions[user_idx])

                top_indices, top_scores = self.model.recommend_items(
                    user_indices, top_k=max(k_values), excluded_items=excluded_items
                )

                top_indices = top_indices.cpu().numpy()

                for i, user_idx in enumerate(user_batch):
                    true_items = user_test_items[user_idx]
                    recommended_items = top_indices[i]

                    for k in k_values:
                        top_k_items = recommended_items[:k]

                        precision = len(set(top_k_items) & set(true_items)) / k
                        metrics['precision'][k] += precision

                        recall = len(set(top_k_items) & set(true_items)) / len(true_items) if true_items else 0
                        metrics['recall'][k] += recall

                        hit_ratio = 1.0 if len(set(top_k_items) & set(true_items)) > 0 else 0.0
                        metrics['hit_ratio'][k] += hit_ratio

                        dcg = 0.0
                        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(true_items), k)))

                        for j, item in enumerate(top_k_items):
                            if item in true_items:
                                dcg += 1.0 / np.log2(j + 2)

                        ndcg = dcg / idcg if idcg > 0 else 0
                        metrics['ndcg'][k] += ndcg

        num_users = len(test_users)
        for metric in metrics:
            for k in k_values:
                metrics[metric][k] /= num_users

        return metrics
    
    @staticmethod
    def compute_diversity(recommendations, item_features=None):
        all_items = set()
        for rec_list in recommendations:
            all_items.update(rec_list)

        diversity = len(all_items) / (len(recommendations) * len(recommendations[0]))
        return diversity

    @staticmethod
    def compute_coverage(recommendations, num_items):
        all_items = set()
        for rec_list in recommendations:
            all_items.update(rec_list)

        coverage = len(all_items) / num_items
        return coverage

