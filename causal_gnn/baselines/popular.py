import numpy as np
import torch
from collections import Counter


class PopularItems:

    def __init__(self, num_users, num_items):
        self.num_users = num_users
        self.num_items = num_items
        self.item_counts = None
        self.popular_items = None
        
    def fit(self, train_data):
        item_indices = train_data['item_idx'].values
        self.item_counts = Counter(item_indices)
        self.popular_items = [item for item, _ in self.item_counts.most_common()]

        print(f"PopularItems: Trained on {len(train_data)} interactions")
        print(f"  Most popular item: {self.popular_items[0]} ({self.item_counts[self.popular_items[0]]} interactions)")
        
    def predict(self, user_idx, top_k=10, exclude_items=None):
        if self.popular_items is None:
            raise ValueError("Model must be fitted before prediction")

        recommendations = []
        for item in self.popular_items:
            if exclude_items and item in exclude_items:
                continue
            recommendations.append((item, self.item_counts[item]))
            if len(recommendations) >= top_k:
                break

        return recommendations

    def predict_batch(self, user_indices, top_k=10, exclude_items_dict=None):
        results = {}
        for user_idx in user_indices:
            exclude = exclude_items_dict.get(user_idx, None) if exclude_items_dict else None
            results[user_idx] = self.predict(user_idx, top_k=top_k, exclude_items=exclude)
        return results

