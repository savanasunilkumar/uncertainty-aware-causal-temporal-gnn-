"""Negative sampling strategies for recommendation systems."""

import numpy as np
import torch
from typing import Dict, Set, List, Optional, Union
from collections import defaultdict


class NegativeSampler:
    """Efficient negative sampler for recommendation systems."""

    def __init__(
        self,
        num_items: int,
        user_interactions: Dict[int, Set[int]],
        device: Union[str, torch.device] = 'cpu',
        strategy: str = 'uniform',
        item_popularity: Optional[np.ndarray] = None,
    ):
        self.num_items = num_items
        self.user_interactions = user_interactions
        self.device = torch.device(device) if isinstance(device, str) else device
        self.strategy = strategy

        if strategy == 'popularity':
            if item_popularity is not None:
                self.item_probs = item_popularity / item_popularity.sum()
            else:
                item_counts = np.zeros(num_items)
                for items in user_interactions.values():
                    for item in items:
                        if item < num_items:
                            item_counts[item] += 1
                item_counts += 1
                self.item_probs = item_counts / item_counts.sum()
        else:
            self.item_probs = None

        self._all_items = np.arange(num_items)

    def sample(self, user_idx: int, num_negatives: int = 1) -> List[int]:
        positive_items = self.user_interactions.get(user_idx, set())
        neg_items = []
        max_attempts = num_negatives * 20
        attempts = 0

        while len(neg_items) < num_negatives and attempts < max_attempts:
            if self.strategy == 'popularity' and self.item_probs is not None:
                neg_item = np.random.choice(self.num_items, p=self.item_probs)
            else:
                neg_item = np.random.randint(0, self.num_items)

            if neg_item not in positive_items and neg_item not in neg_items:
                neg_items.append(neg_item)
            attempts += 1

        while len(neg_items) < num_negatives:
            neg_item = np.random.randint(0, self.num_items)
            if neg_item not in neg_items:
                neg_items.append(neg_item)

        return neg_items

    def sample_batch(
        self,
        user_indices: Union[List[int], np.ndarray, torch.Tensor],
        num_negatives: int = 1,
        return_tensor: bool = True,
    ) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(user_indices, torch.Tensor):
            user_indices = user_indices.cpu().numpy()

        batch_size = len(user_indices)
        neg_items = np.zeros((batch_size, num_negatives), dtype=np.int64)

        for i, user_idx in enumerate(user_indices):
            neg_items[i] = self.sample(int(user_idx), num_negatives)

        if return_tensor:
            return torch.tensor(neg_items, dtype=torch.long, device=self.device)
        return neg_items

    def sample_vectorized(
        self,
        user_indices: torch.Tensor,
        num_negatives: int = 1,
    ) -> torch.Tensor:
        batch_size = user_indices.size(0)

        if self.strategy == 'popularity' and self.item_probs is not None:
            probs = torch.tensor(self.item_probs, device=self.device)
            neg_items = torch.multinomial(
                probs.expand(batch_size, -1),
                num_negatives,
                replacement=False
            )
        else:
            neg_items = torch.randint(
                0, self.num_items,
                (batch_size, num_negatives),
                device=self.device
            )

        return neg_items


class HardNegativeSampler(NegativeSampler):
    """Hard negative sampler that samples items similar to positives."""

    def __init__(
        self,
        num_items: int,
        user_interactions: Dict[int, Set[int]],
        item_embeddings: Optional[torch.Tensor] = None,
        device: Union[str, torch.device] = 'cpu',
        num_candidates: int = 100,
        temperature: float = 1.0,
    ):
        super().__init__(num_items, user_interactions, device, strategy='uniform')
        self.item_embeddings = item_embeddings
        self.num_candidates = num_candidates
        self.temperature = temperature

        if item_embeddings is not None:
            self._compute_similarities()

    def _compute_similarities(self):
        if self.item_embeddings is None:
            self.item_similarities = None
            return

        embeddings = self.item_embeddings
        norms = embeddings.norm(dim=1, keepdim=True)
        normalized = embeddings / (norms + 1e-8)

        if self.num_items <= 10000:
            self.item_similarities = torch.mm(normalized, normalized.t())
        else:
            self.item_similarities = None
            self.normalized_embeddings = normalized

    def sample_hard_negatives(
        self,
        user_idx: int,
        pos_item_idx: int,
        num_negatives: int = 1,
    ) -> List[int]:
        positive_items = self.user_interactions.get(user_idx, set())

        if self.item_embeddings is None:
            return self.sample(user_idx, num_negatives)

        if self.item_similarities is not None:
            similarities = self.item_similarities[pos_item_idx]
        else:
            pos_emb = self.normalized_embeddings[pos_item_idx:pos_item_idx+1]
            similarities = torch.mm(pos_emb, self.normalized_embeddings.t()).squeeze()

        mask = torch.zeros(self.num_items, device=self.device)
        for item in positive_items:
            if item < self.num_items:
                mask[item] = float('-inf')
        similarities = similarities + mask

        top_k = min(self.num_candidates, self.num_items)
        _, top_indices = torch.topk(similarities, top_k)

        top_similarities = similarities[top_indices]
        probs = torch.softmax(top_similarities / self.temperature, dim=0)

        sampled_indices = torch.multinomial(probs, min(num_negatives, top_k), replacement=False)
        neg_items = top_indices[sampled_indices].cpu().tolist()

        while len(neg_items) < num_negatives:
            neg_items.extend(self.sample(user_idx, num_negatives - len(neg_items)))

        return neg_items[:num_negatives]


class MixedNegativeSampler(NegativeSampler):
    """Mixed negative sampler combining hard negatives with random negatives."""

    def __init__(
        self,
        num_items: int,
        user_interactions: Dict[int, Set[int]],
        device: Union[str, torch.device] = 'cpu',
        hard_ratio: float = 0.5,
        pool_size: int = 100,
        item_popularity: Optional[np.ndarray] = None,
    ):
        super().__init__(num_items, user_interactions, device, strategy='uniform')
        self.hard_ratio = hard_ratio
        self.pool_size = min(pool_size, num_items)

        if item_popularity is not None:
            self.item_popularity = item_popularity
        else:
            self.item_popularity = np.zeros(num_items)
            for items in user_interactions.values():
                for item in items:
                    if item < num_items:
                        self.item_popularity[item] += 1

        self.popular_items = np.argsort(self.item_popularity)[-self.pool_size:][::-1]

        popular_counts = self.item_popularity[self.popular_items]
        popular_counts = popular_counts + 1
        self.hard_probs = popular_counts / popular_counts.sum()

    def sample(self, user_idx: int, num_negatives: int = 1) -> List[int]:
        positive_items = self.user_interactions.get(user_idx, set())

        num_hard = int(num_negatives * self.hard_ratio)
        num_random = num_negatives - num_hard

        neg_items = []
        max_attempts = num_negatives * 20

        attempts = 0
        while len(neg_items) < num_hard and attempts < max_attempts:
            idx = np.random.choice(len(self.popular_items), p=self.hard_probs)
            neg_item = self.popular_items[idx]
            if neg_item not in positive_items and neg_item not in neg_items:
                neg_items.append(neg_item)
            attempts += 1

        attempts = 0
        while len(neg_items) < num_hard + num_random and attempts < max_attempts:
            neg_item = np.random.randint(0, self.num_items)
            if neg_item not in positive_items and neg_item not in neg_items:
                neg_items.append(neg_item)
            attempts += 1

        while len(neg_items) < num_negatives:
            neg_item = np.random.randint(0, self.num_items)
            if neg_item not in neg_items:
                neg_items.append(neg_item)

        return neg_items

    def sample_batch(
        self,
        user_indices: Union[List[int], np.ndarray, torch.Tensor],
        num_negatives: int = 1,
        return_tensor: bool = True,
    ) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(user_indices, torch.Tensor):
            user_indices = user_indices.cpu().numpy()

        batch_size = len(user_indices)
        neg_items = np.zeros((batch_size, num_negatives), dtype=np.int64)

        for i, user_idx in enumerate(user_indices):
            neg_items[i] = self.sample(int(user_idx), num_negatives)

        if return_tensor:
            return torch.tensor(neg_items, dtype=torch.long, device=self.device)
        return neg_items


class DynamicNegativeSampler:
    """Dynamic negative sampler that updates item popularity during training."""

    def __init__(
        self,
        num_items: int,
        device: Union[str, torch.device] = 'cpu',
        decay: float = 0.99,
    ):
        self.num_items = num_items
        self.device = torch.device(device) if isinstance(device, str) else device
        self.decay = decay

        self.popularity = np.ones(num_items)
        self.user_history = defaultdict(set)

    def update(self, user_idx: int, item_idx: int):
        """Update sampler with a new interaction."""
        self.user_history[user_idx].add(item_idx)
        self.popularity[item_idx] += 1
        self.popularity *= self.decay

    def sample(self, user_idx: int, num_negatives: int = 1) -> List[int]:
        """Sample negative items avoiding user's history."""
        positive_items = self.user_history.get(user_idx, set())
        neg_items = []
        max_attempts = num_negatives * 20

        probs = self.popularity / self.popularity.sum()

        attempts = 0
        while len(neg_items) < num_negatives and attempts < max_attempts:
            neg_item = np.random.choice(self.num_items, p=probs)
            if neg_item not in positive_items and neg_item not in neg_items:
                neg_items.append(neg_item)
            attempts += 1

        while len(neg_items) < num_negatives:
            neg_item = np.random.randint(0, self.num_items)
            if neg_item not in neg_items:
                neg_items.append(neg_item)

        return neg_items
