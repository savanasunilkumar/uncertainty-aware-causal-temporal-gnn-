import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class BPR(nn.Module):

    def __init__(self, num_users, num_items, embedding_dim=64, learning_rate=0.01, reg_lambda=0.01):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda

        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, user_idx, item_idx):
        user_emb = self.user_embeddings(user_idx)
        item_emb = self.item_embeddings(item_idx)
        return (user_emb * item_emb).sum(dim=-1)

    def bpr_loss(self, user_idx, pos_item_idx, neg_item_idx):
        pos_scores = self.forward(user_idx, pos_item_idx)
        neg_scores = self.forward(user_idx, neg_item_idx)

        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

        user_emb = self.user_embeddings(user_idx)
        pos_item_emb = self.item_embeddings(pos_item_idx)
        neg_item_emb = self.item_embeddings(neg_item_idx)

        reg_loss = self.reg_lambda * (
            user_emb.norm(2).pow(2) +
            pos_item_emb.norm(2).pow(2) +
            neg_item_emb.norm(2).pow(2)
        ) / user_idx.size(0)

        return loss + reg_loss
    
    def fit(self, train_data, num_epochs=20, batch_size=256, device='cpu'):
        self.to(device)
        self.train()

        # Vectorized user -> list(items) via groupby; avoids per-row iterrows().
        user_items = (
            train_data.groupby('user_idx')['item_idx']
            .apply(lambda s: s.astype(int).tolist())
            .to_dict()
        )

        all_items = set(range(self.num_items))

        print(f"Training BPR with {len(train_data)} interactions...")

        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0

            users = list(user_items.keys())
            np.random.shuffle(users)

            pbar = tqdm(range(0, len(users), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}")

            for batch_start in pbar:
                batch_users = users[batch_start:batch_start + batch_size]

                user_idx_list = []
                pos_item_idx_list = []
                neg_item_idx_list = []

                for user in batch_users:
                    pos_items = user_items[user]
                    pos_item = np.random.choice(pos_items)

                    neg_items = list(all_items - set(pos_items))
                    neg_item = np.random.choice(neg_items)

                    user_idx_list.append(user)
                    pos_item_idx_list.append(pos_item)
                    neg_item_idx_list.append(neg_item)

                user_idx = torch.LongTensor(user_idx_list).to(device)
                pos_item_idx = torch.LongTensor(pos_item_idx_list).to(device)
                neg_item_idx = torch.LongTensor(neg_item_idx_list).to(device)

                self.optimizer.zero_grad()
                loss = self.bpr_loss(user_idx, pos_item_idx, neg_item_idx)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    def predict(self, user_idx, top_k=10, exclude_items=None, device='cpu'):
        self.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_idx]).to(device)
            item_tensor = torch.LongTensor(list(range(self.num_items))).to(device)

            user_emb = self.user_embeddings(user_tensor)
            item_embs = self.item_embeddings(item_tensor)

            scores = (user_emb @ item_embs.T).squeeze().cpu().numpy()

            if exclude_items:
                scores[list(exclude_items)] = -np.inf

            top_items = np.argsort(scores)[::-1][:top_k]
            return [(int(item), float(scores[item])) for item in top_items]

    def predict_batch(self, user_indices, top_k=10, exclude_items_dict=None, device='cpu'):
        results = {}
        for user_idx in user_indices:
            exclude = exclude_items_dict.get(user_idx, None) if exclude_items_dict else None
            results[user_idx] = self.predict(user_idx, top_k=top_k, exclude_items=exclude, device=device)
        return results

