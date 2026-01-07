import numpy as np

class TimeSVDppModel:
    def __init__(self, n_users, n_items, n_factors=10):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors

        # Biases
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)
        self.alpha_u = np.zeros(n_users)
        self.beta_bin = np.zeros(n_items)

        # Latent factors
        self.p = np.random.normal(scale=0.01, size=(n_users, n_factors))
        self.q = np.random.normal(scale=0.01, size=(n_items, n_factors))
        self.y = np.random.normal(scale=0.01, size=(n_items, n_factors))

        # Implicit feedback
        self.user_rated_items = [[] for _ in range(n_users)]
        self.sqrt_Nu = np.ones(n_users)
        self.user_mean_time = np.zeros(n_users)

        # Global mean and time mean
        self.mu = 0
        self.time_mean = 0

    def predict(self, u, i, t):
        if u == -1 and i == -1:
            return self.mu
        if u == -1:
            return self.mu + self.bi[i]
        if i == -1:
            return self.mu + self.bu[u]

        dev_u = t - self.user_mean_time[u]
        sum_y = (
            np.sum(self.y[self.user_rated_items[u]], axis=0) / self.sqrt_Nu[u]
            if self.user_rated_items[u]
            else np.zeros(self.n_factors)
        )

        return (
            self.mu
            + self.bu[u]
            + self.alpha_u[u] * dev_u
            + self.bi[i]
            + np.dot(self.q[i], self.p[u] + sum_y)
        )


class TimeSVDppTrainer:
    def __init__(self, model, lr=0.001, reg=0.05, n_epochs=10):
        self.model = model
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs

    def rmse(self, preds, truths):
        return np.sqrt(np.mean((preds - truths) ** 2))

    def fit(self, train_ratings, test_ratings=None):
        train_ratings = np.array(train_ratings, dtype=float)
        train_ratings[:, 0] = train_ratings[:, 0].astype(int)
        train_ratings[:, 1] = train_ratings[:, 1].astype(int)

        self.model.mu = train_ratings[:, 2].mean()
        self.model.time_mean = train_ratings[:, 3].mean()

        # Initialize user -> rated items and user mean time
        n_users = self.model.n_users
        self.model.user_rated_items = [[] for _ in range(n_users)]
        for u, i in train_ratings[:, :2].astype(int):
            self.model.user_rated_items[u].append(i)
        self.model.sqrt_Nu = np.array(
            [np.sqrt(len(r)) if len(r) > 0 else 1.0 for r in self.model.user_rated_items]
        )
        self.model.user_mean_time = np.array(
            [
                np.mean(train_ratings[train_ratings[:, 0] == u, 3])
                for u in range(n_users)
            ]
        )

        # Training loop
        for epoch in range(self.n_epochs):
            for idx in range(len(train_ratings)):
                u, i, r, t = train_ratings[idx].astype(float)
                u, i = int(u), int(i)

                dev_u = t - self.model.user_mean_time[u]

                sum_y = (
                    np.sum(self.model.y[self.model.user_rated_items[u]], axis=0)
                    / self.model.sqrt_Nu[u]
                    if self.model.user_rated_items[u]
                    else np.zeros(self.model.n_factors)
                )

                pred = self.model.predict(u, i, t)
                err = np.clip(r - pred, -5, 5)

                # Update parameters
                p_u_old = self.model.p[u].copy()
                self.model.bu[u] += self.lr * (err - self.reg * self.model.bu[u])
                self.model.bi[i] += self.lr * (err - self.reg * self.model.bi[i])
                self.model.alpha_u[u] += self.lr * (err * dev_u - self.reg * self.model.alpha_u[u])
                self.model.beta_bin[i] += self.lr * (err - self.reg * self.model.beta_bin[i])
                self.model.p[u] += self.lr * (err * self.model.q[i] - self.reg * self.model.p[u])
                self.model.q[i] += self.lr * (err * (p_u_old + sum_y) - self.reg * self.model.q[i])
                if self.model.user_rated_items[u]:
                    grad_y = err * self.model.q[i] / self.model.sqrt_Nu[u]
                    for j in self.model.user_rated_items[u]:
                        self.model.y[j] += self.lr * grad_y
                        self.model.y[j] *= 1 - self.lr * self.reg

            # RMSE
            train_preds = np.array([self.model.predict(int(r[0]), int(r[1]), r[3]) for r in train_ratings])
            train_rmse = self.rmse(train_preds, train_ratings[:, 2])
            if test_ratings is not None:
                test_preds = np.array([self.model.predict(int(r[0]), int(r[1]), r[3]) for r in test_ratings])
                test_rmse = self.rmse(test_preds, test_ratings[:, 2])
                print(f"Epoch {epoch+1}: Train RMSE={train_rmse:.4f}, Test RMSE={test_rmse:.4f}")
            else:
                print(f"Epoch {epoch+1}: Train RMSE={train_rmse:.4f}")


class Evaluator:
    @staticmethod
    def rmse(preds, truths):
        return np.sqrt(np.mean((preds - truths) ** 2))

    @staticmethod
    def mae(preds, truths):
        return np.mean(np.abs(preds - truths))