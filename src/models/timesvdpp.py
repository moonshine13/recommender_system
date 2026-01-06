"""
TimeSVD++ model implementation for recommendation systems.

This module provides classes and functions to train and predict
user-item ratings using the TimeSVD++ collaborative filtering model,
which incorporates temporal dynamics to improve recommendation accuracy.

Example:
    from src.models.timesvdpp import TimeSVDPP
    model = TimeSVDPP()
    model.fit(ratings_data)
    predictions = model.predict(user_id, item_id)
"""

import numpy as np


class TimeSVDppVectorized:
    """
    Vectorized implementation of the TimeSVD++ algorithm for collaborative filtering.

    This class implements the TimeSVD++ model to predict user-item ratings while
    accounting for temporal dynamics and implicit feedback. The model incorporates:
    - User and item biases
    - Time-dependent user biases
    - Implicit feedback from items rated by the user
    - Latent factor decomposition for users and items

    Attributes:
        n_factors (int): Number of latent factors.
        n_epochs (int): Number of training epochs.
        lr (float): Learning rate for gradient updates.
        reg (float): Regularization term for parameters.
        n_users (int): Number of users in the dataset.
        n_items (int): Number of items in the dataset.
        mu (float): Global mean rating.
        time_mean (float): Mean timestamp across all ratings.
        bu (np.ndarray): User bias vector.
        bi (np.ndarray): Item bias vector.
        alpha_u (np.ndarray): Time-dependent user bias.
        beta_bin (np.ndarray): Time-dependent item bias (optional).
        p (np.ndarray): User latent factors.
        q (np.ndarray): Item latent factors.
        y (np.ndarray): Implicit feedback latent factors.
        user_rated_items (list[list[int]]): Items rated by each user.
        sqrt_Nu (np.ndarray): Normalization factor for implicit feedback.
        user_mean_time (np.ndarray): Mean timestamp of ratings per user.
    """

    def __init__(self, n_factors=10, n_epochs=10, lr=0.001, reg=0.05):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.n_users = None
        self.n_items = None
        self.mu = None
        self.time_mean = None
        self.bu = None
        self.bi = None
        self.alpha_u = None
        self.beta_bin = None
        self.p = None
        self.q = None
        self.y = None
        self.user_rated_items = None
        self.sqrt_Nu = None
        self.user_mean_time = None

    def fit(self, train_ratings, test_ratings=None):
        """
        Train the TimeSVD++ model on the provided ratings data.

        Args:
            train_ratings (np.ndarray or list of lists): Training data with columns
                [user_id, item_id, rating, timestamp].
            test_ratings (np.ndarray or list of lists, optional): Optional test data
                for evaluation. Defaults to None.

        Prints:
            Epoch-wise RMSE for training and test data (if provided).
        """
        train_ratings = np.array(train_ratings, dtype=float)
        train_ratings[:, 0] = train_ratings[:, 0].astype(int)
        train_ratings[:, 1] = train_ratings[:, 1].astype(int)

        self.n_users = int(train_ratings[:, 0].max() + 1)
        self.n_items = int(train_ratings[:, 1].max() + 1)
        self.mu = train_ratings[:, 2].mean()
        self.time_mean = train_ratings[:, 3].mean()

        # Initialize parameters
        self.bu = np.zeros(self.n_users)
        self.bi = np.zeros(self.n_items)
        self.alpha_u = np.zeros(self.n_users)
        self.beta_bin = np.zeros(self.n_items)
        self.p = np.random.normal(scale=0.01, size=(self.n_users, self.n_factors))
        self.q = np.random.normal(scale=0.01, size=(self.n_items, self.n_factors))
        self.y = np.random.normal(scale=0.01, size=(self.n_items, self.n_factors))

        def rmse(preds, truths):
            """Calculate RMSE"""
            return np.sqrt(np.mean((preds - truths) ** 2))

        # User -> rated items
        self.user_rated_items = [[] for _ in range(self.n_users)]
        for u, i in train_ratings[:, :2].astype(int):
            self.user_rated_items[u].append(i)

        self.sqrt_Nu = np.array(
            [np.sqrt(len(r)) if len(r) > 0 else 1.0 for r in self.user_rated_items]
        )
        self.user_mean_time = np.array(
            [
                np.mean(train_ratings[train_ratings[:, 0] == u, 3])
                for u in range(self.n_users)
            ]
        )

        # Training loop
        for epoch in range(self.n_epochs):
            for idx in range(len(train_ratings)):
                u = int(train_ratings[idx, 0])
                i = int(train_ratings[idx, 1])
                r = train_ratings[idx, 2]
                t = train_ratings[idx, 3]

                dev_u = t - self.user_mean_time[u]

                # Implicit feedback
                if len(self.user_rated_items[u]) > 0:
                    sum_y = (
                        np.sum(self.y[self.user_rated_items[u]], axis=0)
                        / self.sqrt_Nu[u]
                    )
                else:
                    sum_y = np.zeros(self.n_factors)

                # Prediction
                pred = (
                    self.mu
                    + self.bu[u]
                    + self.alpha_u[u] * dev_u
                    + self.bi[i]
                    + self.beta_bin[i]
                    + np.dot(self.q[i], self.p[u] + sum_y)
                )
                err = r - pred
                err = np.clip(err, -5, 5)
                if np.isnan(err):
                    break

                # Update parameters
                p_u_old = self.p[u].copy()
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                self.alpha_u[u] += self.lr * (err * dev_u - self.reg * self.alpha_u[u])
                self.beta_bin[i] += self.lr * (err - self.reg * self.beta_bin[i])
                self.p[u] += self.lr * (err * self.q[i] - self.reg * self.p[u])
                self.q[i] += self.lr * (err * (p_u_old + sum_y) - self.reg * self.q[i])
                if len(self.user_rated_items[u]) > 0:
                    grad_y = err * self.q[i] / self.sqrt_Nu[u]
                    for j in self.user_rated_items[u]:
                        self.y[j] += self.lr * grad_y
                        self.y[j] *= 1 - self.lr * self.reg

            # Compute RMSE
            train_preds = np.array(
                [self.predict(int(r[0]), int(r[1]), r[3]) for r in train_ratings]
            )
            train_rmse = rmse(train_preds, train_ratings[:, 2])
            if test_ratings is not None:
                test_preds = np.array(
                    [self.predict(int(r[0]), int(r[1]), r[3]) for r in test_ratings]
                )
                test_rmse = rmse(test_preds, test_ratings[:, 2])
                print(
                    f"Epoch {epoch+1}: Train RMSE={train_rmse:.4f}, Test RMSE={test_rmse:.4f}"
                )
            else:
                print(f"Epoch {epoch+1}: Train RMSE={train_rmse:.4f}")

    def predict(self, u, i, t):
        """
        Predict the rating for a given user-item pair at a specific timestamp.

        This method handles cold users and cold items:
            - If both user and item are unknown, returns the global mean rating.
            - If the user is unknown, returns global mean plus item bias.
            - If the item is unknown, returns global mean plus user bias.

        The prediction is computed using the TimeSVD++ model:
            pred = global mean + user bias + time-dependent user bias
                + item bias + dot(item latent factors, user latent factors + implicit feedback)

        Args:
            u (int): User ID. Use -1 for unknown/cold users.
            i (int): Item ID. Use -1 for unknown/cold items.
            t (float): Timestamp of the rating (used for time-dependent biases).

        Returns:
            float: Predicted rating for the user-item pair in a given timestamp.
        """
        # Cold user & cold item
        if u == -1 and i == -1:
            return self.mu

        # Cold user
        if u == -1:
            return self.mu + self.bi[i]

        # Cold item
        if i == -1:
            return self.mu + self.bu[u]

        dev_u = t - self.user_mean_time[u]

        if self.user_rated_items[u]:
            sum_y = np.sum(self.y[self.user_rated_items[u]], axis=0) / self.sqrt_Nu[u]
        else:
            sum_y = np.zeros(self.n_factors)

        return (
            self.mu
            + self.bu[u]
            + self.alpha_u[u] * dev_u
            + self.bi[i]
            + np.dot(self.q[i], self.p[u] + sum_y)
        )
