### User-Based Collaborative Filtering

**Description**
Predicts user preferences based on similarity to other users with overlapping rating histories.

**Steps**
1. Build a user–item rating matrix.
2. Normalize ratings by user mean.
3. Compute cosine similarity between users.
4. Select top-K similar users.
5. Predict ratings via weighted averaging.
6. Recommend top-N items.

**Pros**
- Simple and interpretable
- No training phase
- Adaptive to new ratings

**Cons**
- Poor scalability
- Cold-start for new users
- Ignores temporal effects
