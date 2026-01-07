### Time-Aware User-Based Collaborative Filtering

**Description**  
Extends user-based filtering by weighting ratings with exponential time decay.

**Steps**
1. Load ratings with timestamps.
2. Apply exponential decay to older ratings.
3. Normalize by user mean.
4. Compute similarities.
5. Predict ratings using time-weighted scores.
6. Recommend top-N items.

**Pros**
- Captures preference drift
- More relevant recommendations
- Improves dynamic scenarios

**Cons**
- Additional hyperparameters
- Higher computational cost
- Still sensitive to sparsity