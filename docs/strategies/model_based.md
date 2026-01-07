### Model-Based Recommendation (TimeSVD++)

**Description**  
Latent factor model with temporal dynamics and implicit feedback, trained using stochastic gradient descent.

**Steps**
1. Encode users and items.
2. Normalize timestamps.
3. Apply leave-last-out split.
4. Initialize latent factors and biases.
5. Train with SGD.
6. Predict ratings.
7. Recommend top-N items.

**Pros**
- High predictive accuracy
- Scales to large datasets
- Explicit temporal modeling

**Cons**
- Computationally expensive
- Less interpretable
- Cold-start problem remains
