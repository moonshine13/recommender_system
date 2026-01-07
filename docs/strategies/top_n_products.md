### Top-N Products (Popularity-Based)

**Description**  
Recommends globally popular products within a recent time window.

**Steps**
1. Filter recent ratings.
2. Aggregate rating counts and averages.
3. Enforce minimum rating threshold.
4. Select top-N products.

**Pros**
- Extremely fast
- No personalization required
- Robust to cold-start users

**Cons**
- No personalization
- Popularity bias
- Limited recommendation diversity