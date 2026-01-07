# Data Preparation

## Overview

This document describes the data preparation pipeline used in the recommender system.
The goal of preprocessing is to ensure **data consistency, validity, and suitability** for downstream recommendation algorithms, including user-based and model-based approaches.

The pipeline operates on explicit feedback data containing user ratings, product identifiers, and timestamps.

---

## Input Data Schema

The expected input dataset is a CSV file with the following columns:

| Column Name   | Type        | Description |
|--------------|-------------|-------------|
| user_id      | string/int  | Unique identifier of the user |
| product_id   | string/int  | Unique identifier of the product |
| rating       | numeric     | Explicit rating score |
| timestamp    | integer     | Unix timestamp of the interaction |

---

## Data Cleaning Steps

### 1. File Validation

- Verify that the input file exists.
- Ensure required columns are present.
- Raise an error if validation fails.

---

### 2. Timestamp Conversion

- Convert Unix timestamps to timezone-aware datetime objects (UTC).
- Invalid or zero timestamps are replaced with the dataset minimum timestamp higher than 0.

---

### 3. Rating Validation and Imputation

#### 3.1 Invalid Ratings

- Ratings outside the valid range are treated as invalid.
- Invalid ratings are imputed using the **user's mean rating**.
- If the user has no valid ratings, the **global mean rating** is used.

---

### 4. Missing Value Handling

| Field        | Strategy |
|-------------|----------|
| rating      | User mean → Global mean |
| timestamp   | Minimum timestamp |
| user_id     | Row dropped |
| product_id  | Row dropped |

Rows missing critical identifiers are removed.

---

## Time Normalization (Model-Based Only)

For time-aware model-based recommendations:

```
t_norm = (t - t_min) / (t_max - t_min)
```

This enables temporal bias modeling in TimeSVD++.

---

## Dataset Splitting

### Leave-Last-Out Strategy

- The most recent interaction per user (if it has more than 1 rated products) is used as test data.
- Remaining interactions form the training set.
- Users with a single interaction remain in training.

**Benefits:**

- Preserves temporal order
- Prevents data leakage
- Reflects real-world usage

---

## Design Considerations

### Advantages

- Robust to noisy data
- Consistent preprocessing
- Supports time-aware modeling

### Limitations

- Simple imputation may introduce bias
- Cold-start users still require fallback strategies

---

## Summary

The data preparation pipeline ensures clean, validated, and temporally consistent data suitable for multiple recommendation approaches.
