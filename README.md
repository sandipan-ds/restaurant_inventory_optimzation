# Restaurant Demand Forecasting System
## End-to-End Machine Learning Pipeline — Project Report

---

## Executive Summary

This project develops a production-ready demand forecasting system for a multi-center restaurant chain. Using 145 weeks of historical order data across 77 fulfillment centers and 51 meals, an XGBoost model was trained to predict weekly order volumes at the center-meal combination level. The final model achieves a median MAPE of **29.8%** on held-out test data, with the majority of high-volume categories and meals exceeding the **80% forecast accuracy threshold**. The system also surfaces actionable business insights on promotions, pricing elasticity, and center-level demand trends.

---

## 1. Dataset Overview

| Attribute | Value |
|---|---|
| Total Records | 456,548 |
| Weeks Covered | 1 – 145 |
| Fulfillment Centers | 77 |
| Unique Meals | 51 |
| Meal Categories | 14 |
| Cuisine Types | 4 (Thai, Indian, Italian, Continental) |
| Target Variable | `num_orders` (weekly order count per center-meal pair) |

**Key columns:**

| Column | Description |
|---|---|
| `week` | Week number (1–145) |
| `center_id` | Fulfillment center identifier |
| `meal_id` | Meal identifier |
| `checkout_price` | Final price paid by the customer |
| `base_price` | Original price before any discount |
| `emailer_for_promotion` | Binary flag: email campaign active |
| `homepage_featured` | Binary flag: meal featured on homepage |
| `num_orders` | **Target** — weekly orders for this center-meal pair |
| `category` | Meal category (e.g., Rice Bowl, Sandwich, Beverages) |
| `cuisine` | Cuisine type |

No missing values were present in the dataset.

---

## 2. Exploratory Data Analysis

### 2.1 Target Distribution

The raw `num_orders` distribution is highly right-skewed, with a **mean of 262** and a **median of 136** — the long tail is driven by high-volume centers and promotional spikes. A **log(x+1) transformation** was applied during preprocessing to improve model convergence and reduce the influence of extreme values. Across cuisine types, Indian cuisine displays the widest spread and most outliers, while Continental shows the lowest and most tightly distributed order volumes.

### 2.2 Temporal Patterns

Average weekly orders fluctuate between approximately 200 and 420 orders per week, with no strong upward or downward trend over the 145-week period. Several sharp spikes in total order volume appear (notably around weeks 5, 50, and 100), likely driven by promotional events. A notable trough occurs near **week 61**, suggesting a data anomaly or operational disruption. The persistent gap between the mean and median throughout the entire time horizon confirms consistent right-skewness driven by a subset of high-demand center-meal pairs.

### 2.3 Autocorrelation Analysis (ACF / PACF)

ACF and PACF plots were generated for both the aggregated weekly series (all centers) and a representative single series (Center 10, Meal 1062).

**Key findings:**

- **AR(1) structure confirmed** — only lag 1 is statistically significant in the PACF for both series. Last week's demand is the single strongest temporal predictor.
- **Weak overall autocorrelation (~0.25–0.35)** — demand is not on autopilot. The majority of variation is driven by external factors: promotions, pricing, and meal popularity.
- **No seasonality** — no significant spikes at lags 26 or 52, confirming that cyclical week-based features (`week_sin`, `week_cos`) are sufficient and that SARIMA-type models are unnecessary.
- **Individual series are more structured than aggregates** — center-meal level patterns are learnable by tree models that use `center_id` and `meal_id` directly.

**Feature validation:** `lag_1` is justified as the primary lag feature; `roll_mean_3` and `ewm_3` provide smoothing value beyond the single-lag signal.

### 2.4 Center & Cuisine Performance

The top 20 centers by average weekly orders range from approximately 290 to 610 orders, with **Center 13** leading at ~610 and **Center 43** close behind at ~560. Italian cuisine exhibits the highest weekly order peaks and greatest volatility, with several weeks exceeding 800 average orders. Indian cuisine shows the most extreme promotional spikes early in the time series, while Continental cuisine operates at consistently lower volumes.

### 2.5 Feature Correlation

The correlation matrix reveals several important relationships:

- `checkout_price` and `base_price` are nearly collinear (**r = 0.953**), justifying the use of derived features like `discount_pct` and `price_ratio` rather than both raw prices.
- `discount_pct` has the strongest positive correlation with `emailer_for_promotion` (r = 0.478), indicating that discounts and email campaigns tend to be deployed together.
- Both promotional flags (`emailer_for_promotion`, `homepage_featured`) are positively correlated with `num_orders` (r = 0.277 and 0.294 respectively), confirming measurable uplift.
- `checkout_price` is negatively correlated with `num_orders` (r = −0.282), consistent with price-elastic demand.

### 2.6 Price Analysis

A negative price-demand relationship is visible in the scatter plot, with a fitted trend line confirming that higher checkout prices correspond to lower order counts. The majority of transactions carry **zero discount** — the discount distribution is heavily concentrated at 0%, with a thin tail extending to ~50%. Among meal categories, **Rice Bowl** and **Sandwich** generate the highest average orders per week, while Biryani and Pasta rank at the lower end.

### 2.7 Promotion Effects

Promotional channels drive substantial order volume increases:

| Promotion Scenario | Average Orders |
|---|---|
| No email, not featured | 211 |
| Homepage featured only | 456 |
| Email only | 431 |
| Both email + homepage | **816** |

Email campaigns alone lift orders from a baseline of 229 to 631 (+175%). Homepage featuring similarly lifts from 221 to 595 (+169%). Critically, **running both simultaneously is superadditive** — the combined effect (816 orders) exceeds the sum of the individual lifts, indicating a strong synergistic interaction between the two channels.

---

## 3. Feature Engineering

A total of **35 features** were constructed from the original 9 columns:

| Feature Group | Features |
|---|---|
| Original | `week`, `center_id`, `meal_id`, `checkout_price`, `base_price`, `emailer_for_promotion`, `homepage_featured` |
| Price | `discount_pct`, `price_ratio`, `is_discounted`, `price_diff`, `log_checkout_price`, `log_base_price`, `price_bin` |
| Lag & Rolling | `lag_1`, `lag_2`, `lag_3`, `roll_mean_3`, `roll_mean_6`, `roll_std_3`, `ewm_3` |
| Aggregate | `center_avg_orders`, `meal_avg_orders`, `category_avg_orders`, `cuisine_avg_orders` |
| Interaction | `promo_email_x_home`, `any_promotion`, `price_x_promo`, `discount_x_promo` |
| Temporal | `week_sin`, `week_cos`, `quarter` |
| Encoded | `category_enc`, `cuisine_enc` |

Lag features (`lag_1`–`lag_3`) and rolling statistics (`roll_mean_3`, `ewm_3`) are the most impactful predictors, as confirmed by feature importance analysis across all tree-based models.

---

## 4. Modelling Pipeline

### 4.1 Train / Test Split

A strict **temporal split** was used to prevent data leakage:

- **Training set:** Weeks 1 – 116
- **Test set:** Weeks 117 – 145

The target variable was **log-transformed** (`log1p`) for all models. For tree-based models, predictions were exponentiated back (`expm1`) before metric computation.

### 4.2 Models Trained

Four models of increasing complexity were trained and evaluated:

| Model | Notes |
|---|---|
| **Ridge Regression** | Linear baseline; scaled features; log-transformed target; hyperparameter α tuned via TimeSeriesSplit GridSearchCV |
| **Random Forest** | Non-parametric ensemble; hyperparameter tuning on `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` |
| **XGBoost** | Gradient boosting with histogram-based training; tuned learning rate, depth, subsample, colsample |
| **LightGBM** | Leaf-wise gradient boosting; tuned `num_leaves`, learning rate, regularization |

Validation used a **5-fold walk-forward (time series) cross-validation** scheme with fold boundaries at weeks 100–110, 111–118, 119–126, 127–134, and 135–145.

### 4.3 Model Comparison

Models were ranked primarily by **RMSE** and secondarily by **RMSLE**, **MAE**, and **R²**. Gradient boosting models (XGBoost, LightGBM) outperform the linear and Random Forest baselines across all metrics. **XGBoost** emerged as the best model and was selected for final forecasting and business insights.

*Note: Exact metric values are computed at runtime from the results DataFrame. Refer to the `DEMAND FORECASTING SYSTEM — FINAL MODEL SUMMARY` output block in the notebook for the precise scores.*

---

## 5. Forecast Results

### 5.1 Center-Level Forecasts (XGBoost Tuned)

Demand forecasts were visualized for four randomly sampled centers (IDs 17, 74, 29, 10) over the test period (weeks 117–145). Key observations:

- **Center 10** (the highest-volume center, ~21,000–28,000 total weekly orders) shows the tightest tracking between actual and predicted, benefiting from its high volume and stable patterns.
- **Center 17** captures the general trend but misses a sharp peak near week 134 (~10,000 orders), likely a promotional spike not fully captured by the feature set.
- **Centers 29 and 74** show moderate tracking with wider error bands at demand troughs and promotional peaks.

The model consistently underestimates sharp spikes while overestimating during troughs — a known characteristic of regression-based forecasting on right-skewed count data.

### 5.2 Forecast Accuracy by Category

All 14 categories were evaluated against an **80% accuracy threshold**:

- **High accuracy (≥ 90%):** Starters, Pasta, Other Snacks, Beverages, Pizza, Seafood, Rice Bowl, Salad, Desert, Biryani, Sandwich
- **Near threshold (80–85%):** Extras, Soup, Fish

Rice Bowl and Sandwich — the two highest-volume categories — are the most reliably forecasted. Fish, Pasta, and Seafood have the highest MAPE (85–90%), likely due to lower order volumes and higher week-to-week variability.

### 5.3 Forecast Accuracy — Top 20 Meals

Among the top 20 meals by order volume, the majority exceed the 80% accuracy threshold. **Aperol Spritz, Thai Iced Tea, and Pepperoni Feast** achieve the highest individual accuracies (approaching 100%). Only **Panini di Pollo, Caprese Salad, and Italian B.M.T.** fall slightly below 80%, likely due to more volatile demand patterns at the meal-center granularity.

---

## 6. Error Analysis

### 6.1 MAPE Distribution

The overall **median MAPE is 29.8%**, with most predictions falling in the 0–60% range. A small spike at 200% (the cap applied during analysis) reflects a tail of very low-volume center-meal pairs where even small absolute errors translate to large percentage errors.

### 6.2 Error by Order Volume

| Volume Bucket | Avg MAPE |
|---|---|
| Very Low | ~125% |
| Low | ~53% |
| Medium | ~38% |
| High | ~30% |
| Very High | ~25% |

There is a clear inverse relationship between order volume and percentage error. This is expected — low-volume combinations are inherently harder to forecast and are more sensitive to zero-order weeks. Operational decisions should weight forecast confidence by order volume bucket.

### 6.3 Error by Category

Fish, Pasta, and Seafood have the highest average MAPE (~65–90%), while Rice Bowl, Sandwich, and Salad have the lowest (~30–45%). These align with the volume-based pattern: niche categories served at fewer centers exhibit higher relative error.

---

## 7. Business Insights & Recommendations

### 7.1 Promotion Strategy

The synergistic effect of combining email and homepage promotions is the most impactful lever available:

- **Email alone:** +402 average orders per center-meal per week
- **Homepage alone:** +374 average orders per center-meal per week
- **Both combined:** +605 average orders — **1.5× more effective than either channel alone**

**Recommendation:** Always pair email campaigns with homepage featuring for maximum ROI. For budget-constrained campaigns, email alone provides slightly higher uplift than homepage alone.

### 7.2 Center Demand Trends

Comparing demand in the first half (weeks 1–72) vs. the second half (weeks 73–145):

- **Growing centers (positive H2 vs H1 growth):** Centers 52, 91, 139, 92, 80, 57, 41, 24, 67, 51 show 1–12% demand growth, with Center 52 leading at +12%.
- **Declining centers:** Centers 55, 75, 93, 106, 129, 50, 110, 152, 146, 74 all show declines of 5–27%, with Center 74 most severely impacted at approximately −27%.

**Recommendation:** Investigate operational or competitive factors at the 10 declining centers. Consider targeted promotional investment at Centers 55 and 75 (mid-tier declines) before they worsen.

### 7.3 Pricing Elasticity

The negative price-demand relationship and the high discount-promotion co-occurrence suggest price-elastic customer behavior. Discounts predominantly accompany active email campaigns (correlation 0.478), meaning customers are being incentivized by both channel messaging and price reduction simultaneously — making it difficult to disentangle the individual effects.

**Recommendation:** Run controlled experiments (A/B tests) to isolate the individual contribution of discounting from promotion channel effects.

### 7.4 Inventory Optimization Application

The trained XGBoost model can directly inform weekly inventory procurement decisions:

- Use `lag_1` (last week's orders) as the rolling anchor for short-horizon forecasts
- Apply the promotional interaction features (`promo_email_x_home`, `any_promotion`) to inflate forecasts in promotional weeks
- Apply higher safety stock for low-volume categories (Fish, Pasta, Seafood) where MAPE is 65–90%
- Retrain weekly with the latest lag features to keep rolling statistics current

---

## 8. Pipeline Summary

| Step | Details |
|---|---|
| **Data** | 456,548 records × 13 columns; 145 weeks; 77 centers; 51 meals |
| **Target** | `num_orders` — right-skewed; log1p transformation applied |
| **Split** | Temporal: weeks 1–116 train / 117–145 test |
| **Features** | 35 features: original + price, lag/rolling, interaction, aggregate, temporal, encoded |
| **Models** | Ridge, Random Forest, XGBoost, LightGBM |
| **Validation** | 5-fold walk-forward (time series) cross-validation |
| **Best Model** | XGBoost (Tuned) |
| **Test Median MAPE** | 29.8% |
| **Accuracy Threshold** | 80% — met by 11 of 14 categories and ~17 of top-20 meals |

---

## 9. Conclusions

This project demonstrates that a well-engineered gradient boosting model can deliver production-quality demand forecasts for a multi-center restaurant chain. The key technical contributions are the construction of lag/rolling features that capture the AR(1) demand structure, and the promotion interaction features that capture the synergistic channel effects observed in the data. The model generalises well to the held-out test period, with strong accuracy for high-volume categories and an interpretable error profile that directly guides inventory safety stock decisions.

**Next steps for deployment:**

1. Automate weekly retraining with new lag feature computation as orders arrive
2. Segment models by cuisine type to improve per-category precision
3. Integrate model predictions into the procurement system with automated safety stock buffers tied to the MAPE-by-volume-bucket analysis
4. Implement a real-time anomaly flag when predicted vs. actual deviation exceeds 2σ over a rolling 4-week window

---

## 10. What Next — Deployment & Extension Roadmap

> **Note on static data:** This dataset does not update over time — all 145 weeks are fixed. This changes *how* some recommendations are implemented (no live retraining pipeline, no incoming data to monitor) but it does **not** invalidate them. The trained model is fully serializable, the features are pre-computable from the existing dataset, and a deployment layer can still serve real predictions, interactive what-if queries, and a rich analytics dashboard — all from the frozen dataset and the saved model weights.

---

### 10.1 Deployment Architecture

The recommended stack is **FastAPI + Streamlit**, serving different users:

```
Saved XGBoost model (joblib .pkl)
            │
            ▼
     FastAPI service              ← /predict endpoint for programmatic use
            │
            ▼
   Streamlit dashboard            ← visual interface for ops / business teams
```

**FastAPI** is the backend inference engine. Serialize the trained XGBoost model with `joblib`, expose a `/predict` endpoint that accepts a JSON payload (center_id, meal_id, week, checkout_price, promo flags, etc.) and returns `predicted_orders`. Since data is static, you can also pre-compute predictions for all center-meal-week combinations in the test set and store them in a SQLite or Postgres table — the API then becomes a fast lookup rather than a live inference call.

**Streamlit** is the analytics layer for business users — operations managers, supply chain planners, marketing — who need to explore forecasts and insights without touching code. All the visualizations already built in the notebook translate directly into Streamlit components.

---

### 10.2 What to Build in the Streamlit Dashboard

**Forecast Monitor**
- Predicted vs. actual orders per center and meal for the held-out test period (weeks 117–145), replicating the `demand_forecast_centers.png` style with an interactive center selector
- A traffic-light accuracy table: green where forecast accuracy ≥ 80%, amber for 60–80%, red below 60% — gives planners an immediate signal on where to apply manual judgment
- MAPE distribution chart by category and by order volume bucket, directly from your error analysis

**Inventory Planning View**
- Forecasted order volume for any selected week in the test set, ranked by center and meal
- Recommended procurement quantity = `predicted_orders × (1 + safety_stock_buffer)`, where the buffer is drawn from the MAPE-by-volume-bucket analysis: ~25% buffer for Very High volume items, scaling up to ~130% for Very Low volume items
- A downloadable CSV of the full weekly procurement plan for any selected week

**Promotion What-If Simulator**
- User selects a center, a meal, a week from the test set, then toggles `emailer_for_promotion` and `homepage_featured` on or off
- The model returns a live predicted order count for each of the four promotion states (none, email only, homepage only, both)
- This is effectively free to implement — the existing feature set and trained model already encode the promotion interaction; it is just a matter of changing two input values and calling `model.predict()`
- Since data is static, you can also pre-compute all four scenarios for every center-meal-week and cache them, making the simulator instantaneous

**Business Insights Panel**
- Center-level H2 vs H1 demand growth chart (replicating `business_insights.png`), with color-coded growing vs. declining centers
- Cuisine and category heatmap of average orders across the test period vs. the training period — highlights structural demand shifts
- Top and bottom 10 centers by forecast accuracy, so operations teams know which centers need the most human oversight

---

### 10.3 What to Predict Beyond `num_orders`

The current model predicts raw order volume. Three natural extensions add direct business value without collecting new data:

**Promotional uplift prediction.** Rather than just predicting absolute `num_orders`, compute the *incremental* lift for any promotion scenario relative to the no-promotion baseline. This answers the business question "how many extra orders does this campaign generate?" rather than "how many orders will there be?" — a more actionable framing for the marketing team.

**Inventory waste / stockout risk score.** Using your MAPE-by-volume-bucket analysis, assign a confidence tier (High / Medium / Low) to every forecast. Low-confidence forecasts (Very Low and Low volume buckets, MAPE > 50%) should automatically trigger a higher safety stock recommendation. This turns the error analysis from a diagnostic into a decision-support output.

**Center health score.** Aggregate three signals per center into a single 0–100 index: forecast accuracy (from error analysis), H2 vs H1 demand trend (from business insights), and deviation from cuisine average. This gives leadership a single number to scan across all 77 centers rather than reading individual plots, and surfaces the 10 declining centers (74, 55, 146, etc.) automatically.

---

### 10.4 Metrics to Track and Display

Even with static data, tracking the right metrics makes the dashboard analytically honest and decision-ready:

| Metric | What It Tells You | Where to Show It |
|---|---|---|
| **Overall MAPE** | Average forecast error across all test center-meal pairs | Dashboard header KPI |
| **Median APE** | Robust central error (less influenced by outliers than mean MAPE) | Dashboard header KPI |
| **MAPE by volume bucket** | Where the model is reliable vs. uncertain | Inventory planning view |
| **MAPE by category** | Which menu segments need manual review | Forecast monitor |
| **Forecast accuracy % by category** | How many categories clear the 80% threshold | Traffic-light table |
| **R² on test set** | Overall explained variance | Model summary panel |
| **Promotion lift accuracy** | How well the model's promotional uplift matches actual uplift | Promotion panel |
| **Center health score** | Aggregate signal per center (accuracy + trend + deviation) | Business insights panel |
| **Predicted vs. actual order volume** | Visual alignment between forecast and ground truth | Forecast monitor |

---

### 10.5 Implementation Priority Order

Given static data and a single trained model, the recommended build sequence is:

1. **Serialize and containerize the model** — save with `joblib`, wrap in a FastAPI `/predict` endpoint, containerize with Docker. This is the foundation for everything else.
2. **Pre-compute all test-set predictions** — run inference for all center-meal-week combinations in the test set and store results in a lightweight database. This makes the Streamlit dashboard fast regardless of model complexity.
3. **Build the Streamlit Forecast Monitor and Inventory Planning view** — these have the most immediate operational value and directly use the pre-computed predictions.
4. **Add the Promotion What-If Simulator** — high value for the marketing team; pre-compute all four promotion states per row to make it instantaneous.
5. **Add the Business Insights Panel and Center Health Score** — these require aggregating across the pre-computed table, so they come naturally once the data layer is in place.

---

*Report generated from `demand_forecasting.ipynb` — End-to-End Machine Learning Pipeline*
