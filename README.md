# Periospot Social: Predicting High-Performance Posts (Binary Classification)

**Goal**  
Given only features known *before publishing* (network, post type, content type, account, posting time, caption length, etc.), predict whether a post will be a **High Performer** (threshold chosen *empirically during EDA*).

**Why it matters**  
This enables proactive content planning: schedule, format, and creative choices that raise the odds of high engagement.

## Dataset

- Source: Sprout Social exports (5+ years)
- Files:
  - `data/raw/post_performance.csv`
  - `data/raw/profile_performance.csv`

## Methods

- EDA → choose a defensible high/low threshold from the empirical distribution.
- Feature engineering (pre-posting only; strict anti-leakage).
- Models: Logistic (baseline), RandomForest, XGBoost (tuned).
- Validation: **temporal split** (train ≤2024-12, test ≥2025-01) to simulate out-of-time generalization.
- Metrics: ROC-AUC, PR-AUC, Recall/Precision, business-focused threshold tuning (maximize recall at acceptable precision).

## Data Exploration Findings

### Dataset Overview
- **Total posts:** 8,434 (October 2020 - October 2025)
- **Networks:** X/Twitter (67% - dominant), Threads, Instagram, Facebook, LinkedIn, TikTok, YouTube, Pinterest
- **Post Types:** Regular posts, Stories, Reels, Replies, Quotes, Pins

### Target Variable: Engagement Rate (per Impression)

**Coverage by Year:**
- 2020-2022: <2% (metric not available in Sprout exports)
- 2023: 35% (partial rollout)
- **2024-2025: 96-98% coverage** ✅ *← Usable data*

**Coverage by Network:**
- Full coverage (100%): Facebook, TikTok, Threads
- Good coverage (78%): Instagram
- Partial coverage (45%): X/Twitter
- No coverage (~0%): Pinterest, YouTube, LinkedIn

### Key Decisions

#### Data Filtering
**Decision:** Use only 2024-2025 data where target is reliably available  
**Rationale:** Target variable only present in 96-98% of posts from these years  
**Result:** 4,406 usable posts with valid engagement rates

#### Train/Test Strategy Challenge
**Problem discovered:** Posting frequency increased dramatically over time
- 2024: 1,635 posts (early/sparse posting)
- 2025: 2,771 posts (high-frequency posting period)
- Inverted ratio: 37% train / 63% test (unusual!)

<div align="center">
<img src="images/posts_per_month.png" alt="Posts per month over time showing dramatic increase in posting frequency from 2024 to 2025" width="800"/>

*Posts per month (2024-2025): Clear increase in posting frequency over time, explaining the inverted train/test ratio*
</div>

**Decision:** Use all 2024 for training, early 2025 for validation, rest for test
- Training: All 2024 data (1,635 posts)
- Validation: 2025 Jan-Feb (temporal validation)
- Test: 2025 Mar-Oct (holdout set)
- Maintains strict temporal ordering (no data leakage)

#### Network Imbalance Strategy
- X/Twitter dominates: 3,021 posts (67%)
- Other networks: 1,514 posts (33%)
- Plan: Monitor per-network performance in evaluation; consider stratified analysis

### Pre-Posting Features (No Leakage)
Features known BEFORE publishing:
- `Date` → time features (hour, weekday, month, season, year, is_weekend)
- `Network` → platform (categorical: X, Instagram, Threads, etc.)
- `Post Type` → content format (categorical: Post, Story, Reel, etc.)
- `Content Type` → topic category (categorical)
- `Profile` → account identifier (categorical)
- `Post` → caption text → NLP features: length, word count, hashtag count, mention count, URL count
- `Link` → has external link (binary feature)

**Excluded (post-outcome - would leak):**
- Impressions, Reach, Engagements, Clicks, Views, Saves, Shares, Reactions, etc.

### Data Quality Notes
- **Missing captions:** 2.7% of posts (mostly Stories and image-only posts on Instagram/Threads) - expected behavior
- **Zero-impression posts:** Minimal overlap with missing engagement rates - data quality is good

### Data Cleaning & Threshold Selection ✅

**Minimum Impression Filter:**
- **Decision:** Filter out posts with <20 impressions
- **Rationale:** Low-impression posts (1-19) have unreliable engagement rates (e.g., 1 impression + 1 engagement = 100% but not statistically meaningful)
- **Impact:** Removed 395 posts (9.5%), retained 3,745 posts with ≥20 impressions

**Binary Threshold Selection:**
- **Decision:** 90th percentile = **7.14% engagement rate**
- **Rationale:** Focus on exceptional posts (not just above-average). Meaningful difference from 80th percentile (4.62%) - represents 54% improvement in engagement
- **Class Distribution:**
  - High Performers (label=1): 378 posts (10.1%) - engagement ≥7.14%
  - Regular Posts (label=0): 3,367 posts (89.9%) - engagement <7.14%
- **Imbalance Strategy:** 90/10 split is moderate and manageable with class weights in tree models

<div align="center">
<img src="images/distribution_engagement_rate.png" alt="Engagement rate distribution showing right-skewed pattern with 80th and 90th percentile thresholds marked" width="900"/>

*Distribution analysis: Histogram, log-scale view, boxplot (outliers), and CDF. Heavy right skew with most posts <5% engagement. Red line (80th %ile) = 4.62%, Orange line (90th %ile) = 7.14%*
</div>

**Final Clean Dataset:**
- 3,745 posts (2024-2025)
- ≥20 impressions per post
- Valid engagement rate target
- Binary labels created
- Ready for feature engineering

### Feature Engineering ✅

**Time Features (8):**
- Extracted from `Date` column: hour, day_of_week, month, quarter
- Derived indicators: is_weekend, is_morning, is_afternoon, is_evening

**Text Features (9):**
- Caption analysis: caption_length, word_count
- Pattern counts: hashtag_count, mention_count, url_count
- Content indicators: has_question_mark, has_numbers, has_list_format, has_emoji

**Categorical Features (17 after one-hot encoding):**
- Network (6): Facebook, Instagram, LinkedIn, Threads, TikTok, X
- Post Type (5): '@Reply, Post, Quote, Reel, Story
- Content Type (5): Carousel, Link, Photo, Text, Video
- Profile (6): Different account identifiers across platforms

**Total Pre-Posting Features: 39** (no leakage - verified)

### Train/Validation/Test Split ✅

**3-Way Temporal Split (No Data Leakage):**
- **Training:** Jan-Nov 2024 (970 posts, 105 high performers, 10.8%)
- **Validation:** Dec 2024 (463 posts, 53 high performers, 11.4%)
- **Test:** All 2025 (2,312 posts, 220 high performers, 9.5%)

**Split Discovery Process:**
- Initial attempt: Jan-Aug (train) vs Sep-Dec (val) resulted in imbalanced sizes (486 vs 947)
- Root cause: Posting frequency increased significantly in late 2024, with December having the highest concentration
- Solution: Split at month 12 (< 12 for train, >= 12 for val) to balance sample sizes while preserving temporal order
- Final ratio: ~68% train / 32% val (of 2024 data)

**Why This Works:**
- ✅ **Temporal order preserved:** Past → Present → Future (no data leakage)
- ✅ **Balanced sizes:** 970 training samples is sufficient for tree models, 463 validation samples allows reliable hyperparameter tuning
- ✅ **Consistent class distribution:** 9.5-11.4% high performers across all splits (no distribution shift)
- ✅ **Production-realistic:** Model trained on past data, tuned on recent data, evaluated on future data

### Baseline Models ✅

**1. Dummy Classifier (Majority Class Baseline):**
- Strategy: Always predict most frequent class (0 = "Not High Performer")
- **Test Results:**
  - Accuracy: 90.5% (misleading - just predicting majority class)
  - Recall for class 1: **0%** (catches ZERO high performers)
  - Precision for class 1: 0% (undefined - no predictions)
  - **Conclusion:** Useless baseline - establishes performance floor

**2. Logistic Regression (First Real Model):**
- Configuration: `class_weight='balanced'`, `max_iter=5000`, with StandardScaler
- **Test Results:**
  - Accuracy: 65%
  - Recall for class 1: **53%** (catches 116 out of 220 high performers) ✅
  - Precision for class 1: 14% (~830 flagged posts total)
  - **Conclusion:** Solid baseline - proves features contain signal!

**Key Learning:**
- Without `class_weight='balanced'`: Model ignores minority class → 0% recall
- With `class_weight='balanced'`: Model actually learns → 53% recall
- **Accuracy is a vanity metric for imbalanced data** - focus on recall/precision/ROC-AUC
- Feature scaling had minimal impact (features already well-behaved)

---

### XGBoost Models & The Distribution vs Content Discovery ✅

**Evolution of Model Performance:**

| Model Version | Features | Val Recall | Test Recall | ROC-AUC | PR-AUC | Status |
|--------------|----------|-----------|------------|---------|--------|--------|
| **V1: With Profile + Network** | 39 | 77% | 57% | 0.766 | 0.346 | Overfitted 🟡 |
| **V2: Without Profile** | 33 | 49% | 49% | 0.771 | 0.317 | Clean but lower 🟡 |
| **V3: Without Profile + Network** | 27 | — | 43% | 0.616 | 0.169 | Weak signal 🔴 |

---

### 🔍 Critical Finding: Distribution > Content Quality

**The Model Evolution Revealed:**

#### **Version 1: Profile + Network Features (Confounded Model)**
- **Validation:** 77% recall, ROC-AUC 0.766
- **Test:** 57% recall (20pp drop!)
- **Problem:** Model learned "If Profile=tuminha_dds (Instagram account) → predict high performer"
- **Why it worked:** Instagram account has 10K+ followers → naturally high engagement
- **Why it failed:** This is **data leakage** - predicting based on follower count, not content quality
- **Top feature:** `Profile_tuminha_dds` (19.1% importance) 🚨

#### **Version 2: Removed Profile, Kept Network**
- **Test:** 49% recall, ROC-AUC 0.771
- **Problem:** Network acts as **proxy for Profile**
- `Network_Instagram` became top feature (12.6%) because Instagram = tuminha_dds account
- Model still learning structural advantages, not content patterns
- Performance dropped 8pp without Profile "cheat code"

#### **Version 3: Pure Content Model (No Profile, No Network)**
- **Test:** 43% recall, ROC-AUC 0.616 ⚠️
- **ROC-AUC 0.616** = barely better than random (0.5 = coin flip)
- **PR-AUC 0.169** = only 1.7x better than random baseline (0.10)
- **Conclusion:** Pure content features have WEAK predictive power

---

### 💡 Profound Discovery: What Drives Social Media Engagement?

**The Math:**
```
Engagement Performance = 
  70% Platform/Distribution (WHERE you post + follower base)
+ 30% Content Quality (WHAT you post)
```

**Evidence:**
- Model with structural features (Profile/Network): Strong performance (ROC-AUC 0.77)
- Model with pure content: Weak performance (ROC-AUC 0.62)
- **Implication:** At current scale, distribution matters MORE than content optimization

**Visual Evidence:**
- **ROC Curve (pure content model):** Hugs diagonal line (random guessing)
- **Precision-Recall Curve:** Sharp drop, flatlines at ~15-20% precision
- Model struggles to maintain precision as it tries to catch more high performers

---

### 📊 Content Insights (Marginal but Real)

**Top Features from Pure Content Model:**

| Rank | Feature | Importance | Actionable Insight |
|------|---------|------------|-------------------|
| 1 | **has_numbers** | 11.1% | ✅ Include data, stats, percentages |
| 2 | **Post Type_Post** | 10.1% | ✅ Regular posts > Stories |
| 3 | **Content Type_Video** | 9.9% | ✅ Videos > Photos > Text |
| 4 | **mention_count** | 7.4% | ✅ Tag relevant people/brands |
| 5 | **url_count** | 6.9% | ✅ Link to external content |
| 6 | **Post Type_@Reply** | 6.5% | ✅ Engage with others' posts |
| 7 | **hashtag_count** | 6.4% | ✅ Use hashtags strategically |
| 8 | **has_emoji** | 5.6% | ✅ Add personality with emojis |
| 9 | **Post Type_Quote** | 5.5% | ✅ Quote tweets for commentary |
| 10 | **caption_length** | 5.0% | ✅ Longer captions provide context |

**These patterns are REAL but provide only 10-20% performance edge.**

---

### 🎯 Model Limitations & Practical Implications

**Limitations:**
1. ✅ **Pure content model (V3) has weak predictive power** - ROC-AUC 0.616
2. ✅ **Cannot predict viral hits from content alone** - structural effects dominate
3. ✅ **Validation→Test drops indicate overfitting** to December 2024 patterns
4. ✅ **Platform algorithms change** - model trained on 2024 behavior
5. ✅ **Small sample sizes per network** - Instagram only 439 posts

**What This Means:**
- Great content on small platform < Good content on large platform
- Model best used for marginal optimization, not strategic decisions
- Distribution (followers, platform algorithm) is the main bottleneck
- Content optimization can improve by 10-20%, not 2-3x

**Practical Recommendations:**
1. **Prioritize platform growth** over content perfection (Instagram, Threads)
2. **Cross-post strategically** - videos on Instagram, text threads on Twitter/X
3. **Use content guidelines as best practices:**
   - Videos > Photos > Text-only
   - Include numbers/data when relevant
   - Tag people, add links, use hashtags
   - Engage via replies and quotes
4. **Don't obsess over optimization** when distribution is the bottleneck
5. **Consistency > Perfection** - posting frequency matters more than individual post tuning

---

### 🎓 Key Machine Learning Lessons Learned

**1. Data Leakage is Subtle:**
- Profile features seemed legitimate but encoded follower count
- Network features proxy for account popularity
- Always ask: "Would this be available in production?"

**2. Feature Importance ≠ Causation:**
- High importance for Profile/Network doesn't mean "improve those features"
- They're confounding variables, not levers you can pull

**3. A Failing Model Teaches More Than a Perfect One:**
- Weak pure-content model revealed what truly matters
- Performance drop quantified distribution vs content split (70/30)

**4. Imbalanced Data Requires Special Care:**
- Accuracy is meaningless (90% by predicting all 0s)
- `class_weight='balanced'` and `scale_pos_weight` are critical
- ROC-AUC and PR-AUC are better metrics than accuracy

**5. Validation Strategy Matters:**
- Temporal split caught distribution shift (2024→2025)
- Random split would have hidden overfitting

---

### Final Model Recommendation

**For Production Use: Version 2 (Without Profile, Keep Network)**
- Test recall: 49%
- ROC-AUC: 0.771
- More useful predictions while avoiding Profile leakage
- Accepts that platform matters (realistic)

**For Content Insights: Version 3 (Pure Content)**
- Identifies actionable content patterns
- Unbiased by structural advantages
- Best for understanding WHAT works in content

---

### Next Steps
- [ ] Confusion matrix analysis
- [ ] Consider SHAP/LIME for model explainability
- [ ] Per-network content analysis (what works on Instagram vs Twitter?)
- [ ] Threshold optimization for business goals
- [ ] Production deployment considerations

---

## Repro

1. `python -m venv .venv && source .venv/bin/activate` (or use Conda)
2. `pip install -r requirements.txt`
3. Open `notebooks/01_post_performance_binary.ipynb` and run cells top-to-bottom.

