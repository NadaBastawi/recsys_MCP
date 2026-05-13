"""
TORCO Pest Control — Hybrid Recommendation Engine
Combines:
  1. Content-Based Filtering  — recommends based on customer profile features
  2. Collaborative Filtering  — recommends based on similar customers' behavior
  3. Urgency Scorer           — predicts how urgently a customer needs a service
Final output: Top-N service recommendations with urgency scores
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import warnings
warnings.filterwarnings("ignore")

# Constants
CB_WEIGHT_DEFAULT = 0.4
CF_WEIGHT_DEFAULT = 0.6
SVD_COMPONENTS_DEFAULT = 8
RANDOM_SEED = 42
DESERT_ADJACENT_ZIPS = ["85743", "85745", "85741", "85742", "85718", "85704"]
OLDER_NEIGHBORHOOD_ZIPS = ["85701", "85705", "85706", "85713", "85716", "85719"]

# ── 1. CONTENT-BASED FILTERING ────────────────────────────────────────────────

class ContentBasedRecommender:
    """
    Recommends services based on customer profile features.
    Uses cosine similarity between customer feature vectors and
    service affinity vectors derived from historical data.
    """

    def __init__(self):
        self.scaler = MinMaxScaler()
        self.customer_features = None
        self.service_affinity = None
        self.feature_cols = None

    def _build_customer_features(self, customers_df):
        le_prop = LabelEncoder()
        df = customers_df.copy()
        df["property_type_enc"] = le_prop.fit_transform(df["property_type"])
        feature_cols = [
            "property_type_enc", "building_age_years",
            "is_desert_adjacent", "is_older_neighborhood", "is_commercial"
        ]
        self.feature_cols = feature_cols
        features = df[feature_cols].astype(float).values
        return self.scaler.fit_transform(features), df["customer_id"].values

    def _build_service_affinity(self, interactions_df, customers_df, services_df):
        """
        For each service, compute the average customer feature vector
        of customers who rated it highly (rating >= 4).
        This is the service's 'ideal customer profile'.
        """
        merged = interactions_df.merge(customers_df, on="customer_id")
        high_rated = merged[merged["rating"] >= 4]

        le_prop = LabelEncoder()
        le_prop.fit(customers_df["property_type"])
        high_rated = high_rated.copy()
        high_rated["property_type_enc"] = le_prop.transform(high_rated["property_type"])

        affinity = {}
        for service_id in services_df["service_id"]:
            subset = high_rated[high_rated["service_id"] == service_id]
            if len(subset) > 0:
                vec = subset[self.feature_cols].astype(float).mean().values
                affinity[service_id] = self.scaler.transform([vec])[0]
            else:
                affinity[service_id] = np.zeros(len(self.feature_cols))

        return affinity

    def fit(self, customers_df, interactions_df, services_df):
        print("  Fitting content-based recommender...")
        self.customer_matrix, self.customer_ids = self._build_customer_features(customers_df)
        self.service_affinity = self._build_service_affinity(interactions_df, customers_df, services_df)
        print(f"  ✓ Built feature vectors for {len(self.customer_ids)} customers, {len(self.service_affinity)} services")
        return self

    def recommend(self, customer_id, top_n=5):
        idx = np.where(self.customer_ids == customer_id)[0]
        if len(idx) == 0:
            return []
        customer_vec = self.customer_matrix[idx[0]].reshape(1, -1)
        scores = {}
        for service_id, affinity_vec in self.service_affinity.items():
            sim = cosine_similarity(customer_vec, affinity_vec.reshape(1, -1))[0][0]
            scores[service_id] = sim
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]


# ── 2. COLLABORATIVE FILTERING ────────────────────────────────────────────────

class CollaborativeFilteringRecommender:
    """
    Matrix factorization using SVD.
    Builds a customer x service rating matrix and decomposes it
    to find latent factors — similar to how Spotify finds users
    with similar listening patterns.
    """

    def __init__(self, n_components=SVD_COMPONENTS_DEFAULT):
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
        self.customer_index = {}
        self.service_index = {}
        self.reconstructed = None

    def fit(self, interactions_df):
        print("  Fitting collaborative filtering (SVD)...")
        # Build rating matrix
        customers_list = interactions_df["customer_id"].unique().tolist()
        services_list = interactions_df["service_id"].unique().tolist()
        self.customer_index = {c: i for i, c in enumerate(customers_list)}
        self.service_index = {s: i for i, s in enumerate(services_list)}
        self.service_ids = services_list

        matrix = np.zeros((len(customers_list), len(services_list)))
        for _, row in interactions_df.iterrows():
            ci = self.customer_index.get(row["customer_id"])
            si = self.service_index.get(row["service_id"])
            if ci is not None and si is not None:
                # Use max rating if multiple interactions
                matrix[ci][si] = max(matrix[ci][si], row["rating"])

        # SVD decomposition
        self.customer_factors = self.svd.fit_transform(matrix)
        self.service_factors = self.svd.components_.T
        self.reconstructed = np.dot(self.customer_factors, self.svd.components_)

        explained = self.svd.explained_variance_ratio_.sum()
        print(f"  ✓ SVD: {self.n_components} components, {explained:.1%} variance explained")
        return self

    def recommend(self, customer_id, top_n=5):
        if customer_id not in self.customer_index:
            return []
        ci = self.customer_index[customer_id]
        scores = self.reconstructed[ci]
        # Get top services by predicted rating
        ranked_indices = np.argsort(scores)[::-1][:top_n]
        return [(self.service_ids[i], float(scores[i])) for i in ranked_indices]


# ── 3. URGENCY SCORER ─────────────────────────────────────────────────────────

class UrgencyScorer:
    """
    Predicts urgency score (1-10) for a customer-service pair.
    Uses gradient boosting on contextual features:
    - Season, building age, days since last service, property type, etc.
    """

    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=RANDOM_SEED)
        self.le_season = LabelEncoder()
        self.le_prop = LabelEncoder()

    def _prepare_features(self, df):
        d = df.copy()
        d["season_enc"] = self.le_season.transform(d["season"])
        d["prop_enc"] = self.le_prop.transform(d["property_type"])
        d["service_num"] = d["service_id"].str.replace("S", "").astype(int)
        return d[["season_enc", "prop_enc", "building_age_years",
                   "days_since_last_service", "service_num", "is_commercial",
                   "is_desert_adjacent", "is_older_neighborhood"]].astype(float).values

    def fit(self, interactions_df, customers_df):
        print("  Fitting urgency scorer...")
        merged = interactions_df.merge(customers_df, on="customer_id")
        self.le_season.fit(merged["season"])
        self.le_prop.fit(merged["property_type"])
        X = self._prepare_features(merged)
        y = merged["urgency_score"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"  ✓ Urgency scorer RMSE: {rmse:.3f}")
        return self

    def predict_urgency(self, customer_profile, service_id, season, days_since):
        row = {
            "season": season,
            "property_type": customer_profile["property_type"],
            "building_age_years": customer_profile["building_age_years"],
            "days_since_last_service": days_since,
            "service_id": service_id,
            "is_commercial": int(customer_profile["is_commercial"]),
            "is_desert_adjacent": int(customer_profile["is_desert_adjacent"]),
            "is_older_neighborhood": int(customer_profile["is_older_neighborhood"]),
        }
        df = pd.DataFrame([row])
        df["season_enc"] = self.le_season.transform(df["season"])
        df["prop_enc"] = self.le_prop.transform(df["property_type"])
        df["service_num"] = df["service_id"].str.replace("S", "").astype(int)
        features = df[["season_enc", "prop_enc", "building_age_years",
                        "days_since_last_service", "service_num", "is_commercial",
                        "is_desert_adjacent", "is_older_neighborhood"]].astype(float).values
        score = self.model.predict(features)[0]
        return round(float(np.clip(score, 1, 10)), 1)


# ── 4. HYBRID RECOMMENDER ─────────────────────────────────────────────────────

class HybridRecommender:
    """
    Combines content-based and collaborative filtering with weighted scoring.
    Content weight: 0.4 | Collaborative weight: 0.6
    Final output includes urgency score and service details.
    """

    def __init__(self, cb_weight=CB_WEIGHT_DEFAULT, cf_weight=CF_WEIGHT_DEFAULT):
        self.cb_weight = cb_weight
        self.cf_weight = cf_weight
        self.cb = ContentBasedRecommender()
        self.cf = CollaborativeFilteringRecommender()
        self.urgency = UrgencyScorer()
        self.services_dict = {}

    def fit(self, customers_df, interactions_df, services_df):
        print("\nTraining Hybrid Recommender...")
        self.cb.fit(customers_df, interactions_df, services_df)
        self.cf.fit(interactions_df)
        self.urgency.fit(interactions_df, customers_df)
        self.services_dict = services_df.set_index("service_id").to_dict("index")
        self.customers_df = customers_df.set_index("customer_id")
        print("\n✓ Hybrid Recommender ready\n")
        return self

    def recommend(self, customer_id, top_n=5, season="summer", days_since_last=180):
        # Get scores from both models
        cb_scores = dict(self.cb.recommend(customer_id, top_n=10))
        cf_scores = dict(self.cf.recommend(customer_id, top_n=10))

        # Normalize scores to 0-1
        cb_norm = self._normalize_scores(cb_scores)
        cf_norm = self._normalize_scores(cf_scores)

        # Combine
        all_services = set(cb_norm) | set(cf_norm)
        hybrid_scores = {}
        for sid in all_services:
            cb_s = cb_norm.get(sid, 0)
            cf_s = cf_norm.get(sid, 0)
            hybrid_scores[sid] = self.cb_weight * cb_s + self.cf_weight * cf_s

        ranked = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        # Build final recommendations with urgency
        customer_profile = self.customers_df.loc[customer_id].to_dict() if customer_id in self.customers_df.index else {}
        return self._build_recommendations(ranked, customer_profile, season, days_since_last)

    def _normalize_scores(self, scores_dict):
        """Normalize scores to 0-1 range."""
        if not scores_dict:
            return scores_dict
        min_v, max_v = min(scores_dict.values()), max(scores_dict.values())
        if max_v == min_v:
            return {k: 1.0 for k in scores_dict}
        return {k: (v - min_v) / (max_v - min_v) for k, v in scores_dict.items()}

    def _build_recommendations(self, ranked_scores, customer_profile, season, days_since_last):
        """Build recommendation results with urgency scores."""
        results = []
        for service_id, score in ranked_scores:
            service_info = self.services_dict.get(service_id, {})
            urgency = self.urgency.predict_urgency(
                customer_profile, service_id, season, days_since_last
            ) if customer_profile else 5.0

            results.append({
                "service_id": service_id,
                "service_name": service_info.get("service_name", service_id),
                "category": service_info.get("category", ""),
                "price_usd": service_info.get("price_usd", 0),
                "score": round(score, 4),
                "urgency_score": urgency,
                "urgency_label": "HIGH" if urgency >= 7 else ("MEDIUM" if urgency >= 4 else "LOW"),
            })
        return results

    def recommend_cold_start(self, property_type, zip_code, building_age_years,
                             season="summer", days_since_last=180, top_n=5):
        """
        Recommend services for a new customer without historical data.

        Args:
            property_type: The customer's property type
            zip_code: The customer's ZIP code
            building_age_years: Age of the building
            season: Current season
            days_since_last: Days since last service (default 180)
            top_n: Number of top recommendations

        Returns:
            List of service recommendations with urgency scoring
        """
        profile = self._build_customer_profile(property_type, zip_code, building_age_years)

        results = []
        for service_id, service_info in self.services_dict.items():
            urgency = self.urgency.predict_urgency(
                profile, service_id, season, days_since_last
            )
            results.append({
                "service_id": service_id,
                "service_name": service_info.get("service_name", service_id),
                "category": service_info.get("category", ""),
                "price_usd": service_info.get("price_usd", 0),
                "score": round(urgency / 10, 4),
                "urgency_score": urgency,
                "urgency_label": "HIGH" if urgency >= 7 else ("MEDIUM" if urgency >= 4 else "LOW"),
            })

        results.sort(key=lambda x: x["urgency_score"], reverse=True)
        return results[:top_n]

    def _build_customer_profile(self, property_type, zip_code, building_age_years):
        """Build a customer profile dictionary for new customers."""
        return {
            "property_type": property_type,
            "zip_code": zip_code,
            "building_age_years": building_age_years,
            "is_desert_adjacent": zip_code in DESERT_ADJACENT_ZIPS,
            "is_older_neighborhood": zip_code in OLDER_NEIGHBORHOOD_ZIPS,
            "is_commercial": "commercial" in property_type,
        }

    def save(self, path="torco_recommender.pkl"):
        joblib.dump(self, path)
        print(f"Model saved to {path}")

    @staticmethod
    def load(path="torco_recommender.pkl"):
        return joblib.load(path)


# ── EVALUATION ────────────────────────────────────────────────────────────────

def evaluate_recommender(model, interactions_df, k=5):
    """
    Evaluate using Precision@K and a simulated NDCG@K.
    Holdout: last interaction per customer as ground truth.
    """
    print("\nEvaluating recommender...")
    interactions_sorted = interactions_df.sort_values("date")
    holdout = interactions_sorted.groupby("customer_id").last().reset_index()
    train = interactions_sorted[~interactions_sorted.index.isin(holdout.index)]

    precisions = []
    for _, row in holdout.iterrows():
        customer_id = row["customer_id"]
        true_service = row["service_id"]
        recs = model.recommend(customer_id, top_n=k)
        rec_ids = [r["service_id"] for r in recs]
        hit = 1 if true_service in rec_ids else 0
        precisions.append(hit)

    precision_at_k = np.mean(precisions)
    print(f"  Precision@{k}: {precision_at_k:.4f}")
    print(f"  Coverage: {len(set(r['service_id'] for c in holdout['customer_id'] for r in model.recommend(c, top_n=k)))}/{len(interactions_df['service_id'].unique())} services recommended")
    return precision_at_k


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load data
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
    interactions = pd.read_csv(os.path.join(DATA_DIR, "interactions.csv"))
    services = pd.read_csv(os.path.join(DATA_DIR, "services.csv"))

    print(f"Loaded: {len(customers)} customers, {len(interactions)} interactions, {len(services)} services")

    # Train
    recommender = HybridRecommender(cb_weight=CB_WEIGHT_DEFAULT, cf_weight=CF_WEIGHT_DEFAULT)
    recommender.fit(customers, interactions, services)

    # Evaluate
    precision = evaluate_recommender(recommender, interactions, k=5)

    # Sample recommendation
    sample_customer = customers.iloc[0]["customer_id"]
    print(f"\nSample recommendations for customer {sample_customer}:")
    print(f"Profile: {customers.iloc[0]['property_type']} | ZIP: {customers.iloc[0]['zip_code']} | Age: {customers.iloc[0]['building_age_years']}yrs")
    print()
    recs = recommender.recommend(sample_customer, top_n=5, season="summer", days_since_last=200)
    for i, rec in enumerate(recs, 1):
        print(f"  {i}. {rec['service_name']}")
        print(f"     Score: {rec['score']} | Urgency: {rec['urgency_score']}/10 [{rec['urgency_label']}] | ${rec['price_usd']}")

    # Save model
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)
    recommender.save(os.path.join(MODEL_DIR, "torco_recommender.pkl"))
