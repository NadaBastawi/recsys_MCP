"""
TORCO Pest Control — Synthetic Data Generator
Generates realistic customer, property, service, and interaction data
based on real-world pest control patterns in Tucson, AZ / Southwest US
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from faker import Faker
import random
import json
from datetime import datetime, timedelta

fake = Faker()
np.random.seed(42)
random.seed(42)

# ── DOMAIN CONSTANTS ──────────────────────────────────────────────────────────

SERVICES = {
    "S01": {"name": "General Pest Control",        "price": 120, "category": "general"},
    "S02": {"name": "Termite Inspection",           "price": 150, "category": "termite"},
    "S03": {"name": "Termite Treatment",            "price": 800, "category": "termite"},
    "S04": {"name": "Bed Bug Treatment",            "price": 400, "category": "bed_bug"},
    "S05": {"name": "Rodent Control",               "price": 200, "category": "rodent"},
    "S06": {"name": "Scorpion Control",             "price": 180, "category": "scorpion"},
    "S07": {"name": "Ant & Cockroach Treatment",    "price": 130, "category": "ant_roach"},
    "S08": {"name": "Commercial Pest Management",   "price": 500, "category": "commercial"},
    "S09": {"name": "Preventive Maintenance Plan",  "price": 300, "category": "preventive"},
    "S10": {"name": "Emergency Pest Response",      "price": 250, "category": "emergency"},
}

PROPERTY_TYPES = ["residential_house", "residential_apartment", "commercial_restaurant",
                  "commercial_office", "commercial_warehouse", "mobile_home"]

TUCSON_ZIP_CODES = ["85701", "85704", "85705", "85706", "85710",
                    "85711", "85712", "85713", "85714", "85715",
                    "85716", "85718", "85719", "85730", "85741",
                    "85742", "85743", "85745", "85746", "85747"]

# Desert-adjacent zips (higher scorpion/ant risk)
DESERT_ADJACENT = ["85743", "85745", "85741", "85742", "85718", "85704"]

# Older neighborhoods (higher termite risk)
OLDER_NEIGHBORHOODS = ["85701", "85705", "85706", "85713", "85716", "85719"]

SEASONS = {1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
           6: "summer", 7: "summer", 8: "summer", 9: "fall", 10: "fall",
           11: "fall", 12: "winter"}

# ── RISK PROFILES ─────────────────────────────────────────────────────────────

def get_service_probabilities(property_type, zip_code, season, building_age, prev_services):
    """
    Compute realistic service probabilities based on domain knowledge:
    - Termites peak in spring/summer in Tucson
    - Scorpions are desert-adjacent problem
    - Commercial properties have higher rodent risk
    - Older buildings have higher termite risk
    - Bed bugs are apartment/hotel problem
    """
    probs = {sid: 0.05 for sid in SERVICES}  # base probability

    is_desert = zip_code in DESERT_ADJACENT
    is_old = zip_code in OLDER_NEIGHBORHOODS or building_age > 20
    is_commercial = "commercial" in property_type
    is_apartment = property_type == "residential_apartment"
    is_restaurant = property_type == "commercial_restaurant"
    is_warehouse = property_type == "commercial_warehouse"

    # General pest control — universal baseline
    probs["S01"] = 0.35 if not is_commercial else 0.15

    # Termite — peaks spring/summer, higher in older/desert areas
    termite_base = 0.10
    if season in ["spring", "summer"]: termite_base += 0.15
    if is_old: termite_base += 0.15
    if is_desert: termite_base += 0.05
    probs["S02"] = termite_base
    probs["S03"] = termite_base * 0.6  # treatment less common than inspection

    # Bed bugs — apartments and commercial lodging
    probs["S04"] = 0.20 if is_apartment else 0.05

    # Rodents — commercial, warehouses, restaurants
    probs["S05"] = 0.30 if is_restaurant else (0.25 if is_warehouse else 0.08)

    # Scorpions — desert adjacent, residential
    probs["S06"] = 0.30 if (is_desert and not is_commercial) else 0.05

    # Ants/cockroaches — restaurants and apartments
    probs["S07"] = 0.30 if is_restaurant else (0.20 if is_apartment else 0.10)

    # Commercial management — commercial only
    probs["S08"] = 0.40 if is_commercial else 0.02

    # Preventive plan — higher if had issues before
    probs["S09"] = 0.25 if len(prev_services) > 2 else 0.10

    # Emergency — rare but spikes in summer
    probs["S10"] = 0.10 if season == "summer" else 0.03

    # Boost probability if customer had related service before (repeat patterns)
    if "S02" in prev_services: probs["S03"] += 0.20  # inspection → treatment
    if "S03" in prev_services: probs["S09"] += 0.15  # treatment → preventive plan
    if "S01" in prev_services: probs["S09"] += 0.10  # general → preventive

    # Normalize
    total = sum(probs.values())
    return {k: v / total for k, v in probs.items()}


def get_urgency(service_id, season, building_age, days_since_last_service):
    """Predict urgency score 1-10 based on context"""
    base = 5
    if service_id in ["S03", "S04", "S10"]: base = 7  # high risk services
    if season == "summer" and service_id in ["S02", "S03", "S06"]: base += 2
    if building_age > 30 and service_id in ["S02", "S03"]: base += 1
    if days_since_last_service > 365: base += 1
    if days_since_last_service > 730: base += 2
    return min(10, max(1, base + random.randint(-1, 1)))


# ── GENERATORS ────────────────────────────────────────────────────────────────

def generate_customers(n=500):
    customers = []
    for i in range(n):
        zip_code = random.choice(TUCSON_ZIP_CODES)
        prop_type = random.choices(
            PROPERTY_TYPES,
            weights=[0.40, 0.20, 0.12, 0.12, 0.08, 0.08]
        )[0]
        building_age = random.randint(1, 50)
        customers.append({
            "customer_id": f"C{str(i+1).zfill(4)}",
            "name": fake.name(),
            "email": fake.email(),
            "zip_code": zip_code,
            "property_type": prop_type,
            "building_age_years": building_age,
            "is_desert_adjacent": zip_code in DESERT_ADJACENT,
            "is_older_neighborhood": zip_code in OLDER_NEIGHBORHOODS,
            "is_commercial": "commercial" in prop_type,
            "member_since": fake.date_between(start_date="-5y", end_date="-6m").isoformat(),
            "num_units": random.randint(1, 50) if prop_type in ["residential_apartment", "commercial_warehouse"] else 1,
        })
    return pd.DataFrame(customers)


def generate_interactions(customers_df, n_interactions=3000):
    interactions = []
    for _ in range(n_interactions):
        customer = customers_df.sample(1).iloc[0]
        date = fake.date_between(start_date="-3y", end_date="today")
        season = SEASONS[date.month]

        # Get previous services for this customer
        prev = [r["service_id"] for r in interactions if r["customer_id"] == customer["customer_id"]]

        probs = get_service_probabilities(
            customer["property_type"],
            customer["zip_code"],
            season,
            customer["building_age_years"],
            prev
        )

        service_id = random.choices(list(probs.keys()), weights=list(probs.values()))[0]
        days_since = random.randint(30, 730)
        urgency = get_urgency(service_id, season, customer["building_age_years"], days_since)
        rating = max(1, min(5, round(random.gauss(4.1, 0.8))))

        interactions.append({
            "interaction_id": f"I{str(len(interactions)+1).zfill(5)}",
            "customer_id": customer["customer_id"],
            "service_id": service_id,
            "service_name": SERVICES[service_id]["name"],
            "date": date.isoformat(),
            "season": season,
            "rating": rating,
            "urgency_score": urgency,
            "days_since_last_service": days_since,
            "completed": random.choices([True, False], weights=[0.92, 0.08])[0],
            "repeat_booking": service_id in prev,
        })

    return pd.DataFrame(interactions)


def generate_services_df():
    rows = []
    for sid, info in SERVICES.items():
        rows.append({
            "service_id": sid,
            "service_name": info["name"],
            "price_usd": info["price"],
            "category": info["category"],
            "suitable_residential": info["category"] not in ["commercial"],
            "suitable_commercial": info["category"] in ["commercial", "rodent", "ant_roach", "general", "preventive", "emergency"],
            "peak_season": "spring-summer" if info["category"] in ["termite", "scorpion"] else "year-round",
        })
    return pd.DataFrame(rows)


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent / "data"
    output_dir.mkdir(exist_ok=True)

    print("Generating customers...")
    customers = generate_customers(500)
    customers.to_csv(output_dir / "customers.csv", index=False)
    print(f"  ✓ {len(customers)} customers")

    print("Generating interactions...")
    interactions = generate_interactions(customers, 3000)
    interactions.to_csv(output_dir / "interactions.csv", index=False)
    print(f"  ✓ {len(interactions)} interactions")

    print("Generating services catalog...")
    services = generate_services_df()
    services.to_csv(output_dir / "services.csv", index=False)
    print(f"  ✓ {len(services)} services")

    print("\nSample customer:")
    print(customers.head(2).to_string())
    print("\nSample interactions:")
    print(interactions.head(3).to_string())
    print(f"\nDone. Files saved in {output_dir}")
