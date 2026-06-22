"""
Dark Kitchen Delivery Zone Optimizer
=====================================
DS Layer  : DBSCAN clustering of order demand hotspots
OR Layer  : Facility Location ILP — minimise weighted delivery cost
            while choosing K dark kitchen sites from candidate list
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pulp
import folium
import json
from faker import Faker
import random

random.seed(42)
np.random.seed(42)
fake = Faker("en_IN")

# ─────────────────────────────────────────────
# 1. SYNTHETIC ORDER DATA  (replace with real Zomato/Swiggy export)
# ─────────────────────────────────────────────
CITY_CENTER = (12.9716, 77.5946)   # Bengaluru

def generate_orders(n=800):
    """Simulate geo-tagged food orders around a city."""
    lats, lons, demands = [], [], []
    # Three natural demand clusters (office areas, residential zones)
    cluster_centers = [
        (12.9352, 77.6245),   # Koramangala
        (13.0100, 77.5800),   # Yeshwanthpur
        (12.9010, 77.6352),   # BTM Layout
    ]
    for _ in range(n):
        cx, cy = random.choice(cluster_centers)
        lats.append(cx + np.random.normal(0, 0.02))
        lons.append(cy + np.random.normal(0, 0.02))
        demands.append(random.randint(1, 10))
    return pd.DataFrame({"lat": lats, "lon": lons, "demand": demands})

orders = generate_orders(800)
print(f"[DATA] Generated {len(orders)} orders")

# ─────────────────────────────────────────────
# 2. DS LAYER — DBSCAN DEMAND CLUSTERING
# ─────────────────────────────────────────────
coords = orders[["lat", "lon"]].values
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)

db = DBSCAN(eps=0.15, min_samples=8).fit(coords_scaled)
orders["cluster"] = db.labels_

n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
print(f"[DBSCAN] Found {n_clusters} demand clusters  |  noise points: {(db.labels_==-1).sum()}")

# Cluster centroids = demand hotspots (customer zones)
cluster_info = (
    orders[orders.cluster >= 0]
    .groupby("cluster")
    .agg(lat=("lat", "mean"), lon=("lon", "mean"), demand=("demand", "sum"))
    .reset_index()
)
print(cluster_info)

# ─────────────────────────────────────────────
# 3. CANDIDATE KITCHEN SITES  (sampled from order area + some extras)
# ─────────────────────────────────────────────
def candidate_sites(n=12):
    sites = []
    for i in range(n):
        lat = CITY_CENTER[0] + np.random.uniform(-0.08, 0.08)
        lon = CITY_CENTER[1] + np.random.uniform(-0.08, 0.08)
        fixed_cost = random.randint(50000, 150000)   # INR/month
        sites.append({"site_id": i, "lat": lat, "lon": lon, "fixed_cost": fixed_cost})
    return pd.DataFrame(sites)

sites = candidate_sites(12)

# ─────────────────────────────────────────────
# 4. OR LAYER — FACILITY LOCATION ILP
#    Minimise: Σ fixed_cost * y_j  +  Σ dist(i,j) * demand_i * x_ij
#    Subject to:
#      Σ_j x_ij = 1   (each demand zone served by exactly one kitchen)
#      x_ij ≤ y_j     (can only assign if kitchen is open)
#      Σ_j y_j = K    (open exactly K kitchens)
# ─────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1-a))

I = list(cluster_info.index)          # demand zones
J = list(sites.index)                  # candidate kitchen sites
K = 3                                  # number of kitchens to open
COST_PER_KM = 15                       # INR per km per demand unit

dist = {
    (i, j): haversine(cluster_info.loc[i,"lat"], cluster_info.loc[i,"lon"],
                      sites.loc[j,"lat"],         sites.loc[j,"lon"])
    for i in I for j in J
}
demand = cluster_info["demand"].to_dict()
fixed  = sites["fixed_cost"].to_dict()

prob = pulp.LpProblem("DarkKitchenPlacement", pulp.LpMinimize)
x = pulp.LpVariable.dicts("assign", [(i,j) for i in I for j in J], cat="Binary")
y = pulp.LpVariable.dicts("open",   [j for j in J],                 cat="Binary")

# Objective
prob += (
    pulp.lpSum(fixed[j] * y[j] for j in J) +
    pulp.lpSum(dist[i,j] * demand[i] * COST_PER_KM * x[i,j] for i in I for j in J)
)

# Constraints
for i in I:
    prob += pulp.lpSum(x[i,j] for j in J) == 1            # each zone assigned once
for i in I:
    for j in J:
        prob += x[i,j] <= y[j]                             # open before assign
prob += pulp.lpSum(y[j] for j in J) == K                   # exactly K kitchens

prob.solve(pulp.PULP_CBC_CMD(msg=0))
print(f"\n[OR] Status : {pulp.LpStatus[prob.status]}")
print(f"[OR] Total cost : ₹{pulp.value(prob.objective):,.0f}")

open_kitchens = [j for j in J if pulp.value(y[j]) == 1]
assignments   = {i: next(j for j in J if pulp.value(x[i,j]) == 1) for i in I}
print(f"[OR] Open kitchen sites : {open_kitchens}")

# ─────────────────────────────────────────────
# 5. FOLIUM MAP VISUALISATION
# ─────────────────────────────────────────────
colors = ["red","blue","green","purple","orange","darkred","lightblue"]

m = folium.Map(location=CITY_CENTER, zoom_start=12, tiles="CartoDB positron")

# Plot order heatmap
for _, row in orders.iterrows():
    folium.CircleMarker(
        location=[row.lat, row.lon],
        radius=2, color="gray", fill=True, fill_opacity=0.3
    ).add_to(m)

# Plot demand clusters
for _, row in cluster_info.iterrows():
    folium.CircleMarker(
        location=[row.lat, row.lon],
        radius=10, color="orange", fill=True,
        tooltip=f"Cluster {int(row.cluster)} | Demand: {int(row.demand)}"
    ).add_to(m)

# Plot candidate sites (grey)
for _, row in sites.iterrows():
    folium.CircleMarker(
        location=[row.lat, row.lon],
        radius=6, color="lightgray", fill=True, fill_opacity=0.5,
        tooltip=f"Candidate site {int(row.site_id)}"
    ).add_to(m)

# Highlight open kitchens
for j in open_kitchens:
    row = sites.loc[j]
    folium.Marker(
        location=[row.lat, row.lon],
        icon=folium.Icon(color="green", icon="cutlery", prefix="fa"),
        tooltip=f"✅ OPEN Kitchen {j} | Fixed cost ₹{fixed[j]:,}"
    ).add_to(m)

# Draw assignment lines
for i, j in assignments.items():
    folium.PolyLine(
        locations=[
            [cluster_info.loc[i,"lat"], cluster_info.loc[i,"lon"]],
            [sites.loc[j,"lat"], sites.loc[j,"lon"]]
        ],
        color=colors[j % len(colors)], weight=2, opacity=0.7
    ).add_to(m)

m.save("dark_kitchen_map.html")
print("\n[MAP] Saved → dark_kitchen_map.html")

# ─────────────────────────────────────────────
# 6. SUMMARY TABLE
# ─────────────────────────────────────────────
summary = []
for i in I:
    j = assignments[i]
    summary.append({
        "Demand Zone": f"Cluster {i}",
        "Demand Units": demand[i],
        "Assigned Kitchen": f"Site {j}",
        "Distance (km)": round(dist[i,j], 2),
        "Delivery Cost (₹)": round(dist[i,j]*demand[i]*COST_PER_KM, 0)
    })
df_summary = pd.DataFrame(summary)
print("\n", df_summary.to_string(index=False))
df_summary.to_csv("assignment_summary.csv", index=False)
print("[DONE] Results saved to assignment_summary.csv")
