"""
Streamlit Dashboard — Dark Kitchen Optimizer
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import pulp
import folium
from streamlit_folium import st_folium
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import random

st.set_page_config(page_title="Dark Kitchen Optimizer", layout="wide")
st.title("🍽️ Dark Kitchen Delivery Zone Optimizer")
st.markdown("**DS Layer**: DBSCAN demand clustering &nbsp;|&nbsp; **OR Layer**: Facility Location ILP")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Parameters")
    n_orders   = st.slider("Number of orders", 200, 1000, 600, 100)
    K          = st.slider("Kitchens to open (K)", 1, 6, 3)
    n_sites    = st.slider("Candidate kitchen sites", 6, 20, 12)
    eps        = st.slider("DBSCAN eps", 0.05, 0.4, 0.15, 0.01)
    cost_per_km = st.number_input("Delivery cost/km (₹)", 5, 50, 15)
    run = st.button("🚀 Run Optimizer", type="primary")

CITY_CENTER = (12.9716, 77.5946)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlam = np.radians(lat2-lat1), np.radians(lon2-lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1-a))

if run:
    random.seed(42); np.random.seed(42)

    # Generate orders
    cluster_centers = [(12.9352,77.6245),(13.0100,77.5800),(12.9010,77.6352)]
    lats, lons, demands = [], [], []
    for _ in range(n_orders):
        cx, cy = random.choice(cluster_centers)
        lats.append(cx + np.random.normal(0, 0.02))
        lons.append(cy + np.random.normal(0, 0.02))
        demands.append(random.randint(1, 10))
    orders = pd.DataFrame({"lat": lats, "lon": lons, "demand": demands})

    # DBSCAN
    coords_scaled = StandardScaler().fit_transform(orders[["lat","lon"]].values)
    db = DBSCAN(eps=eps, min_samples=6).fit(coords_scaled)
    orders["cluster"] = db.labels_
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)

    cluster_info = (
        orders[orders.cluster >= 0]
        .groupby("cluster")
        .agg(lat=("lat","mean"), lon=("lon","mean"), demand=("demand","sum"))
        .reset_index()
    )

    # Candidate sites
    sites = pd.DataFrame([{
        "site_id": i,
        "lat": CITY_CENTER[0] + np.random.uniform(-0.08, 0.08),
        "lon": CITY_CENTER[1] + np.random.uniform(-0.08, 0.08),
        "fixed_cost": random.randint(50000, 150000)
    } for i in range(n_sites)])

    # ILP
    I = list(cluster_info.index); J = list(sites.index)
    dist = {(i,j): haversine(cluster_info.loc[i,"lat"],cluster_info.loc[i,"lon"],
                              sites.loc[j,"lat"],sites.loc[j,"lon"]) for i in I for j in J}
    demand_d = cluster_info["demand"].to_dict()
    fixed_d  = sites["fixed_cost"].to_dict()

    prob = pulp.LpProblem("DK", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", [(i,j) for i in I for j in J], cat="Binary")
    y = pulp.LpVariable.dicts("y", J, cat="Binary")
    prob += pulp.lpSum(fixed_d[j]*y[j] for j in J) + pulp.lpSum(dist[i,j]*demand_d[i]*cost_per_km*x[i,j] for i in I for j in J)
    for i in I: prob += pulp.lpSum(x[i,j] for j in J) == 1
    for i in I:
        for j in J: prob += x[i,j] <= y[j]
    prob += pulp.lpSum(y[j] for j in J) == min(K, len(J))
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    open_kitchens = [j for j in J if pulp.value(y[j]) == 1]
    assignments   = {i: next(j for j in J if pulp.value(x[i,j]) == 1) for i in I}
    total_cost    = pulp.value(prob.objective)

    # Metrics
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Orders", n_orders)
    col2.metric("Demand Clusters", n_clusters)
    col3.metric("Kitchens Opened", len(open_kitchens))
    col4.metric("Total Monthly Cost", f"₹{total_cost:,.0f}")

    # Map
    m = folium.Map(location=CITY_CENTER, zoom_start=12, tiles="CartoDB positron")
    colors = ["red","blue","green","purple","orange","darkred"]
    for _, row in orders.iterrows():
        folium.CircleMarker([row.lat,row.lon], radius=2, color="gray", fill=True, fill_opacity=0.2).add_to(m)
    for _, row in cluster_info.iterrows():
        folium.CircleMarker([row.lat,row.lon], radius=10, color="orange", fill=True,
                            tooltip=f"Cluster {int(row.cluster)} | Demand {int(row.demand)}").add_to(m)
    for j in open_kitchens:
        r = sites.loc[j]
        folium.Marker([r.lat,r.lon], icon=folium.Icon(color="green",icon="home"),
                      tooltip=f"Kitchen {j} | ₹{fixed_d[j]:,}/mo").add_to(m)
    for i,j in assignments.items():
        folium.PolyLine([[cluster_info.loc[i,"lat"],cluster_info.loc[i,"lon"]],
                         [sites.loc[j,"lat"],sites.loc[j,"lon"]]],
                        color=colors[j%len(colors)], weight=2).add_to(m)

    st_folium(m, width=900, height=500)

    # Table
    st.subheader("📋 Assignment Summary")
    rows = [{"Demand Zone": f"Cluster {i}", "Demand": demand_d[i],
             "Assigned Kitchen": f"Site {j}", "Distance km": round(dist[i,j],2),
             "Cost ₹": round(dist[i,j]*demand_d[i]*cost_per_km)} for i,j in assignments.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
else:
    st.info("👈 Set parameters in the sidebar and click **Run Optimizer**")
