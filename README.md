# 🍽️ Dark Kitchen Delivery Zone Optimizer

> **DS + OR hybrid project** — Cluster food-order demand with DBSCAN, then place dark kitchens optimally using a Facility Location Integer Linear Program.

## 🎯 Problem Statement
Dark kitchens (ghost kitchens) must be placed where they minimise total delivery cost across all demand zones. Too few kitchens = long delivery times. Too many = high fixed costs. This project finds the optimal K locations.

## 🏗️ Architecture
```
Raw Order Data (lat/lon/demand)
        │
        ▼
  DBSCAN Clustering          ← DS Layer
  (identify demand hotspots)
        │
        ▼
  Facility Location ILP      ← OR Layer
  Minimise: Σ fixed_cost·yⱼ + Σ dist·demand·cost·xᵢⱼ
  Subject to: assignment & open constraints
        │
        ▼
  Folium Map + Assignment Table
```

## 📦 Tech Stack
| Layer | Tool |
|-------|------|
| Data Science | `scikit-learn` DBSCAN, `pandas`, `numpy` |
| Optimisation | `PuLP` + CBC solver (Facility Location ILP) |
| Geo | `folium`, Haversine distance |
| Dashboard | `streamlit`, `streamlit-folium` |

## 🚀 Quick Start
```bash
pip install -r requirements.txt

# Run script
python dark_kitchen_optimizer.py

# Launch interactive dashboard
streamlit run app.py
```

## 📊 What You Get
- Demand heatmap of order locations
- DBSCAN cluster centroids (customer zones)
- ILP-optimal K kitchen locations marked on a Folium map
- Assignment lines showing which kitchen serves which zone
- Monthly cost breakdown table

## 🔑 Key Concepts
- **Facility Location Problem** — classic OR problem (NP-hard in general, solved with ILP for small instances)
- **DBSCAN** — density-based clustering, great for geo data with noise
- **Haversine distance** — accurate great-circle distance for lat/lon

## 📁 Files
```
01_dark_kitchen_optimizer/
├── dark_kitchen_optimizer.py   # Full pipeline (data → cluster → ILP → map)
├── app.py                      # Streamlit interactive dashboard
├── requirements.txt
└── README.md
```

## 💼 Interview Talking Points
1. Why DBSCAN over K-Means for geo demand? *(handles noise, no need to pre-specify K)*
2. Why ILP over greedy selection? *(global optimum, hard constraints on budget)*
3. How would you scale this to 100,000 orders? *(sample/aggregate first, Lagrangian relaxation for large ILP)*

## 📈 Sample Output
```
[DBSCAN] Found 3 demand clusters
[OR] Status : Optimal
[OR] Total cost : ₹4,23,800
[OR] Open kitchen sites : [2, 7, 11]
```
