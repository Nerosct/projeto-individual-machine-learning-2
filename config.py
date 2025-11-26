# config.py
import os
from pathlib import Path

# Paths configuration
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
REPORTS_DIR = RESULTS_DIR / "reports"

# Create directories
for directory in [DATA_DIR, RESULTS_DIR, PLOTS_DIR, REPORTS_DIR]:
    directory.mkdir(exist_ok=True)

# Model parameters
KMEANS_PARAMS = {
    'n_clusters': 4,
    'random_state': 42,
    'n_init': 10
}

DBSCAN_PARAMS = {
    'eps': 0.8,
    'min_samples': 5
}

# File names
DATA_FILES = {
    'customers': 'customers.csv',
    'products': 'products.csv', 
    'orders': 'orders.csv',
    'order_items': 'order_items.csv',
    'suppliers': 'suppliers.csv',
    'reviews': 'reviews.csv',
    'payments': 'payments.csv',
    'shipments': 'shipments.csv'
}