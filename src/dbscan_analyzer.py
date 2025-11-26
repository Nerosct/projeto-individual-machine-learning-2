# src/dbscan_analyzer.py
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import config

class DBSCANAnalyzer:
    """DBSCAN analysis for anomaly detection"""
    
    def __init__(self, eps=None, min_samples=None):
        self.eps = eps or config.DBSCAN_PARAMS['eps']
        self.min_samples = min_samples or config.DBSCAN_PARAMS['min_samples']
        self.model = None
        self.results = {}
    
    def find_optimal_eps(self, data, k=4):
        """Find optimal eps parameter using k-distance graph"""
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(data)
        distances, indices = neighbors_fit.kneighbors(data)
        
        distances = np.sort(distances[:, k-1], axis=0)
        return distances
    
    def fit(self, data, eps=None, min_samples=None):
        """Fit DBSCAN model to data"""
        if eps:
            self.eps = eps
        if min_samples:
            self.min_samples = min_samples
        
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels = self.model.fit_predict(data)
        
        # Calculate anomaly statistics
        n_anomalies = sum(self.labels == -1)
        anomaly_percentage = (n_anomalies / len(data)) * 100
        
        self.results = {
            'labels': self.labels,
            'n_anomalies': n_anomalies,
            'anomaly_percentage': anomaly_percentage,
            'n_clusters': len(set(self.labels)) - (1 if -1 in self.labels else 0)
        }
        
        return self.labels
    
    def analyze_anomalies(self, original_data, feature_names):
        """Analyze detected anomalies"""
        if not hasattr(self, 'labels'):
            raise ValueError("Model must be fitted first")
        
        analyzed_data = original_data.copy()
        analyzed_data['cluster_dbscan'] = self.labels
        analyzed_data['is_anomaly'] = analyzed_data['cluster_dbscan'] == -1
        
        # Anomaly statistics
        anomalies = analyzed_data[analyzed_data['is_anomaly'] == True]
        normals = analyzed_data[analyzed_data['is_anomaly'] == False]
        
        if len(anomalies) > 0:
            anomaly_stats = anomalies[feature_names].describe()
        else:
            anomaly_stats = pd.DataFrame()
        
        self.anomalies = anomalies
        self.normals = normals
        self.analyzed_data = analyzed_data
        self.anomaly_stats = anomaly_stats
        
        return analyzed_data, anomaly_stats
    
    def classify_anomaly_types(self, feature_names):
        """Classify anomalies into different types"""
        if not hasattr(self, 'anomalies') or len(self.anomalies) == 0:
            return {}
        
        anomalies = self.anomalies
        normals = self.normals
        
        # Define thresholds based on normal data
        thresholds = {}
        for feature in feature_names:
            q1 = normals[feature].quantile(0.25)
            q3 = normals[feature].quantile(0.75)
            iqr = q3 - q1
            thresholds[feature] = {
                'high': q3 + 1.5 * iqr,
                'low': q1 - 1.5 * iqr
            }
        
        # Classify anomalies
        anomaly_types = {
            'high_price': len(anomalies[anomalies[feature_names[0]] > thresholds[feature_names[0]]['high']]),
            'low_quantity': len(anomalies[anomalies[feature_names[1]] < thresholds[feature_names[1]]['low']]),
            'low_orders': len(anomalies[anomalies[feature_names[2]] < thresholds[feature_names[2]]['low']]),
            'low_rating': len(anomalies[anomalies[feature_names[3]] < 2.5])  # Absolute threshold for ratings
        }
        
        self.anomaly_types = anomaly_types
        self.thresholds = thresholds
        
        return anomaly_types
    
    def get_anomaly_details(self, product_data=None):
        """Get detailed information about anomalies"""
        if not hasattr(self, 'anomalies'):
            return pd.DataFrame()
        
        if product_data is not None:
            detailed_anomalies = self.anomalies.merge(
                product_data[['product_id', 'product_name', 'category']], 
                on='product_id', how='left'
            )
            return detailed_anomalies
        else:
            return self.anomalies