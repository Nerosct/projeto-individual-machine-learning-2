# src/kmeans_analyzer.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import config

class KMeansAnalyzer:
    """K-Means clustering analysis for customer segmentation"""
    
    def __init__(self, n_clusters=None, random_state=42):
        self.n_clusters = n_clusters or config.KMEANS_PARAMS['n_clusters']
        self.random_state = random_state
        self.model = None
        self.results = {}
    
    def find_optimal_clusters(self, data, max_k=10):
        """Find optimal number of clusters using elbow method"""
        inertias = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
        
        self.inertias = inertias
        return inertias
    
    def fit(self, data, n_clusters=None):
        """Fit K-Means model to data"""
        if n_clusters:
            self.n_clusters = n_clusters
        
        self.model = KMeans(
            n_clusters=self.n_clusters, 
            random_state=self.random_state,
            n_init=10
        )
        
        self.labels = self.model.fit_predict(data)
        self.silhouette = silhouette_score(data, self.labels)
        
        # Store results
        self.results = {
            'labels': self.labels,
            'silhouette_score': self.silhouette,
            'inertia': self.model.inertia_,
            'n_clusters': self.n_clusters
        }
        
        return self.labels
    
    def analyze_clusters(self, original_data, feature_names):
        """Analyze and interpret clusters"""
        if not hasattr(self, 'labels'):
            raise ValueError("Model must be fitted first")
        
        # Add cluster labels to original data
        analyzed_data = original_data.copy()
        analyzed_data['cluster'] = self.labels
        
        # Cluster statistics
        cluster_stats = analyzed_data.groupby('cluster').agg({
            feature_names[0]: ['mean', 'std'],
            feature_names[1]: ['mean', 'std'],
            feature_names[2]: ['mean', 'std'],
            feature_names[3]: ['mean', 'std'],
            'customer_id': 'count'
        }).round(2)
        
        # Flatten column names
        cluster_stats.columns = [
            'recency_mean', 'recency_std',
            'frequency_mean', 'frequency_std', 
            'monetary_mean', 'monetary_std',
            'rating_mean', 'rating_std',
            'customer_count'
        ]
        
        self.cluster_stats = cluster_stats
        self.analyzed_data = analyzed_data
        
        return analyzed_data, cluster_stats
    
    def get_cluster_interpretation(self):
        """Provide business interpretation for each cluster"""
        if not hasattr(self, 'cluster_stats'):
            raise ValueError("Run analyze_clusters first")
        
        interpretations = {}
        stats = self.cluster_stats
        
        for cluster in stats.index:
            recency = stats.loc[cluster, 'recency_mean']
            frequency = stats.loc[cluster, 'frequency_mean']
            monetary = stats.loc[cluster, 'monetary_mean']
            rating = stats.loc[cluster, 'rating_mean']
            
            # Business logic for interpretation
            if recency > 180 and frequency < 2:
                interpretation = "🥉 Clientes Inativos"
            elif frequency > 5 and monetary > stats['monetary_mean'].mean():
                interpretation = "🥇 Clientes Premium" 
            elif monetary > stats['monetary_mean'].quantile(0.75):
                interpretation = "💎 Clientes de Alto Valor"
            else:
                interpretation = "🥈 Clientes Regulares"
            
            interpretations[cluster] = {
                'interpretation': interpretation,
                'recency': recency,
                'frequency': frequency,
                'monetary': monetary,
                'rating': rating
            }
        
        self.interpretations = interpretations
        return interpretations
    
    def get_pca_components(self, data, n_components=2):
        """Get PCA components for visualization"""
        pca = PCA(n_components=n_components)
        pca_components = pca.fit_transform(data)
        return pca_components, pca