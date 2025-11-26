# src/kmeans_analyzer.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

class KMeansAnalyzer:
    """K-Means clustering analysis for customer segmentation"""
    
    def __init__(self, n_clusters=4, random_state=42):
        self.n_clusters = n_clusters
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
        
        print(f"🔍 Analyzing clusters with features: {feature_names}")
        print(f"📊 Original data columns: {list(original_data.columns)}")
        
        # Add cluster labels to original data
        analyzed_data = original_data.copy()
        analyzed_data['cluster'] = self.labels
        
        # Verificar quais colunas existem nos dados
        available_columns = analyzed_data.columns.tolist()
        print(f"📋 Available columns: {available_columns}")
        
        # Mapear nomes das features para colunas reais
        column_mapping = {
            'recency': 'recency',
            'frequency': 'frequency', 
            'monetary': 'monetary',
            'avg_rating': 'avg_rating',
            'customer_id': 'customer_id'
        }
        
        # Criar agregação dinâmica baseada nas colunas disponíveis
        agg_dict = {}
        for feature in feature_names:
            if feature in analyzed_data.columns:
                agg_dict[feature] = ['mean', 'std']
            else:
                print(f"⚠️ Column '{feature}' not found in data")
        
        # Sempre incluir customer_id para contar
        if 'customer_id' in analyzed_data.columns:
            agg_dict['customer_id'] = 'count'
        
        print(f"📈 Aggregation dictionary: {agg_dict}")
        
        # Cluster statistics
        cluster_stats = analyzed_data.groupby('cluster').agg(agg_dict).round(2)
        
        # Flatten column names
        if isinstance(cluster_stats.columns, pd.MultiIndex):
            cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]
        else:
            # Já está flat
            pass
        
        print(f"📊 Cluster stats columns: {list(cluster_stats.columns)}")
        
        self.cluster_stats = cluster_stats
        self.analyzed_data = analyzed_data
        
        return analyzed_data, cluster_stats
    
    def get_cluster_interpretation(self):
        """Provide business interpretation for each cluster"""
        if not hasattr(self, 'cluster_stats'):
            raise ValueError("Run analyze_clusters first")
        
        interpretations = {}
        stats = self.cluster_stats
        
        # Encontrar nomes das colunas estatísticas
        recency_col = None
        frequency_col = None
        monetary_col = None
        rating_col = None
        
        for col in stats.columns:
            if 'recency' in col and 'mean' in col:
                recency_col = col
            elif 'frequency' in col and 'mean' in col:
                frequency_col = col
            elif 'monetary' in col and 'mean' in col:
                monetary_col = col
            elif 'avg_rating' in col and 'mean' in col:
                rating_col = col
            elif 'customer_id_count' in col:
                count_col = col
        
        print(f"📊 Using columns for interpretation:")
        print(f"   Recency: {recency_col}")
        print(f"   Frequency: {frequency_col}")
        print(f"   Monetary: {monetary_col}")
        print(f"   Rating: {rating_col}")
        
        for cluster in stats.index:
            recency = stats.loc[cluster, recency_col] if recency_col else 0
            frequency = stats.loc[cluster, frequency_col] if frequency_col else 0
            monetary = stats.loc[cluster, monetary_col] if monetary_col else 0
            rating = stats.loc[cluster, rating_col] if rating_col else 0
            
            # Business logic for interpretation
            if recency > 180 and frequency < 2:
                interpretation = "🥉 Clientes Inativos"
            elif frequency > 5 and monetary > (stats[monetary_col].mean() if monetary_col else 0):
                interpretation = "🥇 Clientes Premium" 
            elif monetary > (stats[monetary_col].quantile(0.75) if monetary_col else 0):
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