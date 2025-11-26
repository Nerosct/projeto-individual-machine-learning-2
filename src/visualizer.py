# src/visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import config

class Visualizer:
    """Class for creating visualizations"""
    
    def __init__(self, save_dir=None):
        self.save_dir = Path(save_dir) if save_dir else config.PLOTS_DIR
        self.save_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_elbow_method(self, inertias, save_name=None):
        """Plot elbow method for K-Means"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, len(inertias) + 1), inertias, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('Inertia')
        ax.set_title('Elbow Method - Optimal Number of Clusters')
        ax.grid(True, alpha=0.3)
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_kmeans_results(self, data, labels, feature_names, save_name=None):
        """Plot K-Means clustering results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('K-Means Clustering - Customer Segmentation', fontsize=16, fontweight='bold')
        
        # Recency vs Frequency
        scatter1 = axes[0,0].scatter(data[feature_names[0]], data[feature_names[1]], 
                                   c=labels, cmap='viridis', alpha=0.7)
        axes[0,0].set_xlabel(feature_names[0])
        axes[0,0].set_ylabel(feature_names[1])
        axes[0,0].set_title('Recency vs Frequency')
        plt.colorbar(scatter1, ax=axes[0,0])
        
        # Monetary vs Rating
        scatter2 = axes[0,1].scatter(data[feature_names[2]], data[feature_names[3]], 
                                   c=labels, cmap='plasma', alpha=0.7)
        axes[0,1].set_xlabel(feature_names[2])
        axes[0,1].set_ylabel(feature_names[3])
        axes[0,1].set_title('Monetary vs Rating')
        plt.colorbar(scatter2, ax=axes[0,1])
        
        # Cluster distribution
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
        bars = axes[1,0].bar(cluster_counts.index, cluster_counts.values, 
                           color=colors[:len(cluster_counts)], alpha=0.7)
        axes[1,0].set_xlabel('Cluster')
        axes[1,0].set_ylabel('Number of Customers')
        axes[1,0].set_title('Cluster Distribution')
        axes[1,0].set_xticks(cluster_counts.index)
        
        for bar in bars:
            height = bar.get_height()
            axes[1,0].text(bar.get_x() + bar.get_width()/2., height,
                          f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # PCA visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(data)
        
        scatter3 = axes[1,1].scatter(pca_components[:, 0], pca_components[:, 1], 
                                   c=labels, cmap='viridis', alpha=0.7)
        axes[1,1].set_xlabel('Principal Component 1')
        axes[1,1].set_ylabel('Principal Component 2')
        axes[1,1].set_title('PCA - Cluster Visualization')
        plt.colorbar(scatter3, ax=axes[1,1])
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_dbscan_results(self, data, labels, feature_names, save_name=None):
        """Plot DBSCAN anomaly detection results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('DBSCAN - Anomaly Detection in Products', fontsize=16, fontweight='bold')
        
        # Create masks for anomalies and normals
        mask_anomaly = labels == -1
        mask_normal = labels != -1
        
        # Price vs Quantity
        axes[0,0].scatter(data[feature_names[0]][mask_normal], 
                         data[feature_names[1]][mask_normal], 
                         c='blue', alpha=0.6, s=50, label='Normal')
        axes[0,0].scatter(data[feature_names[0]][mask_anomaly], 
                         data[feature_names[1]][mask_anomaly], 
                         c='red', alpha=1.0, s=80, label='Anomaly', edgecolors='black')
        axes[0,0].set_xlabel(feature_names[0])
        axes[0,0].set_ylabel(feature_names[1])
        axes[0,0].set_title('Price vs Quantity')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Orders vs Rating
        axes[0,1].scatter(data[feature_names[2]][mask_normal], 
                         data[feature_names[3]][mask_normal], 
                         c='green', alpha=0.6, s=50, label='Normal')
        axes[0,1].scatter(data[feature_names[2]][mask_anomaly], 
                         data[feature_names[3]][mask_anomaly], 
                         c='red', alpha=1.0, s=80, label='Anomaly', edgecolors='black')
        axes[0,1].set_xlabel(feature_names[2])
        axes[0,1].set_ylabel(feature_names[3])
        axes[0,1].set_title('Orders vs Rating')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Price distribution
        axes[1,0].hist(data[feature_names[0]], bins=30, alpha=0.7, 
                      color='skyblue', edgecolor='black')
        axes[1,0].set_xlabel(feature_names[0])
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Price Distribution')
        axes[1,0].axvline(x=data[feature_names[0]].quantile(0.95), 
                         color='red', linestyle='--', label='95th percentile')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Anomaly types (if available)
        if hasattr(self, 'anomaly_types'):
            anomaly_types_data = self.anomaly_types
            axes[1,1].bar(anomaly_types_data.keys(), anomaly_types_data.values(), 
                         color='red', alpha=0.7)
            axes[1,1].set_xlabel('Anomaly Type')
            axes[1,1].set_ylabel('Count')
            axes[1,1].set_title('Types of Anomalies Detected')
            plt.setp(axes[1,1].xaxis.get_majorticklabels(), rotation=45)
        else:
            axes[1,1].text(0.5, 0.5, 'No anomaly types analyzed', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Anomaly Types')
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_comparison(self, kmeans_data, dbscan_data, save_name=None):
        """Plot comparison between K-Means and DBSCAN results"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Comparison: K-Means vs DBSCAN', fontsize=16, fontweight='bold')
        
        # K-Means PCA
        from sklearn.decomposition import PCA
        pca_kmeans = PCA(n_components=2)
        kmeans_pca = pca_kmeans.fit_transform(kmeans_data['features'])
        
        scatter1 = axes[0].scatter(kmeans_pca[:, 0], kmeans_pca[:, 1], 
                                 c=kmeans_data['labels'], cmap='viridis', alpha=0.7)
        axes[0].set_xlabel('PC1')
        axes[0].set_ylabel('PC2')
        axes[0].set_title('K-Means: Customer Segmentation')
        plt.colorbar(scatter1, ax=axes[0])
        
        # DBSCAN PCA
        pca_dbscan = PCA(n_components=2)
        dbscan_pca = pca_dbscan.fit_transform(dbscan_data['features'])
        
        mask_anomaly = dbscan_data['labels'] == -1
        mask_normal = dbscan_data['labels'] != -1
        
        axes[1].scatter(dbscan_pca[mask_normal, 0], dbscan_pca[mask_normal, 1], 
                       c='blue', alpha=0.6, s=50, label='Normal')
        axes[1].scatter(dbscan_pca[mask_anomaly, 0], dbscan_pca[mask_anomaly, 1], 
                       c='red', alpha=1.0, s=80, label='Anomaly', edgecolors='black')
        axes[1].set_xlabel('PC1')
        axes[1].set_ylabel('PC2')
        axes[1].set_title('DBSCAN: Anomaly Detection')
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        
        return fig