# src/preprocessor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Class for data preprocessing and feature engineering"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.features_created = False
    
    def prepare_customer_features(self, orders, order_items, reviews):
        """Create customer features for clustering (RFM + Reviews)"""
        
        # Basic RFM features from orders
        customer_orders = orders.groupby('customer_id').agg({
            'order_id': 'count',
            'total_price': ['sum', 'mean'],
            'order_date': ['min', 'max']
        }).reset_index()
        
        customer_orders.columns = [
            'customer_id', 'order_count', 'total_spent', 
            'avg_order_value', 'first_order', 'last_order'
        ]
        
        # Calculate recency (assuming current date)
        customer_orders['last_order'] = pd.to_datetime(customer_orders['last_order'])
        current_date = pd.to_datetime('2024-01-01')  # Adjust as needed
        customer_orders['recency'] = (current_date - customer_orders['last_order']).dt.days
        
        # Review features
        customer_reviews = reviews.groupby('customer_id').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        customer_reviews.columns = ['customer_id', 'avg_rating', 'review_count']
        
        # Merge features
        customer_features = customer_orders.merge(
            customer_reviews, on='customer_id', how='left'
        ).fillna(0)
        
        # Select final features for clustering
        features_for_clustering = customer_features[[
            'recency', 'order_count', 'total_spent', 'avg_rating'
        ]]
        
        # Rename for RFM convention
        features_for_clustering.columns = [
            'recency', 'frequency', 'monetary', 'avg_rating'
        ]
        
        self.customer_features = customer_features
        self.features_for_clustering = features_for_clustering
        self.features_created = True
        
        return features_for_clustering, customer_features
    
    def prepare_product_features(self, products, order_items, reviews):
        """Create product features for anomaly detection"""
        
        # Sales features from order_items
        product_sales = order_items.groupby('product_id').agg({
            'quantity': ['sum', 'mean', 'count'],
            'price_at_purchase': ['mean', 'std']
        }).reset_index()
        
        product_sales.columns = [
            'product_id', 'total_quantity', 'avg_quantity', 
            'order_count', 'avg_price', 'price_std'
        ]
        
        # Review features
        product_reviews = reviews.groupby('product_id').agg({
            'rating': 'mean'
        }).reset_index()
        product_reviews.columns = ['product_id', 'avg_rating']
        
        # Merge with product information
        product_features = products.merge(
            product_sales, on='product_id', how='left'
        ).merge(
            product_reviews, on='product_id', how='left'
        ).fillna(0)
        
        # Select features for anomaly detection
        features_for_anomaly = product_features[[
            'price', 'total_quantity', 'order_count', 'avg_rating'
        ]]
        
        self.product_features = product_features
        self.features_for_anomaly = features_for_anomaly
        
        return features_for_anomaly, product_features
    
    def scale_features(self, features):
        """Scale features using StandardScaler"""
        return self.scaler.fit_transform(features)
    
    def get_feature_names(self):
        """Get feature names"""
        if hasattr(self, 'features_for_clustering'):
            return list(self.features_for_clustering.columns)
        return []