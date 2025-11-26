# src/data_loader.py
import pandas as pd
from pathlib import Path
import config

class DataLoader:
    """Class to load and validate e-commerce datasets"""
    
    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir) if data_dir else config.DATA_DIR
        self._validate_data_files()
    
    def _validate_data_files(self):
        """Check if all required data files exist"""
        missing_files = []
        for file_name in config.DATA_FILES.values():
            if not (self.data_dir / file_name).exists():
                missing_files.append(file_name)
        
        if missing_files:
            raise FileNotFoundError(f"Missing data files: {missing_files}")
    
    def load_customers(self):
        """Load customers dataset"""
        return pd.read_csv(self.data_dir / config.DATA_FILES['customers'])
    
    def load_products(self):
        """Load products dataset"""
        return pd.read_csv(self.data_dir / config.DATA_FILES['products'])
    
    def load_orders(self):
        """Load orders dataset"""
        return pd.read_csv(self.data_dir / config.DATA_FILES['orders'])
    
    def load_order_items(self):
        """Load order items dataset"""
        return pd.read_csv(self.data_dir / config.DATA_FILES['order_items'])
    
    def load_suppliers(self):
        """Load suppliers dataset"""
        return pd.read_csv(self.data_dir / config.DATA_FILES['suppliers'])
    
    def load_reviews(self):
        """Load reviews dataset"""
        return pd.read_csv(self.data_dir / config.DATA_FILES['reviews'])
    
    def load_payments(self):
        """Load payments dataset"""
        return pd.read_csv(self.data_dir / config.DATA_FILES['payments'])
    
    def load_shipments(self):
        """Load shipments dataset"""
        return pd.read_csv(self.data_dir / config.DATA_FILES['shipments'])
    
    def load_all_data(self):
        """Load all datasets into a dictionary"""
        return {
            'customers': self.load_customers(),
            'products': self.load_products(),
            'orders': self.load_orders(),
            'order_items': self.load_order_items(),
            'suppliers': self.load_suppliers(),
            'reviews': self.load_reviews(),
            'payments': self.load_payments(),
            'shipments': self.load_shipments()
        }