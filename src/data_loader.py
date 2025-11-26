# src/data_loader.py
import pandas as pd
from pathlib import Path
import sys
import os

# Importação direta do config
try:
    import config
except ImportError:
    # Fallback: configuração inline
    ROOT_DIR = Path(__file__).parent.parent
    DATA_DIR = ROOT_DIR / "data"
    RESULTS_DIR = ROOT_DIR / "results"
    
    DATA_FILES = {
        'customers': 'customers.csv',
        'products': 'products.csv', 
        'orders': 'orders.csv',
        'order_items': 'order_items.csv',
        'suppliers': 'suppliers.csv',
        'reviews': 'reviews.csv',
        'payments': 'payment.csv',  # CORRIGIDO: payment.csv em vez de payments.csv
        'shipments': 'shipments.csv'
    }
    
    # Criar variáveis globais como fallback
    config = type('Config', (), {
        'DATA_DIR': DATA_DIR,
        'DATA_FILES': DATA_FILES
    })()

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
            print(f"⚠️ Missing files: {missing_files}")
        else:
            print("✅ All data files found!")
    
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
        data_dict = {}
        
        # Carregar apenas os arquivos que existem
        for data_name, file_name in config.DATA_FILES.items():
            file_path = self.data_dir / file_name
            if file_path.exists():
                try:
                    data_dict[data_name] = pd.read_csv(file_path)
                    print(f"✅ Loaded {data_name}: {len(data_dict[data_name])} records")
                    print(f"   Columns: {list(data_dict[data_name].columns)}")
                except Exception as e:
                    print(f"❌ Error loading {data_name}: {e}")
                    data_dict[data_name] = pd.DataFrame()
            else:
                print(f"⚠️ File not found: {file_name}")
                data_dict[data_name] = pd.DataFrame()
        
        return data_dict