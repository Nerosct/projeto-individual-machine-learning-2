# src/main.py
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Adicionar o diretório atual ao path para imports
sys.path.append(os.path.dirname(__file__))

from data_loader import DataLoader
from preprocessor import DataPreprocessor
from kmeans_analyzer import KMeansAnalyzer
from dbscan_analyzer import DBSCANAnalyzer
from visualizer import Visualizer

class EcommerceAnalysis:
    """Main class for e-commerce clustering analysis"""
    
    def __init__(self, data_dir=None):
        self.data_loader = DataLoader(data_dir)
        self.preprocessor = DataPreprocessor()
        
        # Configurar diretório de resultados
        self.root_dir = Path(__file__).parent.parent
        self.results_dir = self.root_dir / "results"
        self.plots_dir = self.results_dir / "plots"
        self.reports_dir = self.results_dir / "reports"
        
        # Criar diretórios
        for directory in [self.results_dir, self.plots_dir, self.reports_dir]:
            directory.mkdir(exist_ok=True)
            
        self.visualizer = Visualizer(str(self.plots_dir))
        self.results = {}
    
    def load_and_preprocess_data(self):
        """Load and preprocess all data"""
        print("📊 Loading e-commerce datasets...")
        self.data = self.data_loader.load_all_data()
        
        # Verificar se temos dados mínimos necessários
        required_tables = ['customers', 'orders', 'order_items', 'products', 'reviews']
        available_tables = [table for table in required_tables if table in self.data and not self.data[table].empty]
        
        print(f"✅ Available tables: {available_tables}")
        
        if len(available_tables) < 3:
            print(f"❌ Insufficient data. Need at least 3 of: {required_tables}")
            return False
        
        print("🔄 Preprocessing data and creating features...")
        
        try:
            # Customer features for K-Means
            if 'orders' in self.data and 'order_items' in self.data:
                self.customer_features, self.customer_data = self.preprocessor.prepare_customer_features(
                    self.data['orders'], self.data['order_items'], 
                    self.data.get('reviews', pd.DataFrame())
                )
                print(f"✅ Customer features created: {len(self.customer_features)} records")
            else:
                print("❌ Missing orders or order_items for customer analysis")
                return False
            
            # Product features for DBSCAN
            if 'products' in self.data and 'order_items' in self.data:
                self.product_features, self.product_data = self.preprocessor.prepare_product_features(
                    self.data['products'], self.data['order_items'],
                    self.data.get('reviews', pd.DataFrame())
                )
                print(f"✅ Product features created: {len(self.product_features)} records")
            else:
                print("❌ Missing products or order_items for product analysis")
                return False
            
            # Scale features
            self.customer_features_scaled = self.preprocessor.scale_features(self.customer_features)
            self.product_features_scaled = self.preprocessor.scale_features(self.product_features)
            
            print("✅ Data preprocessing completed!")
            return True
            
        except Exception as e:
            print(f"❌ Error in preprocessing: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_kmeans_analysis(self):
        """Run K-Means clustering analysis"""
        print("\n🎯 Running K-Means clustering analysis...")
        
        try:
            self.kmeans_analyzer = KMeansAnalyzer()
            
            # Find optimal clusters
            print("🔍 Finding optimal number of clusters...")
            inertias = self.kmeans_analyzer.find_optimal_clusters(
                self.customer_features_scaled, max_k=8
            )
            
            # Fit model
            print("🏗️ Fitting K-Means model...")
            labels = self.kmeans_analyzer.fit(self.customer_features_scaled)
            
            # Analyze clusters
            print("📈 Analyzing clusters...")
            analyzed_data, cluster_stats = self.kmeans_analyzer.analyze_clusters(
                self.customer_data, list(self.customer_features.columns)
            )
            
            # Get interpretations
            interpretations = self.kmeans_analyzer.get_cluster_interpretation()
            
            # Store results
            self.results['kmeans'] = {
                'labels': labels,
                'analyzed_data': analyzed_data,
                'cluster_stats': cluster_stats,
                'interpretations': interpretations,
                'silhouette_score': self.kmeans_analyzer.silhouette,
                'inertias': inertias
            }
            
            print(f"✅ K-Means analysis completed! Silhouette Score: {self.kmeans_analyzer.silhouette:.3f}")
            return True
            
        except Exception as e:
            print(f"❌ Error in K-Means analysis: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_dbscan_analysis(self):
        """Run DBSCAN anomaly detection analysis"""
        print("\n🔍 Running DBSCAN anomaly detection analysis...")
        
        try:
            self.dbscan_analyzer = DBSCANAnalyzer()
            
            # Fit model
            print("🏗️ Fitting DBSCAN model...")
            labels = self.dbscan_analyzer.fit(self.product_features_scaled)
            
            # Analyze anomalies
            print("📈 Analyzing anomalies...")
            analyzed_data, anomaly_stats = self.dbscan_analyzer.analyze_anomalies(
                self.product_data, list(self.product_features.columns)
            )
            
            # Classify anomaly types
            anomaly_types = self.dbscan_analyzer.classify_anomaly_types(
                list(self.product_features.columns)
            )
            
            # Get detailed anomalies
            detailed_anomalies = self.dbscan_analyzer.get_anomaly_details(self.product_data)
            
            # Store results
            self.results['dbscan'] = {
                'labels': labels,
                'analyzed_data': analyzed_data,
                'anomaly_stats': anomaly_stats,
                'anomaly_types': anomaly_types,
                'detailed_anomalies': detailed_anomalies,
                'n_anomalies': self.dbscan_analyzer.results['n_anomalies'],
                'anomaly_percentage': self.dbscan_analyzer.results['anomaly_percentage']
            }
            
            print(f"✅ DBSCAN analysis completed! Anomalies detected: {self.dbscan_analyzer.results['n_anomalies']}")
            return True
            
        except Exception as e:
            print(f"❌ Error in DBSCAN analysis: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_visualizations(self):
        """Generate all visualizations"""
        print("\n🎨 Generating visualizations...")
        
        try:
            # K-Means visualizations
            print("📊 Creating K-Means visualizations...")
            self.visualizer.plot_elbow_method(
                self.results['kmeans']['inertias'], 
                'kmeans_elbow_method'
            )
            
            self.visualizer.plot_kmeans_results(
                self.customer_features,
                self.results['kmeans']['labels'],
                list(self.customer_features.columns),
                'kmeans_clustering_results'
            )
            
            # DBSCAN visualizations
            print("📊 Creating DBSCAN visualizations...")
            if hasattr(self.results['dbscan'], 'anomaly_types'):
                self.visualizer.anomaly_types = self.results['dbscan']['anomaly_types']
            self.visualizer.plot_dbscan_results(
                self.product_features,
                self.results['dbscan']['labels'], 
                list(self.product_features.columns),
                'dbscan_anomaly_detection'
            )
            
            # Comparison visualization
            print("📊 Creating comparison visualization...")
            comparison_data = {
                'kmeans': {
                    'features': self.customer_features_scaled,
                    'labels': self.results['kmeans']['labels']
                },
                'dbscan': {
                    'features': self.product_features_scaled,
                    'labels': self.results['dbscan']['labels']
                }
            }
            
            self.visualizer.plot_comparison(
                comparison_data['kmeans'],
                comparison_data['dbscan'],
                'methods_comparison'
            )
            
            print("✅ Visualizations generated and saved!")
            return True
            
        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\n📈 Generating analysis report...")
        
        try:
            kmeans_results = self.results['kmeans']
            dbscan_results = self.results['dbscan']
            
            report = {
                'overview': {
                    'total_customers': len(self.customer_data),
                    'total_products': len(self.product_data),
                    'total_orders': len(self.data['orders']) if 'orders' in self.data else 0,
                    'total_reviews': len(self.data['reviews']) if 'reviews' in self.data else 0
                },
                'kmeans_summary': {
                    'n_clusters': kmeans_results['cluster_stats'].shape[0],
                    'silhouette_score': kmeans_results['silhouette_score'],
                    'cluster_distribution': kmeans_results['analyzed_data']['cluster'].value_counts().to_dict(),
                    'interpretations': kmeans_results['interpretations']
                },
                'dbscan_summary': {
                    'n_anomalies': dbscan_results['n_anomalies'],
                    'anomaly_percentage': dbscan_results['anomaly_percentage'],
                    'anomaly_types': dbscan_results['anomaly_types'],
                    'n_clusters': len(set(dbscan_results['labels'])) - (1 if -1 in dbscan_results['labels'] else 0)
                }
            }
            
            # Save report
            report_df = pd.DataFrame([
                {'Metric': 'Total Customers', 'Value': report['overview']['total_customers']},
                {'Metric': 'Total Products', 'Value': report['overview']['total_products']},
                {'Metric': 'Total Orders', 'Value': report['overview']['total_orders']},
                {'Metric': 'Total Reviews', 'Value': report['overview']['total_reviews']},
                {'Metric': 'K-Means Silhouette Score', 'Value': f"{report['kmeans_summary']['silhouette_score']:.3f}"},
                {'Metric': 'DBSCAN Anomalies Detected', 'Value': report['dbscan_summary']['n_anomalies']},
                {'Metric': 'Anomaly Percentage', 'Value': f"{report['dbscan_summary']['anomaly_percentage']:.2f}%"}
            ])
            
            report_df.to_csv(self.reports_dir / 'analysis_summary.csv', index=False)
            
            # Save detailed results
            kmeans_results['analyzed_data'].to_csv(self.reports_dir / 'customer_segmentation.csv', index=False)
            dbscan_results['analyzed_data'].to_csv(self.reports_dir / 'product_anomalies.csv', index=False)
            
            print("✅ Analysis report generated!")
            return report
            
        except Exception as e:
            print(f"❌ Error generating report: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def run_complete_analysis(self):
        """Run complete analysis pipeline"""
        print("🚀 Starting Complete E-commerce Analysis Pipeline")
        print("=" * 60)
        
        try:
            # 1. Load and preprocess data
            if not self.load_and_preprocess_data():
                print("❌ Failed to load and preprocess data")
                return False
            
            # 2. Run analyses
            if not self.run_kmeans_analysis():
                print("❌ K-Means analysis failed")
                return False
                
            if not self.run_dbscan_analysis():
                print("❌ DBSCAN analysis failed")
                return False
            
            # 3. Generate visualizations
            if not self.generate_visualizations():
                print("❌ Visualization generation failed")
                return False
            
            # 4. Generate report
            report = self.generate_report()
            
            # 5. Print summary
            self.print_summary(report)
            
            print("\n🎉 Analysis completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def print_summary(self, report):
        """Print analysis summary"""
        print("\n" + "=" * 60)
        print("📊 ANALYSIS SUMMARY")
        print("=" * 60)
        
        if not report:
            print("No report generated")
            return
            
        print(f"\n📈 OVERVIEW:")
        print(f"   • Customers analyzed: {report['overview']['total_customers']}")
        print(f"   • Products analyzed: {report['overview']['total_products']}")
        print(f"   • Orders processed: {report['overview']['total_orders']}")
        print(f"   • Reviews analyzed: {report['overview']['total_reviews']}")
        
        print(f"\n🎯 K-MEANS RESULTS:")
        print(f"   • Clusters identified: {report['kmeans_summary']['n_clusters']}")
        print(f"   • Silhouette Score: {report['kmeans_summary']['silhouette_score']:.3f}")
        print(f"   • Cluster distribution:")
        for cluster, count in report['kmeans_summary']['cluster_distribution'].items():
            percentage = (count / report['overview']['total_customers']) * 100
            interpretation = report['kmeans_summary']['interpretations'][cluster]['interpretation']
            print(f"     - Cluster {cluster}: {count} customers ({percentage:.1f}%) - {interpretation}")
        
        print(f"\n🔍 DBSCAN RESULTS:")
        print(f"   • Anomalies detected: {report['dbscan_summary']['n_anomalies']}")
        print(f"   • Anomaly rate: {report['dbscan_summary']['anomaly_percentage']:.2f}%")
        print(f"   • Clusters found: {report['dbscan_summary']['n_clusters']}")
        if report['dbscan_summary']['anomaly_types']:
            print(f"   • Anomaly types:")
            for anomaly_type, count in report['dbscan_summary']['anomaly_types'].items():
                print(f"     - {anomaly_type}: {count} products")

if __name__ == "__main__":
    # Run complete analysis
    print("🛒 E-COMMERCE CLUSTERING ANALYSIS")
    print("=" * 50)
    
    analysis = EcommerceAnalysis()
    success = analysis.run_complete_analysis()
    
    if success:
        print("\n" + "=" * 50)
        print("✅ ALL ANALYSES COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("📁 Check the 'results' folder for:")
        print("   • plots/ - Visualizations and charts")
        print("   • reports/ - CSV files with detailed results")
    else:
        print("\n" + "=" * 50)
        print("❌ ANALYSIS FAILED")
        print("=" * 50)
        print("Please check the error messages above and ensure all required CSV files are in the 'data' folder.")