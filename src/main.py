# src/main.py
import pandas as pd
import numpy as np
from data_loader import DataLoader
from preprocessor import DataPreprocessor
from kmeans_analyzer import KMeansAnalyzer
from dbscan_analyzer import DBSCANAnalyzer
from visualizer import Visualizer
import config

class EcommerceAnalysis:
    """Main class for e-commerce clustering analysis"""
    
    def __init__(self, data_dir=None):
        self.data_loader = DataLoader(data_dir)
        self.preprocessor = DataPreprocessor()
        self.visualizer = Visualizer()
        self.results = {}
    
    def load_and_preprocess_data(self):
        """Load and preprocess all data"""
        print("📊 Loading e-commerce datasets...")
        self.data = self.data_loader.load_all_data()
        
        print("🔄 Preprocessing data and creating features...")
        # Customer features for K-Means
        self.customer_features, self.customer_data = self.preprocessor.prepare_customer_features(
            self.data['orders'], self.data['order_items'], self.data['reviews']
        )
        
        # Product features for DBSCAN
        self.product_features, self.product_data = self.preprocessor.prepare_product_features(
            self.data['products'], self.data['order_items'], self.data['reviews']
        )
        
        # Scale features
        self.customer_features_scaled = self.preprocessor.scale_features(self.customer_features)
        self.product_features_scaled = self.preprocessor.scale_features(self.product_features)
        
        print("✅ Data preprocessing completed!")
    
    def run_kmeans_analysis(self):
        """Run K-Means clustering analysis"""
        print("\n🎯 Running K-Means clustering analysis...")
        
        self.kmeans_analyzer = KMeansAnalyzer()
        
        # Find optimal clusters
        inertias = self.kmeans_analyzer.find_optimal_clusters(
            self.customer_features_scaled, max_k=10
        )
        
        # Fit model
        labels = self.kmeans_analyzer.fit(self.customer_features_scaled)
        
        # Analyze clusters
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
        
        return self.results['kmeans']
    
    def run_dbscan_analysis(self):
        """Run DBSCAN anomaly detection analysis"""
        print("\n🔍 Running DBSCAN anomaly detection analysis...")
        
        self.dbscan_analyzer = DBSCANAnalyzer()
        
        # Fit model
        labels = self.dbscan_analyzer.fit(self.product_features_scaled)
        
        # Analyze anomalies
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
        
        return self.results['dbscan']
    
    def generate_visualizations(self):
        """Generate all visualizations"""
        print("\n🎨 Generating visualizations...")
        
        # K-Means visualizations
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
        self.visualizer.anomaly_types = self.results['dbscan']['anomaly_types']
        self.visualizer.plot_dbscan_results(
            self.product_features,
            self.results['dbscan']['labels'], 
            list(self.product_features.columns),
            'dbscan_anomaly_detection'
        )
        
        # Comparison visualization
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
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\n📈 Generating analysis report...")
        
        kmeans_results = self.results['kmeans']
        dbscan_results = self.results['dbscan']
        
        report = {
            'overview': {
                'total_customers': len(self.customer_data),
                'total_products': len(self.product_data),
                'total_orders': len(self.data['orders']),
                'total_reviews': len(self.data['reviews'])
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
            {'Metric': 'K-Means Silhouette Score', 'Value': report['kmeans_summary']['silhouette_score']},
            {'Metric': 'DBSCAN Anomalies Detected', 'Value': report['dbscan_summary']['n_anomalies']},
            {'Metric': 'Anomaly Percentage', 'Value': f"{report['dbscan_summary']['anomaly_percentage']:.2f}%"}
        ])
        
        report_df.to_csv(config.REPORTS_DIR / 'analysis_summary.csv', index=False)
        
        print("✅ Analysis report generated!")
        return report
    
    def run_complete_analysis(self):
        """Run complete analysis pipeline"""
        print("🚀 Starting Complete E-commerce Analysis Pipeline")
        print("=" * 60)
        
        try:
            # 1. Load and preprocess data
            self.load_and_preprocess_data()
            
            # 2. Run analyses
            self.run_kmeans_analysis()
            self.run_dbscan_analysis()
            
            # 3. Generate visualizations
            self.generate_visualizations()
            
            # 4. Generate report
            report = self.generate_report()
            
            # 5. Print summary
            self.print_summary(report)
            
            print("\n🎉 Analysis completed successfully!")
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            raise
    
    def print_summary(self, report):
        """Print analysis summary"""
        print("\n" + "=" * 60)
        print("📊 ANALYSIS SUMMARY")
        print("=" * 60)
        
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
    analysis = EcommerceAnalysis()
    analysis.run_complete_analysis()