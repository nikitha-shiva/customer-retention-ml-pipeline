"""
Advanced Feature Engineering for Customer Retention ML Pipeline
Comprehensive feature creation with PySpark UDFs and real-time processing
Author: Nikitha Shiva
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomerFeatureEngineer:
    """
    Advanced feature engineering for customer retention modeling
    Implements RFM analysis, CLV prediction, and behavioral scoring
    """
    
    def __init__(self, spark_session: SparkSession):
        """Initialize feature engineer with Spark session"""
        self.spark = spark_session
        
        # Register custom UDFs
        self._register_udfs()
        
        # Feature configuration
        self.feature_config = {
            'recency_windows': [7, 30, 90, 180, 365],
            'frequency_windows': [30, 90, 180, 365],
            'monetary_windows': [30, 90, 180, 365],
            'engagement_weights': {
                'login_frequency': 0.3,
                'feature_usage': 0.25,
                'support_interaction': 0.2,
                'community_participation': 0.15,
                'feedback_submission': 0.1
            }
        }
        
        logger.info("CustomerFeatureEngineer initialized successfully")
    
    def _register_udfs(self):
        """Register custom UDFs for advanced calculations"""
        
        # Customer Lifetime Value UDF
        @udf(returnType=DoubleType())
        def calculate_clv_score(total_revenue, avg_order_value, purchase_frequency, tenure_months):
            """Calculate Customer Lifetime Value with predictive modeling"""
            if not all([total_revenue, tenure_months]) or tenure_months == 0:
                return 0.0
            
            # Base monthly value
            monthly_value = float(total_revenue) / max(float(tenure_months), 1)
            
            # Predicted lifetime (with growth assumption)
            predicted_lifetime = min(float(tenure_months) * 1.2, 60)  # Cap at 5 years
            
            # Purchase frequency multiplier
            freq_multiplier = min(float(purchase_frequency or 0) * 2, 3.0)
            
            # Value tier adjustment
            aov = float(avg_order_value or 0)
            if aov > 500:
                value_tier = 1.5
            elif aov > 200:
                value_tier = 1.2
            elif aov > 50:
                value_tier = 1.0
            else:
                value_tier = 0.8
            
            clv_score = monthly_value * predicted_lifetime * freq_multiplier * value_tier
            return float(clv_score)
        
        # Risk Score UDF
        @udf(returnType=DoubleType())
        def calculate_risk_score(recency, frequency, monetary, engagement, support_tickets):
            """Calculate comprehensive churn risk score"""
            risk_score = 0.0
            
            # Recency risk (higher recency = higher risk)
            if recency and recency > 90:
                risk_score += 0.3
            elif recency and recency > 60:
                risk_score += 0.2
            elif recency and recency > 30:
                risk_score += 0.1
            
            # Frequency risk (lower frequency = higher risk)
            if frequency and frequency < 0.1:  # Less than once per 10 days
                risk_score += 0.25
            elif frequency and frequency < 0.2:
                risk_score += 0.15
            
            # Monetary risk (declining spend = higher risk)
            if monetary and monetary < 100:  # Low monetary value
                risk_score += 0.2
            
            # Engagement risk
            if engagement and engagement < 0.3:
                risk_score += 0.15
            
            # Support ticket risk (too many or too few)
            if support_tickets:
                if support_tickets > 10:  # Too many issues
                    risk_score += 0.1
                elif support_tickets == 0:  # No engagement
                    risk_score += 0.05
            
            return min(float(risk_score), 1.0)
        
        # Behavioral Trend UDF
        @udf(returnType=StringType())
        def determine_customer_trend(current_value, previous_value, threshold=0.1):
            """Determine if customer behavior is improving, declining, or stable"""
            if not all([current_value, previous_value]) or previous_value == 0:
                return "unknown"
            
            change_rate = (float(current_value) - float(previous_value)) / float(previous_value)
            
            if change_rate > threshold:
                return "improving"
            elif change_rate < -threshold:
                return "declining"
            else:
                return "stable"
        
        # Register UDFs with Spark
        self.spark.udf.register("calculate_clv_score", calculate_clv_score)
        self.spark.udf.register("calculate_risk_score", calculate_risk_score)
        self.spark.udf.register("determine_customer_trend", determine_customer_trend)
        
        # Store UDF references
        self.clv_udf = calculate_clv_score
        self.risk_udf = calculate_risk_score
        self.trend_udf = determine_customer_trend
    
    def create_rfm_features(self, customer_df: DataFrame, transaction_df: DataFrame) -> DataFrame:
        """
        Create comprehensive RFM (Recency, Frequency, Monetary) features
        
        Args:
            customer_df: Customer master data
            transaction_df: Transaction history
            
        Returns:
            DataFrame with RFM features
        """
        logger.info("Creating RFM features...")
        
        current_date = lit(datetime.now().date())
        
        # Calculate RFM metrics for different time windows
        rfm_features = customer_df
        
        for window_days in self.feature_config['recency_windows']:
            window_start = date_sub(current_date, window_days)
            
            # Filter transactions for current window
            window_transactions = transaction_df.filter(
                col("transaction_date") >= window_start
            )
            
            # Aggregate by customer
            window_agg = window_transactions.groupBy("customer_id").agg(
                # Recency: Days since last transaction
                datediff(current_date, max("transaction_date")).alias(f"recency_{window_days}d"),
                
                # Frequency: Number of transactions
                count("*").alias(f"frequency_{window_days}d"),
                
                # Monetary: Total and average transaction values
                sum("transaction_amount").alias(f"monetary_total_{window_days}d"),
                avg("transaction_amount").alias(f"monetary_avg_{window_days}d"),
                
                # Additional metrics
                countDistinct("product_category").alias(f"category_diversity_{window_days}d"),
                stddev("transaction_amount").alias(f"spending_variance_{window_days}d"),
                
                # Purchase patterns
                (count("*") / window_days).alias(f"purchase_frequency_{window_days}d"),
                
                # Monetary trends
                sum(when(col("transaction_date") >= date_sub(current_date, window_days // 2), 
                        col("transaction_amount")).otherwise(0)).alias(f"recent_spending_{window_days}d")
            )
            
            # Join with main customer DataFrame
            rfm_features = rfm_features.join(
                window_agg, "customer_id", "left"
            )
        
        # Calculate RFM scores and segments
        rfm_features = self._calculate_rfm_scores(rfm_features)
        
        return rfm_features
    
    def _calculate_rfm_scores(self, df: DataFrame) -> DataFrame:
        """Calculate RFM scores and customer segments"""
        
        # Define window for percentile calculations
        window_spec = Window.partitionBy()
        
        # Calculate quintiles for primary RFM metrics (30-day window)
        df = df.withColumn("recency_score",
                          6 - ntile(5).over(window_spec.orderBy(col("recency_30d").asc())))
        
        df = df.withColumn("frequency_score", 
                          ntile(5).over(window_spec.orderBy(col("frequency_30d").asc())))
        
        df = df.withColumn("monetary_score",
                          ntile(5).over(window_spec.orderBy(col("monetary_total_30d").asc())))
        
        # Combined RFM score
        df = df.withColumn("rfm_score",
                          col("recency_score") * 100 + 
                          col("frequency_score") * 10 + 
                          col("monetary_score"))
        
        # Customer segments based on RFM scores
        df = df.withColumn("customer_segment",
                          when(col("rfm_score") >= 555, "Champions")
                          .when(col("rfm_score") >= 454, "Loyal Customers")
                          .when(col("rfm_score") >= 344, "Potential Loyalists")
                          .when(col("rfm_score") >= 334, "New Customers")
                          .when(col("rfm_score") >= 313, "Promising")
                          .when(col("rfm_score") >= 244, "Need Attention")
                          .when(col("rfm_score") >= 144, "About to Sleep")
                          .when(col("rfm_score") >= 133, "At Risk")
                          .when(col("rfm_score") >= 123, "Cannot Lose Them")
                          .otherwise("Lost"))
        
        return df
    
    def create_behavioral_features(self, customer_df: DataFrame, 
                                 engagement_df: DataFrame) -> DataFrame:
        """
        Create behavioral and engagement features
        
        Args:
            customer_df: Customer master data
            engagement_df: Customer engagement events
            
        Returns:
            DataFrame with behavioral features
        """
        logger.info("Creating behavioral features...")
        
        current_date = lit(datetime.now().date())
        
        # Time-based engagement windows
        behavioral_features = customer_df
        
        for window_days in [7, 30, 90]:
            window_start = date_sub(current_date, window_days)
            
            # Filter engagement events
            window_engagement = engagement_df.filter(
                col("event_date") >= window_start
            )
            
            # Aggregate engagement metrics
            engagement_agg = window_engagement.groupBy("customer_id").agg(
                # Login behavior
                sum(when(col("event_type") == "login", 1).otherwise(0)).alias(f"login_count_{window_days}d"),
                countDistinct(when(col("event_type") == "login", col("event_date"))).alias(f"active_days_{window_days}d"),
                
                # Feature usage
                sum(when(col("event_type") == "feature_usage", 1).otherwise(0)).alias(f"feature_usage_{window_days}d"),
                countDistinct(when(col("event_type") == "feature_usage", col("feature_name"))).alias(f"features_used_{window_days}d"),
                
                # Support interactions
                sum(when(col("event_type") == "support_ticket", 1).otherwise(0)).alias(f"support_tickets_{window_days}d"),
                sum(when(col("event_type") == "support_resolution", 1).otherwise(0)).alias(f"support_resolved_{window_days}d"),
                
                # Community engagement
                sum(when(col("event_type") == "forum_post", 1).otherwise(0)).alias(f"forum_posts_{window_days}d"),
                sum(when(col("event_type") == "review_submission", 1).otherwise(0)).alias(f"reviews_{window_days}d"),
                
                # Content consumption
                sum(when(col("event_type") == "content_view", col("duration_minutes")).otherwise(0)).alias(f"content_minutes_{window_days}d"),
                countDistinct(when(col("event_type") == "content_view", col("content_id"))).alias(f"unique_content_{window_days}d")
            )
            
            # Join with main DataFrame
            behavioral_features = behavioral_features.join(
                engagement_agg, "customer_id", "left"
            )
        
        # Calculate derived behavioral metrics
        behavioral_features = self._calculate_engagement_scores(behavioral_features)
        
        return behavioral_features
    
    def _calculate_engagement_scores(self, df: DataFrame) -> DataFrame:
        """Calculate comprehensive engagement scores"""
        
        weights = self.feature_config['engagement_weights']
        
        # Normalize engagement metrics (30-day window)
        df = df.fillna(0, subset=[col for col in df.columns if col.endswith('_30d')])
        
        # Calculate individual engagement components
        df = df.withColumn("login_engagement",
                          least(col("login_count_30d") / 30.0, lit(1.0)))
        
        df = df.withColumn("feature_engagement", 
                          least(col("features_used_30d") / 10.0, lit(1.0)))
        
        df = df.withColumn("support_engagement",
                          when(col("support_tickets_30d") > 0,
                               least(col("support_resolved_30d") / col("support_tickets_30d"), lit(1.0)))
                          .otherwise(0.5))  # Neutral if no tickets
        
        df = df.withColumn("community_engagement",
                          least((col("forum_posts_30d") + col("reviews_30d")) / 5.0, lit(1.0)))
        
        df = df.withColumn("content_engagement",
                          least(col("content_minutes_30d") / 300.0, lit(1.0)))  # 5 hours max
        
        # Weighted overall engagement score
        df = df.withColumn("engagement_score",
                          (col("login_engagement") * weights['login_frequency'] +
                           col("feature_engagement") * weights['feature_usage'] +
                           col("support_engagement") * weights['support_interaction'] +
                           col("community_engagement") * weights['community_participation'] +
                           col("content_engagement") * weights['feedback_submission']))
        
        # Engagement trend analysis
        df = df.withColumn("engagement_trend_30_90",
                          self.trend_udf(col("engagement_score"), 
                                       col("engagement_score") * 0.9))  # Approximate previous
        
        return df
    
    def create_advanced_features(self, df: DataFrame) -> DataFrame:
        """
        Create advanced features using custom UDFs and complex calculations
        
        Args:
            df: DataFrame with basic RFM and behavioral features
            
        Returns:
            DataFrame with advanced features
        """
        logger.info("Creating advanced features...")
        
        # Customer Lifetime Value prediction
        df = df.withColumn("predicted_clv",
                          self.clv_udf(col("monetary_total_365d"),
                                     col("monetary_avg_90d"),
                                     col("purchase_frequency_90d"),
                                     col("customer_tenure_months")))
        
        # Comprehensive risk scoring
        df = df.withColumn("churn_risk_score",
                          self.risk_udf(col("recency_30d"),
                                      col("purchase_frequency_30d"),
                                      col("monetary_total_30d"),
                                      col("engagement_score"),
                                      col("support_tickets_30d")))
        
        # Customer health score (opposite of risk)
        df = df.withColumn("customer_health_score", 
                          lit(1.0) - col("churn_risk_score"))
        
        # Seasonality features
        df = self._add_seasonality_features(df)
        
        # Product affinity features
        df = self._add_product_affinity_features(df)
        
        # Demographic features
        df = self._add_demographic_features(df)
        
        return df
    
    def _add_seasonality_features(self, df: DataFrame) -> DataFrame:
        """Add seasonality and time-based features"""
        
        current_date = current_date()
        
        df = df.withColumn("days_since_signup",
                          datediff(current_date, col("signup_date")))
        
        df = df.withColumn("customer_lifecycle_stage",
                          when(col("days_since_signup") <= 30, "new")
                          .when(col("days_since_signup") <= 180, "growing")
                          .when(col("days_since_signup") <= 365, "mature")
                          .otherwise("veteran"))
        
        # Seasonal patterns
        df = df.withColumn("signup_month", month(col("signup_date")))
        df = df.withColumn("signup_quarter", quarter(col("signup_date")))
        
        # Current season impact
        df = df.withColumn("current_month", month(current_date))
        df = df.withColumn("is_holiday_season",
                          when(col("current_month").isin([11, 12, 1]), True)
                          .otherwise(False))
        
        return df
    
    def _add_product_affinity_features(self, df: DataFrame) -> DataFrame:
        """Add product usage and affinity features"""
        
        # Product diversity metrics (already calculated in RFM)
        df = df.withColumn("product_diversity_score",
                          col("category_diversity_90d") / 10.0)  # Normalize to 0-1
        
        # Spending consistency
        df = df.withColumn("spending_consistency",
                          when(col("spending_variance_90d") > 0,
                               1.0 / (1.0 + col("spending_variance_90d") / col("monetary_avg_90d")))
                          .otherwise(1.0))
        
        # Value tier classification
        df = df.withColumn("value_tier",
                          when(col("monetary_total_365d") > 5000, "premium")
                          .when(col("monetary_total_365d") > 1000, "high_value")
                          .when(col("monetary_total_365d") > 200, "medium_value")
                          .otherwise("low_value"))
        
        return df
    
    def _add_demographic_features(self, df: DataFrame) -> DataFrame:
        """Add demographic and profile-based features"""
        
        # Age group features
        df = df.withColumn("age_group",
                          when(col("age") < 25, "young_adult")
                          .when(col("age") < 35, "millennial")
                          .when(col("age") < 50, "gen_x")
                          .otherwise("baby_boomer"))
        
        # Location-based features
        df = df.withColumn("is_urban",
                          when(col("city_type") == "metro", True)
                          .otherwise(False))
        
        # Account features
        df = df.withColumn("account_age_months",
                          months_between(current_date(), col("signup_date")))
        
        df = df.withColumn("has_premium_features",
                          when(col("subscription_tier").isin(["premium", "enterprise"]), True)
                          .otherwise(False))
        
        return df
    
    def create_feature_pipeline(self, customer_df: DataFrame, 
                              transaction_df: DataFrame,
                              engagement_df: DataFrame) -> DataFrame:
        """
        Complete feature engineering pipeline
        
        Args:
            customer_df: Customer master data
            transaction_df: Transaction history
            engagement_df: Engagement events
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting complete feature engineering pipeline...")
        
        # Step 1: Create RFM features
        features_df = self.create_rfm_features(customer_df, transaction_df)
        
        # Step 2: Add behavioral features
        features_df = self.create_behavioral_features(features_df, engagement_df)
        
        # Step 3: Create advanced features
        features_df = self.create_advanced_features(features_df)
        
        # Step 4: Feature selection and final processing
        features_df = self._finalize_features(features_df)
        
        logger.info("Feature engineering pipeline completed successfully")
        return features_df
    
    def _finalize_features(self, df: DataFrame) -> DataFrame:
        """Final feature processing and selection"""
        
        # Fill missing values
        numeric_columns = [field.name for field in df.schema.fields 
                         if isinstance(field.dataType, (IntegerType, LongType, FloatType, DoubleType))]
        
        # Fill numeric columns with 0
        df = df.fillna(0, subset=numeric_columns)
        
        # Fill categorical columns with 'unknown'
        categorical_columns = [field.name for field in df.schema.fields 
                             if isinstance(field.dataType, StringType)]
        df = df.fillna('unknown', subset=categorical_columns)
        
        # Add feature creation timestamp
        df = df.withColumn("feature_creation_timestamp", current_timestamp())
        
        return df
    
    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """
        Get feature importance scores for model interpretation
        
        Args:
            feature_names: List of feature column names
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        # This would typically be calculated after model training
        # For now, return business-logic based importance
        
        importance_map = {}
        
        for feature in feature_names:
            if 'rfm_score' in feature or 'clv' in feature:
                importance_map[feature] = 0.9
            elif 'engagement_score' in feature or 'churn_risk' in feature:
                importance_map[feature] = 0.8
            elif 'frequency' in feature or 'monetary' in feature:
                importance_map[feature] = 0.7
            elif 'recency' in feature:
                importance_map[feature] = 0.6
            else:
                importance_map[feature] = 0.4
        
        return importance_map


# Main execution for testing
if __name__ == "__main__":
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("CustomerRetentionFeatureEngineering") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    # Initialize feature engineer
    feature_engineer = CustomerFeatureEngineer(spark)
    
    # Sample data for testing
    customer_data = [
        ("C001", "John Doe", 35, "metro", "2022-01-15", "premium", 12),
        ("C002", "Jane Smith", 28, "suburban", "2023-06-20", "basic", 6),
        ("C003", "Bob Johnson", 42, "rural", "2021-08-10", "enterprise", 18)
    ]
    
    customer_df = spark.createDataFrame(customer_data, [
        "customer_id", "customer_name", "age", "city_type", 
        "signup_date", "subscription_tier", "customer_tenure_months"
    ])
    
    # Convert string dates to date type
    customer_df = customer_df.withColumn("signup_date", 
                                       to_date(col("signup_date"), "yyyy-MM-dd"))
    
    print("Feature engineering pipeline test completed successfully!")
    
    spark.stop()
