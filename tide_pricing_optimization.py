"""
Tide Pricing Optimization Model with MLflow Integration
========================================================
This script implements a complete ML pipeline for optimizing Tide's pricing strategy
to maximize revenue while maintaining demand levels.

Features:
- Multiple regression models with hyperparameter tuning
- MLflow experiment tracking and model versioning
- Feature engineering for pricing, competition, and inventory
- Business constraint validation
- Model performance comparison and optimization
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================


class Config:
    """Configuration parameters for the pricing optimization model"""

    # Data parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5

    # Business constraints
    MIN_DEMAND_THRESHOLD = 1500  # Minimum acceptable demand level
    MIN_MARGIN = 0.15  # Minimum 15% profit margin
    MAX_PRICE_INCREASE = 0.20  # Maximum 20% price increase from baseline

    # MLflow parameters
    EXPERIMENT_NAME = "Tide_Pricing_Optimization"
    TRACKING_URI = "mlruns"  # Local directory for tracking


# ============================================================================
# DATA PREPROCESSING & FEATURE ENGINEERING
# ============================================================================


def load_and_prepare_data(file_path):
    """
    Load and prepare the dataset with feature engineering

    Args:
        file_path: Path to the CSV data file

    Returns:
        DataFrame with engineered features
    """
    df = pd.read_csv(file_path)

    # Convert date column to datetime - try multiple formats
    try:
        df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%y")
    except ValueError:
        try:
            df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
        except ValueError:
            # If both fail, let pandas infer the format
            df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True)

    df = df.sort_values("Date").reset_index(drop=True)

    return df


def engineer_features(df):
    """
    Create advanced features for pricing optimization

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with engineered features
    """
    df = df.copy()

    # 1. Price-related features
    df["PriceDiscount"] = (df["MRP"] - df["SellingPrice"]) / df["MRP"]
    df["PriceVsNoPromo"] = df["SellingPrice"] / df["NoPromoPrice"]
    df["RelativePrice"] = df["SellingPrice"] / df["MRP"]

    # 2. Competition features
    df["AvgCompetitorPrice"] = (
        df["FinalPrice_Nirma"] + df["FinalPrice_Surf Excel"]
    ) / 2
    df["PriceAdvantage"] = df["AvgCompetitorPrice"] - df["SellingPrice"]
    df["PricePositionVsComp"] = df["SellingPrice"] / df["AvgCompetitorPrice"]
    df["CompetitorDiscountAvg"] = (
        df["DiscountRate_Nirma"] + df["DiscountRate_Surf Excel"]
    ) / 2

    # 3. Inventory features
    df["StockRatio"] = df["StockStart"] / (df["Demand"].replace(0, 1))
    df["BackorderRate"] = df["Backorders"] / (df["Demand"].replace(0, 1))
    df["FulfillmentGap"] = df["Demand"] - df["DemandFulfilled"]
    df["InventoryEfficiency"] = df["InventoryEfficiency"].fillna(
        df["InventoryEfficiency"].median()
    )

    # 4. Customer behavior features
    df["ConversionRate"] = df["CTR"] * (1 - df["AbandonedCartRate"])
    df["OverallFunnelEfficiency"] = (
        df["CTR"] * (1 - df["AbandonedCartRate"]) * (1 - df["BounceRate"])
    )
    df["EngagementScore"] = df["AvgSessionDuration_sec"] * df["CTR"]

    # 5. Temporal features (already present but ensuring they're used)
    df["IsWeekend"] = df["IsWeekend"].astype(int)
    df["Quarter"] = df["Quarter"].astype(int)

    # 6. Lag features (using rolling windows for demand patterns)
    df["Demand_Lag1"] = df["Demand"].shift(1)
    df["Demand_Lag7"] = df["Demand"].shift(7)
    df["UnitsSold_Lag1"] = df["UnitsSold"].shift(1)
    df["UnitsSold_Lag7"] = df["UnitsSold"].shift(7)

    # 7. Revenue feature (target for optimization)
    df["Revenue"] = df["SellingPrice"] * df["UnitsSold"]

    # Fill NaN values created by lag features using forward fill
    df = df.fillna(method="ffill").fillna(method="bfill")

    return df


def select_features(df):
    """
    Select relevant features for modeling

    Args:
        df: DataFrame with all features

    Returns:
        Tuple of (feature_columns, target_column)
    """
    # Target variable: UnitsSold (to predict demand based on pricing)
    target = "UnitsSold"

    # Selected features for prediction
    feature_columns = [
        # Price features
        "SellingPrice",
        "PriceDiscount",
        "RelativePrice",
        "PriceVsNoPromo",
        # Competition features
        "AvgCompetitorPrice",
        "PriceAdvantage",
        "PricePositionVsComp",
        "CompetitorDiscountAvg",
        "CrossPriceElasticity_Nirma_vs_SurfExcel",
        # Inventory features
        "StockStart",
        "StockRatio",
        "BackorderRate",
        "InventoryEfficiency",
        "SafetyStock",
        "ReorderPoint",
        # Customer behavior features
        "CTR",
        "AbandonedCartRate",
        "BounceRate",
        "ConversionRate",
        "OverallFunnelEfficiency",
        "EngagementScore",
        "PurchaseIntent_Score",
        # Temporal features
        "DayOfWeek",
        "Month",
        "Quarter",
        "IsWeekend",
        "DayOfWeek_Sin",
        "DayOfWeek_Cos",
        "Month_Sin",
        "Month_Cos",
        # Historical features
        "Demand_MA7",
        "Demand_Lag1",
        "Demand_Lag7",
        "UnitsSold_Lag1",
        "UnitsSold_Lag7",
    ]

    return feature_columns, target


# ============================================================================
# MODEL TRAINING & HYPERPARAMETER TUNING
# ============================================================================


def train_linear_models(X_train, y_train, X_test, y_test):
    """
    Train and tune linear regression models (Ridge, Lasso, ElasticNet)

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data

    Returns:
        Dictionary of trained models with metrics
    """
    models = {}

    # 1. Ridge Regression
    with mlflow.start_run(run_name="Ridge_Regression"):
        param_grid = {
            "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            "solver": ["auto", "svd", "cholesky"],
        }

        ridge = Ridge(random_state=Config.RANDOM_STATE)
        grid_search = GridSearchCV(
            ridge,
            param_grid,
            cv=Config.CV_FOLDS,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        # Log parameters and metrics
        mlflow.log_params(grid_search.best_params_)
        log_metrics(y_test, y_pred)

        # Log model
        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(best_model, "ridge_model", signature=signature)

        models["Ridge"] = {
            "model": best_model,
            "predictions": y_pred,
            "params": grid_search.best_params_,
        }

    # 2. Lasso Regression
    with mlflow.start_run(run_name="Lasso_Regression"):
        param_grid = {
            "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            "selection": ["cyclic", "random"],
        }

        lasso = Lasso(random_state=Config.RANDOM_STATE, max_iter=10000)
        grid_search = GridSearchCV(
            lasso,
            param_grid,
            cv=Config.CV_FOLDS,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        mlflow.log_params(grid_search.best_params_)
        log_metrics(y_test, y_pred)

        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(best_model, "lasso_model", signature=signature)

        models["Lasso"] = {
            "model": best_model,
            "predictions": y_pred,
            "params": grid_search.best_params_,
        }

    # 3. ElasticNet
    with mlflow.start_run(run_name="ElasticNet_Regression"):
        param_grid = {
            "alpha": [0.01, 0.1, 1.0, 10.0],
            "l1_ratio": [0.2, 0.5, 0.8],
            "selection": ["cyclic", "random"],
        }

        elasticnet = ElasticNet(random_state=Config.RANDOM_STATE, max_iter=10000)
        grid_search = GridSearchCV(
            elasticnet,
            param_grid,
            cv=Config.CV_FOLDS,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        mlflow.log_params(grid_search.best_params_)
        log_metrics(y_test, y_pred)

        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(best_model, "elasticnet_model", signature=signature)

        models["ElasticNet"] = {
            "model": best_model,
            "predictions": y_pred,
            "params": grid_search.best_params_,
        }

    return models


def train_ensemble_models(X_train, y_train, X_test, y_test):
    """
    Train and tune ensemble models (Random Forest, Gradient Boosting)

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data

    Returns:
        Dictionary of trained models with metrics
    """
    models = {}

    # 1. Random Forest
    with mlflow.start_run(run_name="Random_Forest"):
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, 30, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }

        rf = RandomForestRegressor(random_state=Config.RANDOM_STATE, n_jobs=-1)
        grid_search = GridSearchCV(
            rf,
            param_grid,
            cv=Config.CV_FOLDS,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        mlflow.log_params(grid_search.best_params_)
        log_metrics(y_test, y_pred)

        # Log feature importance
        feature_importance = pd.DataFrame(
            {"feature": X_train.columns, "importance": best_model.feature_importances_}
        ).sort_values("importance", ascending=False)
        mlflow.log_dict(feature_importance.to_dict(), "feature_importance.json")

        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(best_model, "random_forest_model", signature=signature)

        models["RandomForest"] = {
            "model": best_model,
            "predictions": y_pred,
            "params": grid_search.best_params_,
            "feature_importance": feature_importance,
        }

    # 2. Gradient Boosting
    with mlflow.start_run(run_name="Gradient_Boosting"):
        param_grid = {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
            "min_samples_split": [2, 5, 10],
            "subsample": [0.8, 0.9, 1.0],
        }

        gb = GradientBoostingRegressor(random_state=Config.RANDOM_STATE)
        grid_search = GridSearchCV(
            gb,
            param_grid,
            cv=Config.CV_FOLDS,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        mlflow.log_params(grid_search.best_params_)
        log_metrics(y_test, y_pred)

        # Log feature importance
        feature_importance = pd.DataFrame(
            {"feature": X_train.columns, "importance": best_model.feature_importances_}
        ).sort_values("importance", ascending=False)
        mlflow.log_dict(feature_importance.to_dict(), "feature_importance.json")

        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(
            best_model, "gradient_boosting_model", signature=signature
        )

        models["GradientBoosting"] = {
            "model": best_model,
            "predictions": y_pred,
            "params": grid_search.best_params_,
            "feature_importance": feature_importance,
        }

    return models


def log_metrics(y_true, y_pred):
    """
    Calculate and log evaluation metrics to MLflow

    Args:
        y_true: Actual values
        y_pred: Predicted values
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mape", mape)

    print(
        f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}, MAPE: {mape:.2f}%"
    )


# ============================================================================
# PRICING OPTIMIZATION
# ============================================================================


def optimize_pricing(model, X_sample, base_price, price_range):
    """
    Find optimal price point that maximizes revenue while maintaining demand

    Args:
        model: Trained ML model
        X_sample: Sample feature vector
        base_price: Current/baseline price
        price_range: Tuple of (min_price, max_price) to search

    Returns:
        Dictionary with optimal price and predicted demand
    """
    prices = np.linspace(price_range[0], price_range[1], 100)
    revenues = []
    demands = []

    for price in prices:
        # Update price in feature vector
        X_test = X_sample.copy()
        X_test["SellingPrice"] = price
        X_test["RelativePrice"] = price / X_test["MRP"]
        X_test["PriceDiscount"] = (X_test["MRP"] - price) / X_test["MRP"]

        # Predict demand
        predicted_demand = model.predict(X_test.values.reshape(1, -1))[0]
        predicted_revenue = price * predicted_demand

        # Apply business constraints
        if predicted_demand >= Config.MIN_DEMAND_THRESHOLD:
            revenues.append(predicted_revenue)
            demands.append(predicted_demand)
        else:
            revenues.append(0)  # Penalize prices that drop demand too low
            demands.append(predicted_demand)

    # Find optimal price
    optimal_idx = np.argmax(revenues)
    optimal_price = prices[optimal_idx]
    optimal_demand = demands[optimal_idx]
    optimal_revenue = revenues[optimal_idx]

    return {
        "optimal_price": optimal_price,
        "predicted_demand": optimal_demand,
        "predicted_revenue": optimal_revenue,
        "price_change": (optimal_price - base_price) / base_price,
        "prices": prices,
        "demands": demands,
        "revenues": revenues,
    }


# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================


def main(data_file_path):
    """
    Main execution pipeline for Tide pricing optimization

    Args:
        data_file_path: Path to input CSV data file
    """
    print("=" * 80)
    print("TIDE PRICING OPTIMIZATION WITH MLFLOW")
    print("=" * 80)

    # Set up MLflow
    mlflow.set_tracking_uri(Config.TRACKING_URI)
    mlflow.set_experiment(Config.EXPERIMENT_NAME)

    # Step 1: Load and prepare data
    print("\n[1/6] Loading and preparing data...")
    df = load_and_prepare_data(data_file_path)
    df = engineer_features(df)
    print(f"Dataset shape: {df.shape}")

    # Step 2: Feature selection
    print("\n[2/6] Selecting features...")
    feature_columns, target = select_features(df)
    X = df[feature_columns]
    y = df[target]
    print(f"Number of features: {len(feature_columns)}")
    print(f"Target variable: {target}")

    # Step 3: Train-test split and scaling
    print("\n[3/6] Splitting data and scaling features...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Step 4: Train linear models
    print("\n[4/6] Training linear regression models...")
    linear_models = train_linear_models(X_train_scaled, y_train, X_test_scaled, y_test)

    # Step 5: Train ensemble models
    print("\n[5/6] Training ensemble models...")
    ensemble_models = train_ensemble_models(
        X_train_scaled, y_train, X_test_scaled, y_test
    )

    # Step 6: Select best model and demonstrate optimization
    print("\n[6/6] Selecting best model and optimizing pricing...")
    all_models = {**linear_models, **ensemble_models}

    # Compare model performance
    model_comparison = []
    for name, model_info in all_models.items():
        r2 = r2_score(y_test, model_info["predictions"])
        rmse = np.sqrt(mean_squared_error(y_test, model_info["predictions"]))
        model_comparison.append({"Model": name, "R² Score": r2, "RMSE": rmse})

    comparison_df = pd.DataFrame(model_comparison).sort_values(
        "R² Score", ascending=False
    )
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(comparison_df.to_string(index=False))

    # Select best model
    best_model_name = comparison_df.iloc[0]["Model"]
    best_model = all_models[best_model_name]["model"]
    print(f"\nBest Model: {best_model_name}")

    # Demonstrate pricing optimization on a sample
    print("\n" + "=" * 80)
    print("PRICING OPTIMIZATION EXAMPLE")
    print("=" * 80)

    sample_idx = X_test.index[0]
    X_sample = X_test.iloc[0]
    current_price = df.loc[sample_idx, "SellingPrice"]
    mrp = df.loc[sample_idx, "MRP"]

    # Define price range (within business constraints)
    min_price = current_price * 0.8
    max_price = min(current_price * (1 + Config.MAX_PRICE_INCREASE), mrp)

    # Scale the sample for prediction
    X_sample_scaled = pd.Series(
        scaler.transform(X_sample.values.reshape(1, -1))[0], index=X_sample.index
    )

    optimization_result = optimize_pricing(
        best_model, X_sample_scaled, current_price, (min_price, max_price)
    )

    print(f"\nCurrent Price: ${current_price:.2f}")
    print(f"Optimal Price: ${optimization_result['optimal_price']:.2f}")
    print(f"Price Change: {optimization_result['price_change']*100:.2f}%")
    print(f"Predicted Demand: {optimization_result['predicted_demand']:.0f} units")
    print(f"Predicted Revenue: ${optimization_result['predicted_revenue']:.2f}")

    # Log optimization results
    with mlflow.start_run(run_name="Pricing_Optimization_Demo"):
        mlflow.log_param("current_price", current_price)
        mlflow.log_param("optimal_price", optimization_result["optimal_price"])
        mlflow.log_metric("price_change_pct", optimization_result["price_change"] * 100)
        mlflow.log_metric("predicted_demand", optimization_result["predicted_demand"])
        mlflow.log_metric("predicted_revenue", optimization_result["predicted_revenue"])

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment: {Config.EXPERIMENT_NAME}")
    print("=" * 80)

    return best_model, scaler, comparison_df


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Replace with your actual data file path
    DATA_FILE = "tide_sales_data.csv"

    # Run the complete pipeline
    best_model, scaler, results = main(DATA_FILE)

    print("\n✓ Model training and optimization complete!")
    print(f"✓ View results in MLflow UI by running: mlflow ui")
