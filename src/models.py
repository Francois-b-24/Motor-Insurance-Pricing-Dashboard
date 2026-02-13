"""
Modeling module: GLM (Poisson frequency + Gamma severity) and XGBoost.
Pure Premium = Frequency × Severity
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from scipy import stats
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore")


def gini_coefficient(y_true, y_pred, weights=None):
    """Compute the normalized Gini coefficient."""
    if weights is None:
        weights = np.ones(len(y_true))

    order = np.argsort(y_pred)
    y_true_sorted = y_true[order]
    w_sorted = weights[order]

    cum_w = np.cumsum(w_sorted)
    cum_loss = np.cumsum(y_true_sorted * w_sorted)

    total_w = cum_w[-1]
    total_loss = cum_loss[-1]

    lorenz = cum_loss / total_loss
    cum_pop = cum_w / total_w
    gini_raw = np.sum((lorenz[:-1] + lorenz[1:]) * np.diff(cum_pop)) - 1

    order_perfect = np.argsort(y_true)
    y_perfect_sorted = y_true[order_perfect]
    w_perfect_sorted = weights[order_perfect]

    cum_w_p = np.cumsum(w_perfect_sorted)
    cum_loss_p = np.cumsum(y_perfect_sorted * w_perfect_sorted)
    lorenz_p = cum_loss_p / total_loss
    cum_pop_p = cum_w_p / total_w
    gini_perfect = np.sum((lorenz_p[:-1] + lorenz_p[1:]) * np.diff(cum_pop_p)) - 1

    if gini_perfect == 0:
        return 0.0
    return gini_raw / gini_perfect


def compute_lift_curve(y_true, y_pred, weights=None, n_bins=10):
    """Compute lift curve data: actual vs predicted by decile."""
    if weights is None:
        weights = np.ones(len(y_true))

    df_lift = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "w": weights})
    df_lift["decile"] = pd.qcut(df_lift["y_pred"], q=n_bins, labels=False,
                                 duplicates="drop")

    lift = df_lift.groupby("decile", observed=True)[["y_pred", "y_true", "w"]].apply(
        lambda g: pd.Series({
            "avg_predicted": np.average(g["y_pred"], weights=g["w"]),
            "avg_actual": np.average(g["y_true"], weights=g["w"]),
            "exposure": g["w"].sum(),
            "count": len(g)
        })
    ).reset_index()

    lift = lift.sort_values("avg_predicted")
    lift["decile_label"] = [f"D{i+1}" for i in range(len(lift))]
    return lift


def train_glm_frequency(X_train, y_train, w_train, X_test):
    """Train a Poisson GLM for claim frequency."""
    X_train_c = sm.add_constant(X_train, has_constant="add")
    X_test_c = sm.add_constant(X_test, has_constant="add")

    glm_model = sm.GLM(
        y_train, X_train_c,
        family=sm.families.Poisson(link=sm.families.links.Log()),
        freq_weights=w_train
    )
    glm_results = glm_model.fit(maxiter=100, method="IRLS")

    y_pred_train = glm_results.predict(X_train_c)
    y_pred_test = glm_results.predict(X_test_c)

    return glm_results, y_pred_train, y_pred_test


def train_glm_severity(X_train, y_train, w_train, X_test):
    """
    Train a Gamma GLM (log link) for claim severity.
    Only fitted on policies WITH claims (severity > 0).
    """
    X_train_c = sm.add_constant(X_train, has_constant="add")
    X_test_c = sm.add_constant(X_test, has_constant="add")

    glm_model = sm.GLM(
        y_train, X_train_c,
        family=sm.families.Gamma(link=sm.families.links.Log()),
        freq_weights=w_train
    )
    glm_results = glm_model.fit(maxiter=100, method="IRLS")

    y_pred_train = glm_results.predict(X_train_c)
    y_pred_test = glm_results.predict(X_test_c)

    return glm_results, y_pred_train, y_pred_test


def train_xgboost(X_train, y_train, w_train, X_test):
    """Train an XGBoost model for frequency modeling."""
    xgb_model = XGBRegressor(
        objective="count:poisson",
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=50,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=-1
    )

    xgb_model.fit(X_train, y_train, sample_weight=w_train, verbose=False)

    y_pred_train = np.clip(xgb_model.predict(X_train), 0, None)
    y_pred_test = np.clip(xgb_model.predict(X_test), 0, None)

    return xgb_model, y_pred_train, y_pred_test


def compute_metrics(y_true, y_pred, weights=None):
    """Compute evaluation metrics."""
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=weights)),
        "MAE": mean_absolute_error(y_true, y_pred, sample_weight=weights),
        "Gini": gini_coefficient(y_true, y_pred, weights),
        "Mean Predicted": np.average(y_pred, weights=weights),
        "Mean Actual": np.average(y_true, weights=weights),
    }


def compute_glm_diagnostics(glm_results, y_true, y_pred, weights=None):
    """
    Compute statistical diagnostics for GLM models.
    Returns dictionary with test statistics and diagnostics.
    """
    try:
        if glm_results is None or y_true is None or y_pred is None:
            return {}

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if len(y_true) == 0 or len(y_pred) == 0:
            return {}

        if weights is None:
            weights = np.ones(len(y_true))
        else:
            weights = np.asarray(weights)

        diagnostics = {}

        # Basic model statistics
        diagnostics["deviance"] = glm_results.deviance
        diagnostics["null_deviance"] = glm_results.null_deviance
        diagnostics["df_resid"] = glm_results.df_resid
        diagnostics["df_model"] = glm_results.df_model
        diagnostics["aic"] = glm_results.aic
        diagnostics["bic"] = glm_results.bic

        # Pseudo R-squared
        if glm_results.null_deviance > 0:
            diagnostics["pseudo_r2"] = 1 - (glm_results.deviance / glm_results.null_deviance)
        else:
            diagnostics["pseudo_r2"] = 0.0

        # Pearson Chi-square test
        y_pred_safe = np.clip(y_pred, 1e-10, None)
        pearson_residuals = (y_true - y_pred) / np.sqrt(y_pred_safe)
        pearson_residuals = np.nan_to_num(pearson_residuals, nan=0.0, posinf=0.0, neginf=0.0)
        pearson_chi2 = np.sum(weights * pearson_residuals**2)
        diagnostics["pearson_chi2"] = pearson_chi2
        if glm_results.df_resid > 0:
            diagnostics["pearson_chi2_pvalue"] = max(0, min(1, 1 - stats.chi2.cdf(
                pearson_chi2, glm_results.df_resid
            )))
        else:
            diagnostics["pearson_chi2_pvalue"] = 1.0

        # Overdispersion test (for Poisson)
        if hasattr(glm_results, 'family') and 'Poisson' in str(glm_results.family):
            if glm_results.df_resid > 0:
                dispersion_ratio = pearson_chi2 / glm_results.df_resid
                diagnostics["dispersion_ratio"] = dispersion_ratio
                diagnostics["overdispersed"] = dispersion_ratio > 1.5
            else:
                diagnostics["dispersion_ratio"] = 1.0
                diagnostics["overdispersed"] = False

            dean_denom = np.sqrt(2 * np.sum(weights * y_pred_safe**2))
            if dean_denom > 0:
                dean_stat = np.sum(weights * ((y_true - y_pred)**2 - y_true)) / dean_denom
                diagnostics["dean_statistic"] = dean_stat
                diagnostics["dean_pvalue"] = max(0, min(1, 2 * (1 - stats.norm.cdf(abs(dean_stat)))))
            else:
                diagnostics["dean_statistic"] = 0.0
                diagnostics["dean_pvalue"] = 1.0

        # Likelihood Ratio Test (model vs null)
        lr_stat = glm_results.null_deviance - glm_results.deviance
        diagnostics["lr_statistic"] = lr_stat
        if glm_results.df_model > 0:
            diagnostics["lr_pvalue"] = max(0, min(1, 1 - stats.chi2.cdf(
                lr_stat, glm_results.df_model
            )))
        else:
            diagnostics["lr_pvalue"] = 1.0

        # Residuals — limited sample for visualization
        max_residuals = 10000
        if len(pearson_residuals) > max_residuals:
            diagnostics["pearson_residuals"] = pearson_residuals[:max_residuals]
        else:
            diagnostics["pearson_residuals"] = pearson_residuals

        if hasattr(glm_results, 'resid_deviance'):
            dev_res = glm_results.resid_deviance
            dev_res = np.nan_to_num(dev_res, nan=0.0, posinf=0.0, neginf=0.0)
            if len(dev_res) > max_residuals:
                diagnostics["deviance_residuals"] = dev_res[:max_residuals]
            else:
                diagnostics["deviance_residuals"] = dev_res
        else:
            diagnostics["deviance_residuals"] = np.array([])

        return diagnostics
    except Exception:
        return {}


def compute_severity_analysis(severity_values, claim_counts=None):
    """
    Analyze severity distribution: VaR, TVaR, distribution tests.
    """
    try:
        if severity_values is None or len(severity_values) == 0:
            return {}
        
        if claim_counts is None:
            claim_counts = np.ones(len(severity_values))
        
        # Ensure arrays are numpy arrays
        severity_values = np.asarray(severity_values)
        claim_counts = np.asarray(claim_counts)
        
        # Filter non-zero severities
        mask = severity_values > 0
        if not mask.any():
            return {}
            
        sev_nonzero = severity_values[mask]
        if len(claim_counts) == len(severity_values):
            counts_nonzero = claim_counts[mask]
        else:
            counts_nonzero = np.ones(len(sev_nonzero))
        
        if len(sev_nonzero) == 0:
            return {}
        
        analysis = {}
        
        # Basic statistics
        analysis["mean"] = np.average(sev_nonzero, weights=counts_nonzero)
        analysis["median"] = np.median(sev_nonzero)
        analysis["std"] = np.sqrt(np.average((sev_nonzero - analysis["mean"])**2, weights=counts_nonzero))
        analysis["cv"] = analysis["std"] / analysis["mean"] if analysis["mean"] > 0 else 0
        
        # Percentiles
        percentiles = [50, 75, 90, 95, 99, 99.5, 99.9]
        for p in percentiles:
            analysis[f"p{p}"] = np.percentile(sev_nonzero, p)
        
        # VaR and TVaR
        analysis["var_95"] = np.percentile(sev_nonzero, 95)
        analysis["var_99"] = np.percentile(sev_nonzero, 99)
        
        # TVaR (Conditional Tail Expectation)
        analysis["tvar_95"] = sev_nonzero[sev_nonzero >= analysis["var_95"]].mean() if len(sev_nonzero[sev_nonzero >= analysis["var_95"]]) > 0 else analysis["var_95"]
        analysis["tvar_99"] = sev_nonzero[sev_nonzero >= analysis["var_99"]].mean() if len(sev_nonzero[sev_nonzero >= analysis["var_99"]]) > 0 else analysis["var_99"]
        
        # Skewness and Kurtosis
        try:
            analysis["skewness"] = stats.skew(sev_nonzero)
            analysis["kurtosis"] = stats.kurtosis(sev_nonzero)
        except Exception:
            analysis["skewness"] = 0.0
            analysis["kurtosis"] = 0.0
        
        # Distribution tests (log-normal vs gamma)
        try:
            log_sev = np.log(sev_nonzero[sev_nonzero > 0])
            if len(log_sev) > 0:
                analysis["log_mean"] = np.mean(log_sev)
                analysis["log_std"] = np.std(log_sev)
            else:
                analysis["log_mean"] = 0.0
                analysis["log_std"] = 0.0
        except Exception:
            analysis["log_mean"] = 0.0
            analysis["log_std"] = 0.0
        
        return analysis
    except Exception:
        return {}


def run_models(X, y, w, claim_count, df_model=None):
    """
    Full modeling pipeline:
    - Frequency: Poisson GLM + XGBoost
    - Severity: Gamma GLM (on claims only)
    - Pure Premium = Frequency × Severity
    """
    # ── Train/test split ──────────────────────────────────────────────
    if df_model is not None:
        severity = df_model["Severity"].values
        total_amount = df_model["TotalClaimAmount"].values
        pure_premium = df_model["PurePremium"].values
    else:
        severity = np.zeros(len(y))
        total_amount = np.zeros(len(y))
        pure_premium = np.zeros(len(y))

    (X_train, X_test,
     y_train, y_test,
     w_train, w_test,
     cc_train, cc_test,
     sev_train, sev_test,
     amt_train, amt_test,
     pp_train, pp_test) = train_test_split(
        X, y, w, claim_count, severity, total_amount, pure_premium,
        test_size=0.25, random_state=42
    )

    # ── FREQUENCY MODELS ──────────────────────────────────────────────
    glm_freq_results, glm_freq_pred_train, glm_freq_pred_test = \
        train_glm_frequency(X_train, y_train, w_train, X_test)

    xgb_model, xgb_pred_train, xgb_pred_test = \
        train_xgboost(X_train, y_train, w_train, X_test)

    # ── SEVERITY MODEL (Gamma GLM on claims only) ────────────────────
    claims_mask_train = (cc_train > 0) & (sev_train > 0)
    claims_mask_test = (cc_test > 0) & (sev_test > 0)

    glm_sev_results = None
    glm_sev_pred_all_test = None
    avg_severity = np.mean(sev_train[claims_mask_train]) if claims_mask_train.any() else 1500.0

    if claims_mask_train.sum() > 100:
        try:
            glm_sev_results, glm_sev_pred_train_claims, glm_sev_pred_test_claims = \
                train_glm_severity(
                    X_train[claims_mask_train],
                    sev_train[claims_mask_train],
                    cc_train[claims_mask_train],  # weight by nb claims
                    X_test[claims_mask_test] if claims_mask_test.any() else X_test[:1]
                )

            # Predict severity for ALL test observations (needed for pure premium)
            X_test_c = sm.add_constant(X_test, has_constant="add")
            glm_sev_pred_all_test = glm_sev_results.predict(X_test_c)

        except Exception:
            glm_sev_results = None
            glm_sev_pred_all_test = np.full(len(X_test), avg_severity)
    else:
        glm_sev_pred_all_test = np.full(len(X_test), avg_severity)

    # ── PURE PREMIUM = Frequency × Severity ───────────────────────────
    glm_pp_pred_test = glm_freq_pred_test * glm_sev_pred_all_test
    xgb_pp_pred_test = xgb_pred_test * glm_sev_pred_all_test  # XGB freq × GLM sev

    # ── METRICS ───────────────────────────────────────────────────────
    glm_metrics_test = compute_metrics(y_test, glm_freq_pred_test, w_test)
    xgb_metrics_test = compute_metrics(y_test, xgb_pred_test, w_test)
    glm_metrics_train = compute_metrics(y_train, glm_freq_pred_train, w_train)
    xgb_metrics_train = compute_metrics(y_train, xgb_pred_train, w_train)

    # Pure premium metrics
    pp_metrics_glm = compute_metrics(pp_test, glm_pp_pred_test, w_test)
    pp_metrics_xgb = compute_metrics(pp_test, xgb_pp_pred_test, w_test)

    # Lift curves
    glm_lift = compute_lift_curve(y_test, glm_freq_pred_test, w_test)
    xgb_lift = compute_lift_curve(y_test, xgb_pred_test, w_test)
    
    # GLM Diagnostics
    try:
        glm_diagnostics = compute_glm_diagnostics(glm_freq_results, y_train, glm_freq_pred_train, w_train)
    except Exception as e:
        # If diagnostics fail, return empty dict to avoid breaking the app
        glm_diagnostics = {}
    
    # Severity analysis
    try:
        severity_analysis = compute_severity_analysis(sev_train, cc_train) if len(sev_train) > 0 else {}
    except Exception as e:
        severity_analysis = {}

    return {
        # Frequency models
        "glm_results": glm_freq_results,
        "xgb_model": xgb_model,
        # Severity model
        "glm_sev_results": glm_sev_results,
        "avg_severity": avg_severity,
        # Pure premium metrics
        "pp_metrics_glm": pp_metrics_glm,
        "pp_metrics_xgb": pp_metrics_xgb,
        # Data splits (only what the app needs)
        "X_test": X_test,
        "y_test": y_test,
        "w_test": w_test,
        "sev_train": sev_train,
        "cc_train": cc_train,
        # Frequency predictions (test only)
        "glm_pred_test": glm_freq_pred_test,
        "xgb_pred_test": xgb_pred_test,
        # Metrics
        "glm_metrics_test": glm_metrics_test,
        "xgb_metrics_test": xgb_metrics_test,
        # Lift curves
        "glm_lift": glm_lift,
        "xgb_lift": xgb_lift,
        # Diagnostics
        "glm_diagnostics": glm_diagnostics,
        "severity_analysis": severity_analysis,
    }
