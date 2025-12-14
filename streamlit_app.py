# streamlit_app.py

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
)
from sklearn.calibration import calibration_curve

import statsmodels.api as sm


# --- Global style (simple + consistent)
sns.set_style("whitegrid")


# --- Load raw data (long format) from GitHub (cached)
@st.cache_data
def load_raw_framingham():
    url = (
        "https://raw.githubusercontent.com/"
        "LUCE-Blockchain/Databases-for-teaching/"
        "refs/heads/main/Framingham%20Dataset.csv"
    )
    return pd.read_csv(url)


# --- Load project outputs exported from Colab (cached)
@st.cache_data
def load_project_outputs():
    analytic_df = pd.read_csv("analytic_dataset.csv")
    model_results = pd.read_csv("model_results.csv")
    all_model_outputs = joblib.load("all_model_outputs.pkl")
    return analytic_df, model_results, all_model_outputs


# --- Helper: recompute confusion matrix + metrics for a custom threshold
def compute_threshold_metrics(y_true, y_proba, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba)

    y_pred_thr = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred_thr)
    acc = accuracy_score(y_true, y_pred_thr)

    # AUC uses probabilities, not thresholded labels
    try:
        auc = roc_auc_score(y_true, y_proba)
    except Exception:
        auc = np.nan

    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan

    return {
        "cm": cm,
        "accuracy": acc,
        "roc_auc": auc,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "y_pred_thr": y_pred_thr,
    }


# --- Helper: ROC plot
def plot_roc(y_test, y_proba):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_val = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("False Positive Rate (1 - specificity)")
    ax.set_ylabel("True Positive Rate (sensitivity)")
    ax.set_title("ROC curve")
    ax.legend()
    st.pyplot(fig)


# --- Helper: PR plot
def plot_pr(y_test, y_proba):
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    baseline = float(np.mean(y_test))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"AP = {ap:.3f}")
    ax.axhline(baseline, linestyle="--", color="grey", label=f"Baseline = {baseline:.3f}")
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title("Precision–Recall curve")
    ax.legend()
    st.pyplot(fig)


# --- Helper: calibration plot
def plot_calibration(y_test, y_proba, bins=10):
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=bins, strategy="quantile")
    brier = brier_score_loss(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.set_title(f"Calibration plot (Brier = {brier:.3f})")
    st.pyplot(fig)


# --- Page 1: overview + text
def page_overview(raw_df, analytic_df):
    st.header("Project overview & research question")

    st.markdown(
        """
### Main research question
**Can changes in pulse pressure between Visit 1 and Visit 2 predict the occurrence of a CVD event by Visit 3 in Framingham participants?**

### Subquestions
1. **Does the association between ΔPP and CVD differ for women vs men?** (interaction test)
2. **Does a clinically simple threshold in ΔPP identify higher CVD risk by Visit 3?** (threshold explorer)

### Data & design 
- Framingham dataset is **long format**: multiple rows per participant across **PERIOD 1–3**.
- We built an **analytic dataset (wide + engineered)**:
  - **One row per participant**
  - Baseline covariates from Visit 1 (age, sex, BMI, BP, glucose, cholesterol, smoking)
  - Pulse pressure per visit: **PP = SYSBP − DIABP**
  - ΔPP = **PP₂ − PP₁**
  - Outcome = **CVD at Visit 3** (binary)
"""
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Raw rows (long)", f"{len(raw_df):,}")
    with c2:
        st.metric("Unique participants (raw)", f"{raw_df['RANDID'].nunique():,}")
    with c3:
        st.metric("Analytic rows", f"{len(analytic_df):,}")

    st.markdown("---")

    st.subheader("Quick snapshot of analytic dataset")
    with st.expander("Show first 10 rows"):
        st.dataframe(analytic_df.head(10), use_container_width=True)

    if "CVD" in analytic_df.columns:
        prev = analytic_df["CVD"].mean() * 100
        st.write(f"Analytic CVD prevalence: **{prev:.1f}%**")

    st.markdown("---")
    st.subheader("Methods summary")

    st.markdown(
        """
**Missingness**
- LDLC/HDLC have large missingness due to being measured mainly in Period 3 → treated as structural missingness and excluded.
- Remaining missing values were imputed:
  - numeric: median
  - categorical: mode

**Outliers**
- We used *clinical capping (winsorizing)* instead of dropping rows to avoid losing data.
- Example ranges:
  - SYSBP 80–250, DIABP 40–140, BMI 15–70, TOTCHOL 75–600, GLUCOSE 40–500, CIGPDAY 0–90

**Transforms**
- Log1p transform for skewed variables: GLUCOSE, TOTCHOL, CIGPDAY

**Modeling**
- Outcome: CVD at Visit 3
- Predictors: ΔPP + baseline Visit 1 covariates
- Evaluation: train/test split + ROC AUC as main metric
"""
    )


# --- Page 2: raw EDA
def page_eda_raw(raw_df):
    st.header("Exploratory data analysis – raw Framingham dataset (long format)")

    st.markdown(
        """
This page is about the **raw data** (multiple rows per participant).
We look at:
- How visits (PERIOD) are distributed
- Missingness patterns (overall + by PERIOD)
- Example patient trajectories across visits
"""
    )

    # Basic info
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Dataset info**")
        st.write(f"Rows × columns: `{raw_df.shape[0]:,}` × `{raw_df.shape[1]}`")
        st.write(f"Unique participants: `{raw_df['RANDID'].nunique():,}`")

    with c2:
        st.write("**PERIOD distribution**")
        period_counts = raw_df["PERIOD"].value_counts().sort_index()
        st.write(period_counts)

        fig, ax = plt.subplots(figsize=(4, 3))
        sns.barplot(x=period_counts.index.astype(str), y=period_counts.values, ax=ax)
        ax.set_xlabel("PERIOD")
        ax.set_ylabel("Row count")
        ax.set_title("Rows per PERIOD")
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")

    # Missingness overview
    st.subheader("Missingness overview (raw)")
    missing_pct = (raw_df.isna().mean() * 100).round(2)
    missing_nonzero = missing_pct[missing_pct > 0].sort_values(ascending=False)

    st.write("Top variables by % missing:")
    st.dataframe(missing_nonzero.head(15).to_frame("% missing"), use_container_width=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    top = missing_nonzero.head(15)
    sns.barplot(x=top.values, y=top.index, ax=ax)
    ax.set_xlabel("% missing")
    ax.set_ylabel("")
    ax.set_title("Missingness (top 15)")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    # Missingness by period heatmap (interactive column select)
    st.subheader("Missingness by PERIOD (selected variables)")
    default_cols = ["LDLC", "HDLC", "GLUCOSE", "BPMEDS", "TOTCHOL", "educ", "CIGPDAY", "BMI", "HEARTRTE"]
    available_defaults = [c for c in default_cols if c in raw_df.columns]

    selected_cols = st.multiselect(
        "Variables to include",
        options=sorted(raw_df.columns),
        default=available_defaults,
    )

    if selected_cols:
        miss_by_period = (
            raw_df.groupby("PERIOD")[selected_cols]
            .apply(lambda d: d.isna().mean() * 100)
            .round(1)
        )
        st.dataframe(miss_by_period, use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(miss_by_period.T, annot=True, fmt=".1f", ax=ax)
        ax.set_xlabel("PERIOD")
        ax.set_ylabel("Variable")
        ax.set_title("Missingness by PERIOD (%)")
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")

    # Patient trajectories
    st.subheader("Individual patient trajectories")
    ids = sorted(raw_df["RANDID"].unique())
    selected_id = st.selectbox("Select RANDID", options=ids)

    user_data = raw_df[raw_df["RANDID"] == selected_id].sort_values("PERIOD")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # BP
    axes[0].plot(user_data["PERIOD"], user_data["SYSBP"], marker="o", label="SYSBP")
    axes[0].plot(user_data["PERIOD"], user_data["DIABP"], marker="o", label="DIABP")
    axes[0].set_title("Blood pressure over visits")
    axes[0].set_xlabel("PERIOD")
    axes[0].set_ylabel("mmHg")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # BMI
    if "BMI" in user_data.columns:
        axes[1].plot(user_data["PERIOD"], user_data["BMI"], marker="s", linestyle="--")
        axes[1].set_title("BMI over visits")
        axes[1].set_xlabel("PERIOD")
        axes[1].set_ylabel("BMI (kg/m²)")
        axes[1].grid(True, linestyle="--", alpha=0.4)
    else:
        axes[1].axis("off")

    # Cholesterol
    if "TOTCHOL" in user_data.columns:
        axes[2].plot(user_data["PERIOD"], user_data["TOTCHOL"], marker="^")
        axes[2].set_title("Total cholesterol over visits")
        axes[2].set_xlabel("PERIOD")
        axes[2].set_ylabel("mg/dL")
        axes[2].grid(True, linestyle="--", alpha=0.4)
    else:
        axes[2].axis("off")

    has_cvd = "YES" if user_data["CVD"].max() == 1 else "NO"
    fig.suptitle(f"Trajectory for RANDID {selected_id} (CVD by Visit 3: {has_cvd})")
    plt.tight_layout()
    st.pyplot(fig)


# --- Page 3: analytic / ΔPP exploration
def page_delta_pp(analytic_df):
    st.header("ΔPP & analytic dataset exploration")

    if "DELTA_PP" not in analytic_df.columns:
        st.error("Analytic dataset missing DELTA_PP.")
        return

    st.markdown(
        """
The **change in pulse pressure** over time (ΔPP) can be a *predictor* of future CVD events. A significant increase or decrease in PP between visits could indicate changes in cardiovascular health status. The focus is on the change between visit 1 and visit 2 to predict CVD outcomes at visit 3.

The final analytic dataset represents individuals with:
* Complete BP data at Visits 1 & 2 (to compute ΔPP)
* Observed CVD outcome by Visit 3
"""
    )

    c1, c2 = st.columns(2)
    with c1:
        st.write(f"Rows × columns: `{analytic_df.shape[0]:,}` × `{analytic_df.shape[1]}`")
    with c2:
        if "CVD" in analytic_df.columns:
            prev = analytic_df["CVD"].mean() * 100
            st.write(f"CVD prevalence: **{prev:.1f}%**")

    st.markdown("---")
    st.subheader("ΔPP distribution")
    st.write(analytic_df["DELTA_PP"].describe().round(2))

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(analytic_df["DELTA_PP"], bins=30, kde=True, ax=ax)
    ax.set_xlabel("ΔPP (PP₂ − PP₁, mmHg)")
    ax.set_title("Distribution of ΔPP")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("ΔPP vs CVD")
    if "CVD" in analytic_df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=analytic_df, x="CVD", y="DELTA_PP", ax=ax)
        ax.set_xlabel("CVD at Visit 3 (0=no, 1=yes)")
        ax.set_ylabel("ΔPP (mmHg)")
        ax.set_title("ΔPP by CVD outcome")
        plt.tight_layout()
        st.pyplot(fig)

    if "V1_SEX" in analytic_df.columns and "CVD" in analytic_df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=analytic_df, x="V1_SEX", y="DELTA_PP", hue="CVD", ax=ax)
        ax.set_xlabel("Sex at Visit 1 (0=male, 1=female)")
        ax.set_ylabel("ΔPP (mmHg)")
        ax.set_title("ΔPP by sex and CVD")
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Correlations (baseline + ΔPP)")
    corr_vars = [c for c in [
        "DELTA_PP", "V1_AGE", "V1_BMI", "V1_SYSBP", "V1_DIABP",
        "V1_GLUCOSE", "V1_TOTCHOL", "V1_CIGPDAY"
    ] if c in analytic_df.columns]

    if len(corr_vars) >= 2:
        corr = analytic_df[corr_vars].corr().round(2)
        st.dataframe(corr, use_container_width=True)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, annot=True, center=0, ax=ax)
        ax.set_title("Correlation matrix")
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Interactive scatter")
    numeric_cols = analytic_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        c1, c2, c3 = st.columns(3)
        with c1:
            x_var = st.selectbox("X-axis", options=sorted(numeric_cols))
        with c2:
            y_var = st.selectbox("Y-axis", options=sorted(numeric_cols), index=1)
        with c3:
            color_by = st.selectbox("Color by", options=["None"] + analytic_df.columns.tolist(),
                                    index=(analytic_df.columns.tolist().index("CVD") + 1) if "CVD" in analytic_df.columns else 0)

        fig, ax = plt.subplots(figsize=(7, 5))
        if color_by != "None":
            sns.scatterplot(data=analytic_df, x=x_var, y=y_var, hue=color_by, alpha=0.6, ax=ax)
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        else:
            sns.scatterplot(data=analytic_df, x=x_var, y=y_var, alpha=0.6, ax=ax)

        ax.set_title(f"{x_var} vs {y_var}")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)


# --- Page 4: model comparison
def page_models_overview(model_results):
    st.header("Model comparison")

    if model_results is None or model_results.empty:
        st.error("model_results.csv not loaded or empty.")
        return

    if "ROC_AUC" in model_results.columns:
        model_results_sorted = model_results.sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
    else:
        model_results_sorted = model_results.copy()

    st.subheader("Model comparison table (sorted by test ROC AUC)")
    st.dataframe(model_results_sorted, use_container_width=True)

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Accuracy (test)")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=model_results_sorted, x="Accuracy", y="Model", ax=ax)
        ax.set_xlim(0.5, 1.0)
        plt.tight_layout()
        st.pyplot(fig)

    with c2:
        st.subheader("ROC AUC (test)")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=model_results_sorted, x="ROC_AUC", y="Model", ax=ax)
        ax.set_xlim(0.45, 0.85)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")

    if "Type" in model_results_sorted.columns:
        st.subheader("Accuracy vs ROC AUC (by model type)")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(data=model_results_sorted, x="ROC_AUC", y="Accuracy", hue="Type", style="Type", s=90, ax=ax)
        ax.set_xlabel("ROC AUC")
        ax.set_ylabel("Accuracy")
        ax.set_title("Trade-off: accuracy vs discrimination")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)


# --- Page 5: model detail + threshold + plots
def page_model_detail(all_model_outputs):
    st.header("Model detail + threshold explorer")

    model_name = st.selectbox("Select model", list(all_model_outputs.keys()))
    res = all_model_outputs[model_name]

    y_test = np.asarray(res["y_test"]).astype(int)
    y_proba = np.asarray(res["y_proba"])

    st.subheader("Base performance (stored)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Accuracy (0.5)", f"{res.get('accuracy', np.nan):.3f}")
    with c2:
        st.metric("ROC AUC", f"{res.get('roc_auc', np.nan):.3f}")
    with c3:
        cv = res.get("cv_best_auc", np.nan)
        st.metric("CV Best AUC", f"{cv:.3f}" if pd.notna(cv) else "–")

    st.markdown("---")

    st.subheader("Threshold explorer")
    thr = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)
    m = compute_threshold_metrics(y_test, y_proba, thr)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Accuracy", f"{m['accuracy']:.3f}")
    with c2:
        st.metric("Sensitivity", f"{m['sensitivity']:.3f}" if pd.notna(m["sensitivity"]) else "–")
    with c3:
        st.metric("Specificity", f"{m['specificity']:.3f}" if pd.notna(m["specificity"]) else "–")
    with c4:
        st.metric("Precision", f"{m['precision']:.3f}" if pd.notna(m["precision"]) else "–")

    st.write("Confusion matrix (at chosen threshold):")
    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=m["cm"])
    disp.plot(ax=ax, values_format="d")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("ROC / PR / Calibration (test set)")

    c1, c2 = st.columns(2)
    with c1:
        st.write("ROC curve")
        plot_roc(y_test, y_proba)
    with c2:
        st.write("Precision–Recall curve")
        plot_pr(y_test, y_proba)

    st.markdown("---")
    bins = st.slider("Calibration bins", 5, 20, 10)
    plot_calibration(y_test, y_proba, bins=bins)


# --- Page 6: interaction test (fixed robust SE call)
def page_interaction_test(analytic_df):
    st.header("Subquestion 1: Interaction test (ΔPP × sex)")

    st.markdown(
        """
We test whether the association between ΔPP and CVD differs by sex
using a logistic regression interaction model.

**Model form**
`logit(P(CVD = 1)) = β0 + β1·ΔPP + β2·Sex + β3·(ΔPP × Sex) + controls`

The interaction term **β3** tests whether the ΔPP slope differs between women and men.

This is an **association analysis**, not a causal claim.
"""
    )

    required = ["CVD", "DELTA_PP", "V1_SEX"]
    controls = ["V1_AGE", "V1_BMI", "V1_SYSBP", "V1_DIABP", "V1_GLUCOSE", "V1_TOTCHOL", "V1_CIGPDAY"]
    controls = [c for c in controls if c in analytic_df.columns]

    missing = [c for c in required if c not in analytic_df.columns]
    if missing:
        st.error(f"Analytic dataset missing required columns: {missing}")
        return

    df = analytic_df[required + controls].dropna().copy()
    df["interaction"] = df["DELTA_PP"] * df["V1_SEX"]

    X = df[["DELTA_PP", "V1_SEX", "interaction"] + controls]
    X = sm.add_constant(X, has_constant="add")
    y = df["CVD"].astype(int)

    try:
        # Robust SE specified at fit time (avoids version issues)
        model = sm.Logit(y, X).fit(cov_type="HC3", disp=False)

        out = pd.DataFrame({
            "coef (log-odds)": model.params,
            "OR": np.exp(model.params),
            "p-value": model.pvalues,
        }).round(4)

        st.subheader("Interaction model results")
        st.dataframe(out, use_container_width=True)

        p_int = out.loc["interaction", "p-value"]
        st.markdown(
            f"""
**Interpretation**
- Interaction p-value: **{p_int:.4f}**
- If p < 0.05 → evidence that ΔPP–CVD association differs by sex
- If p ≥ 0.05 → no statistical evidence for sex effect modification
"""
        )

    except Exception as e:
        st.error(f"Interaction model failed to fit: {e}")
        st.info("Common reasons: perfect separation, collinearity, or too few events after dropping missing rows.")


# --- Page 7: final recap
def page_final_recap(model_results):
    st.header("Final research question recap")

    st.markdown(
        """
### Main research question
**Can changes in pulse pressure between Visit 1 and Visit 2 predict CVD by Visit 3?**

ΔPP contributes some predictive signal, but overall performance is moderate.
A multivariable approach is more realistic than using ΔPP alone.

### Subquestion 1 (sex differences)
We used an interaction term (ΔPP × sex). If non-significant, it suggests
no evidence that the ΔPP slope differs between women and men in our model.

### Subquestion 2 (clinical threshold)
Threshold exploration shows the typical trade-off:
- lower threshold → higher sensitivity, lower precision
- higher threshold → higher precision, lower sensitivity

### Overall conclusion
ΔPP has clinical meaning as a longitudinal vascular marker, but its practical value
is in combination with baseline covariates inside a risk model (not as a simple cut-off).
"""
    )

    if model_results is not None and not model_results.empty and "ROC_AUC" in model_results.columns:
        best = model_results.sort_values("ROC_AUC", ascending=False).iloc[0]
        st.markdown("---")
        st.subheader("Best model (based on test ROC AUC)")
        st.write(best)


# --- App entry point
st.set_page_config(page_title="Framingham ΔPP & CVD", layout="wide")

raw_df = load_raw_framingham()
analytic_df, model_results, all_model_outputs = load_project_outputs()

page = st.sidebar.radio(
    "Navigation",
    [
        "1) Overview",
        "2) Raw EDA (long data)",
        "3) ΔPP & analytic exploration",
        "4) Model comparison",
        "5) Model detail + threshold",
        "6) Interaction test (ΔPP × sex)",
        "7) Final RQ recap",
    ],
)

if page == "1) Overview":
    page_overview(raw_df, analytic_df)
elif page == "2) Raw EDA (long data)":
    page_eda_raw(raw_df)
elif page == "3) ΔPP & analytic exploration":
    page_delta_pp(analytic_df)
elif page == "4) Model comparison":
    page_models_overview(model_results)
elif page == "5) Model detail + threshold":
    page_model_detail(all_model_outputs)
elif page == "6) Interaction test (ΔPP × sex)":
    page_interaction_test(analytic_df)
elif page == "7) Final RQ recap":
    page_final_recap(model_results)
