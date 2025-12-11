import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    accuracy_score,
    classification_report,
)

import streamlit as st

# ============================================================
# Global style & colour scheme
# ============================================================

sns.set_style("whitegrid")

# Continuous "crest" colormap to keep everything visually consistent
crest_colors = sns.color_palette("crest", as_cmap=False)
crest_cmap = LinearSegmentedColormap.from_list("crest_cmap", crest_colors, N=256)


# ============================================================
# Data loading (cached)
# ============================================================

@st.cache_data
def load_raw_framingham():
    """Load original long Framingham dataset from GitHub."""
    url = "https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/refs/heads/main/Framingham%20Dataset.csv"
    df = pd.read_csv(url)
    return df


@st.cache_data
def load_project_outputs():
    """
    Load analytic dataset, model summary table, and detailed model outputs
    that were exported from the Colab notebook.
    """
    analytic = pd.read_csv("analytic_dataset.csv")
    model_results = pd.read_csv("model_results.csv")
    all_model_outputs = joblib.load("all_model_outputs.pkl")
    return analytic, model_results, all_model_outputs


# ============================================================
# Helper: metrics at custom threshold
# ============================================================

def compute_threshold_metrics(y_true, y_proba, threshold: float):
    """Recompute confusion matrix & derived metrics for a custom decision threshold."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    y_pred_thr = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred_thr)
    acc = accuracy_score(y_true, y_pred_thr)

    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = np.nan

    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        "cm": cm,
        "accuracy": acc,
        "roc_auc": auc,
        "sensitivity": sens,
        "specificity": spec,
    }


# ============================================================
# PAGE 1: Overview & research question
# ============================================================

def page_overview(raw_df, analytic_df, model_results):
    st.header("Project overview & research question")

    st.markdown(
        """
        ### Clinical question

        **Main research question**

        > *Can changes in pulse pressure between Visit 1 and Visit 2 predict the occurrence of a CVD event before Visit 3 in Framingham participants?*

        **Subquestions**

        1. Is the association between ΔPP (PP₂ − PP₁) and CVD different for women vs men?
        2. Does a clinically simple threshold in ΔPP identify higher CVD risk by Visit 3?

        ### Data & design

        - Cohort: Framingham Heart Study, three visits (PERIOD 1–3)
        - Long format: multiple rows per participant (by PERIOD)
        - Analytic dataset:
          - One row per participant
          - Includes:
            - Baseline features from Visit 1 (age, sex, BMI, BP, lipids, smoking)
            - ΔPP = pulse pressure change from Visit 1 → Visit 2
            - CVD outcome at Visit 3 (binary)
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Raw rows (long)", f"{len(raw_df):,}")
    with col2:
        st.metric("Unique participants (raw)", f"{raw_df['RANDID'].nunique():,}")
    with col3:
        st.metric("Analytic sample (ΔPP + CVD)", f"{len(analytic_df):,}")

    if "CVD" in analytic_df.columns:
        prev = analytic_df["CVD"].mean() * 100
        st.markdown(f"**Analytic CVD prevalence (Visit 3)**: `{prev:.1f}%`")

    st.markdown("---")
    st.subheader("Quick descriptive snapshot (analytic dataset)")

    with st.expander("Show first 10 rows of analytic dataset"):
        st.dataframe(analytic_df.head(10), use_container_width=True)

    # Basic ΔPP description
    if "DELTA_PP" in analytic_df.columns:
        st.markdown("#### Distribution of ΔPP (PP₂ − PP₁)")
        stats = analytic_df["DELTA_PP"].describe().round(2)
        st.write(stats)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(
            analytic_df["DELTA_PP"],
            bins=30,
            kde=True,
            color=crest_colors[4],
            ax=ax,
        )
        ax.set_xlabel("ΔPP (mmHg)")
        ax.set_title("Distribution of ΔPP in analytic sample")
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Preprocessing & modelling pipeline")

    st.markdown(
        """
        **Data cleaning & feature engineering**

        - **Outliers:** clinically motivated capping for blood pressure, BMI, cholesterol, glucose, and cigarettes/day  
        - **Missing data:**  
          - Dropped **LDLC/HDLC** (structural Period 3–only, MNAR)  
          - Median imputation for numeric variables (e.g. GLUCOSE, TOTCHOL, BMI)  
          - Mode imputation for categorical variables (e.g. BPMEDS, education)  
        - **Encoding:** recoded **SEX** to 0 = male / 1 = female; harmonised all binary variables to 0/1  
        - **Transforms:** log1p-transform for skewed variables (GLUCOSE, TOTCHOL, CIGPDAY)  
        - **Pulse pressure:**  
          - PP = SYSBP − DIABP per visit  
          - Built wide PP table (PP₁, PP₂, PP₃)  
          - Computed **ΔPP = PP₂ − PP₁** and kept only participants with complete PP at Visits 1 & 2  

        **Modelling strategy**

        - Outcome: CVD at Visit 3 (binary)  
        - Features: ΔPP plus Visit-1 covariates (age, sex, BMI, SBP, DBP, glucose, cholesterol, smoking)  
        - Train–test split: stratified, 80/20 on the analytic dataset  
        - Pipeline:
          - Column-wise preprocessing (median imputation, log transforms, scaling)
          - **SMOTE** applied in the training folds to address class imbalance
          - Stratified **5-fold cross-validation** for hyperparameter tuning (ROC AUC)  
        - Models evaluated:
          - Dummy baseline
          - Elastic-net logistic regression
          - Decision tree & random forest
          - KNN
          - SVM (RBF kernel)
          - Gradient boosting
          - MLP (sklearn)
          - TensorFlow neural network
        """
    )


# ============================================================
# PAGE 2: EDA on raw long data
# ============================================================

def page_eda_raw(raw_df):
    st.header("Exploratory data analysis – raw Framingham dataset")

    st.markdown(
        """
        Here we look at the **original long dataset**:

        - Structure (PERIOD, RANDID)
        - Missingness patterns
        - High-level visit distribution
        - Individual patient trajectories for BP, BMI, and cholesterol
        """
    )

    # Basic structure
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Dataset info**")
        st.write(f"Shape: `{raw_df.shape[0]:,}` rows × `{raw_df.shape[1]}` columns")
        st.write(f"Unique participants: `{raw_df['RANDID'].nunique():,}`")

    with col2:
        st.write("**PERIOD distribution**")
        period_counts = raw_df["PERIOD"].value_counts().sort_index()
        st.write(period_counts)

        fig, ax = plt.subplots(figsize=(4, 3))
        sns.barplot(
            x=period_counts.index.astype(str),
            y=period_counts.values,
            palette="crest",
            ax=ax,
        )
        ax.set_xlabel("PERIOD")
        ax.set_ylabel("Row count")
        ax.set_title("Rows per PERIOD")
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Missingness overview")

    missing_pct = (raw_df.isna().mean() * 100).round(2)
    missing_pct_nonzero = missing_pct[missing_pct > 0].sort_values(ascending=False)

    st.markdown("**Top 15 variables by % missing:**")
    st.dataframe(
        missing_pct_nonzero.head(15).to_frame("% missing"),
        use_container_width=True,
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    top_n = missing_pct_nonzero.head(15)
    sns.barplot(
        x=top_n.values,
        y=top_n.index,
        palette="crest",
        ax=ax,
    )
    ax.set_xlabel("% missing")
    ax.set_ylabel("")
    ax.set_title("Missing values by variable (top 15)")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("#### Missingness by PERIOD (selected variables)")

    default_cols = [
        "LDLC", "HDLC", "GLUCOSE", "BPMEDS",
        "TOTCHOL", "educ", "CIGPDAY", "BMI", "HEARTRTE",
    ]
    available_defaults = [c for c in default_cols if c in raw_df.columns]

    selected_cols = st.multiselect(
        "Variables for missingness heatmap",
        options=sorted(raw_df.columns),
        default=available_defaults,
    )

    if len(selected_cols) > 0 and "PERIOD" in raw_df.columns:
        miss_by_period = (
            raw_df.groupby("PERIOD")[selected_cols]
            .apply(lambda d: d.isna().mean() * 100)
            .round(1)
        )
        st.write("Missingness (%), by PERIOD:")
        st.dataframe(miss_by_period, use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(
            miss_by_period.T,
            annot=True,
            fmt=".1f",
            cmap=crest_cmap,
            cbar_kws={"label": "% missing"},
            ax=ax,
        )
        ax.set_xlabel("PERIOD")
        ax.set_ylabel("Variable")
        ax.set_title("Missingness by PERIOD")
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Individual patient trajectories")

    ids = sorted(raw_df["RANDID"].unique())
    selected_id = st.selectbox("Select RANDID", options=ids)

    user_data = raw_df[raw_df["RANDID"] == selected_id].sort_values("PERIOD")

    if user_data.empty:
        st.warning("No data for that ID.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # BP
    axes[0].plot(
        user_data["PERIOD"],
        user_data["SYSBP"],
        marker="o",
        label="SYSBP",
        color="tab:red",
    )
    axes[0].plot(
        user_data["PERIOD"],
        user_data["DIABP"],
        marker="o",
        label="DIABP",
        color="tab:blue",
    )
    axes[0].set_title("Blood pressure over visits")
    axes[0].set_xlabel("PERIOD")
    axes[0].set_ylabel("mmHg")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # BMI
    if "BMI" in user_data.columns:
        axes[1].plot(
            user_data["PERIOD"],
            user_data["BMI"],
            marker="s",
            linestyle="--",
            color="tab:green",
        )
        axes[1].set_title("BMI over visits")
        axes[1].set_xlabel("PERIOD")
        axes[1].set_ylabel("BMI (kg/m²)")
        axes[1].grid(True, linestyle="--", alpha=0.5)
    else:
        axes[1].axis("off")

    # Cholesterol
    if "TOTCHOL" in user_data.columns:
        axes[2].plot(
            user_data["PERIOD"],
            user_data["TOTCHOL"],
            marker="^",
            color="tab:orange",
        )
        axes[2].set_title("Total cholesterol over visits")
        axes[2].set_xlabel("PERIOD")
        axes[2].set_ylabel("mg/dL")
        axes[2].grid(True, linestyle="--", alpha=0.5)
    else:
        axes[2].axis("off")

    has_cvd = "YES" if user_data["CVD"].max() == 1 else "NO"
    fig.suptitle(f"Trajectory for RANDID {selected_id} (CVD by Visit 3: {has_cvd})")
    plt.tight_layout()
    st.pyplot(fig)


# ============================================================
# PAGE 3: ΔPP & analytic dataset exploration
# ============================================================

def page_delta_pp(analytic_df):
    st.header("ΔPP & analytic dataset")

    if "DELTA_PP" not in analytic_df.columns:
        st.error("Analytic dataset has no 'DELTA_PP' column. Check your export.")
        return

    st.markdown(
        """
        This page focuses on the **analytic dataset** used for modelling:

        - ΔPP (PP₂ − PP₁) distribution
        - Relationship between ΔPP and CVD
        - Correlations with baseline covariates
        - Interactive scatter plots
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Analytic dataset shape**")
        st.write(f"{analytic_df.shape[0]:,} rows × {analytic_df.shape[1]} columns")

    with col2:
        if "CVD" in analytic_df.columns:
            prev = analytic_df["CVD"].mean() * 100
            st.write(f"**CVD prevalence (analytic)**: {prev:.1f}%")

    st.markdown("---")
    st.subheader("ΔPP distribution & CVD")

    stats = analytic_df["DELTA_PP"].describe().round(2)
    st.write("ΔPP summary statistics:")
    st.write(stats)

    # Histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(
        analytic_df["DELTA_PP"],
        bins=30,
        kde=True,
        color=crest_colors[4],
        ax=ax,
    )
    ax.set_xlabel("ΔPP (PP₂ − PP₁, mmHg)")
    ax.set_title("Distribution of ΔPP")
    plt.tight_layout()
    st.pyplot(fig)

    # Boxplot by CVD
    if "CVD" in analytic_df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(
            data=analytic_df,
            x="CVD",
            y="DELTA_PP",
            palette="crest",
            ax=ax,
        )
        ax.set_xlabel("CVD at Visit 3 (0 = No, 1 = Yes)")
        ax.set_ylabel("ΔPP (mmHg)")
        ax.set_title("ΔPP by CVD outcome")
        plt.tight_layout()
        st.pyplot(fig)

    # If sex present
    if "V1_SEX" in analytic_df.columns:
        st.markdown("**ΔPP by CVD and sex (baseline)**")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(
            data=analytic_df,
            x="V1_SEX",
            y="DELTA_PP",
            hue="CVD",
            palette="crest",
            ax=ax,
        )
        ax.set_xlabel("Sex at Visit 1 (0 = male, 1 = female)")
        ax.set_ylabel("ΔPP (mmHg)")
        ax.set_title("ΔPP by sex and CVD")
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Correlations between ΔPP and baseline covariates")

    corr_vars = [c for c in [
        "DELTA_PP",
        "V1_AGE",
        "V1_BMI",
        "V1_SYSBP",
        "V1_DIABP",
        "V1_GLUCOSE",
        "V1_TOTCHOL",
        "V1_CIGPDAY",
    ] if c in analytic_df.columns]

    if len(corr_vars) >= 2:
        corr_table = analytic_df[corr_vars].corr().round(2)
        st.dataframe(corr_table, use_container_width=True)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            corr_table,
            annot=True,
            cmap=crest_cmap,
            center=0,
            ax=ax,
        )
        ax.set_title("Correlation among ΔPP and baseline variables")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Not enough numeric baseline variables found for correlation heatmap.")

    st.markdown("---")
    st.subheader("Interactive scatter exploration")

    numeric_cols = analytic_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        st.info("Not enough numeric variables to build scatter plots.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        x_var = st.selectbox("X-axis", options=sorted(numeric_cols), index=0)
    with col2:
        y_var = st.selectbox("Y-axis", options=sorted(numeric_cols), index=1)
    with col3:
        color_vars = ["None"] + list(analytic_df.columns)
        color_by = st.selectbox(
            "Colour by",
            options=color_vars,
            index=color_vars.index("CVD") if "CVD" in color_vars else 0,
        )

    fig, ax = plt.subplots(figsize=(7, 5))
    if color_by != "None":
        sns.scatterplot(
            data=analytic_df,
            x=x_var,
            y=y_var,
            hue=color_by,
            palette="crest",
            alpha=0.6,
            ax=ax,
        )
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    else:
        sns.scatterplot(
            data=analytic_df,
            x=x_var,
            y=y_var,
            color=crest_colors[4],
            alpha=0.6,
            ax=ax,
        )

    ax.set_title(f"{x_var} vs {y_var}")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)


# ============================================================
# PAGE 4: Model comparison (global)
# ============================================================

def page_models_overview(model_results):
    st.header("Model zoo – global performance comparison")

    if model_results is None or model_results.empty:
        st.error("No model_results loaded. Check that model_results.csv is present.")
        return

    st.markdown(
        """
        This page compares all trained models on the **held-out test set**:

        - Accuracy
        - ROC AUC
        - Cross-validated best AUC (CV_Best_AUC)
        """
    )

    if "ROC_AUC" in model_results.columns:
        model_results_sorted = model_results.sort_values(
            "ROC_AUC", ascending=False
        ).reset_index(drop=True)
    else:
        model_results_sorted = model_results.copy()

    # Highlight best model at the top
    st.subheader("Best-performing model (test ROC AUC)")

    if not model_results_sorted.empty:
        best_row = model_results_sorted.iloc[0]
        best_model_name = best_row.get("Model", "Unknown model")
        best_acc = float(best_row.get("Accuracy", np.nan))
        best_auc = float(best_row.get("ROC_AUC", np.nan))
        best_cv = float(best_row.get("CV_Best_AUC", np.nan)) if "CV_Best_AUC" in best_row else np.nan

        st.markdown(f"**Best model:** `{best_model_name}`")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Test accuracy", f"{best_acc:.3f}" if not np.isnan(best_acc) else "–")
        with c2:
            st.metric("Test ROC AUC", f"{best_auc:.3f}" if not np.isnan(best_auc) else "–")
        with c3:
            if not np.isnan(best_cv):
                st.metric("Best CV AUC", f"{best_cv:.3f}")
            else:
                st.metric("Best CV AUC", "–")

    st.markdown("---")
    st.subheader("Model comparison table")
    st.dataframe(
        model_results_sorted.style.format(
            {
                "Accuracy": "{:.3f}",
                "ROC_AUC": "{:.3f}",
                "CV_Best_AUC": "{:.3f}" if "CV_Best_AUC" in model_results.columns else "{:.3f}",
            },
            na_rep="-",
        ),
        use_container_width=True,
    )

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Accuracy by model (test set)**")
        fig_acc, ax_acc = plt.subplots(figsize=(6, 4))
        sns.barplot(
            data=model_results_sorted,
            x="Accuracy",
            y="Model",
            palette="crest",
            ax=ax_acc,
        )
        ax_acc.set_xlim(0.5, 1.0)
        ax_acc.set_xlabel("Accuracy")
        ax_acc.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig_acc)

    with col2:
        st.markdown("**ROC AUC by model (test set)**")
        fig_auc, ax_auc = plt.subplots(figsize=(6, 4))
        sns.barplot(
            data=model_results_sorted,
            x="ROC_AUC",
            y="Model",
            palette="crest",
            ax=ax_auc,
        )
        ax_auc.set_xlim(0.5, 1.0)
        ax_auc.set_xlabel("ROC AUC")
        ax_auc.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig_auc)

    st.markdown("---")
    st.subheader("Accuracy vs ROC AUC (model types)")

    if "Type" in model_results_sorted.columns:
        plot_df = model_results_sorted.drop_duplicates(
            subset=["ID", "Model", "Accuracy", "ROC_AUC"]
        ).copy()

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(
            data=plot_df,
            x="ROC_AUC",
            y="Accuracy",
            hue="Type",
            style="Type",
            s=90,
            palette="crest",
            ax=ax,
        )

        for _, row in plot_df.iterrows():
            ax.text(
                x=row["ROC_AUC"] + 0.002,
                y=row["Accuracy"] + 0.002,
                s=str(row["ID"]),
                fontsize=8,
            )

        ax.set_xlim(0.45, 0.8)
        ax.set_ylim(0.55, 0.8)
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
        ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel("ROC AUC")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs ROC AUC per model (ID labels)")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Column 'Type' not found in model_results; skipping type-based scatter.")


# ============================================================
# PAGE 5: Model detail & threshold explorer
# ============================================================

def page_model_detail(selected_model_name, threshold, all_model_outputs, model_results):
    st.header(f"Detailed model view: {selected_model_name}")

    if selected_model_name not in all_model_outputs:
        st.error("Selected model not found in all_model_outputs.")
        return

    res = all_model_outputs[selected_model_name]

    base_acc = float(res.get("accuracy", np.nan))
    base_auc = float(res.get("roc_auc", np.nan))
    cv_best_auc = float(res.get("cv_best_auc", np.nan))

    # If missing CV_Best_AUC, try to pull from model_results by name
    if (np.isnan(cv_best_auc)
            and model_results is not None
            and "Model" in model_results.columns
            and "CV_Best_AUC" in model_results.columns):
        row = model_results[model_results["Model"] == selected_model_name]
        if not row.empty:
            cv_best_auc = float(row.iloc[0]["CV_Best_AUC"])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Test accuracy (threshold 0.5)", f"{base_acc:.3f}")
    with col2:
        st.metric("Test ROC AUC", f"{base_auc:.3f}")
    with col3:
        if not np.isnan(cv_best_auc):
            st.metric("Best CV AUC", f"{cv_best_auc:.3f}")
        else:
            # Use pipeline (preprocessing + classifier)
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('clf', LogisticRegression(max_iter=200, class_weight='balanced'))
            ])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

        # Compute metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        st.subheader("Model Evaluation")
        st.write(f"Rows used for modeling: {df_model.shape[0]}")
        st.write(f"Accuracy: {acc:.4f}")
        st.write(f"Precision: {prec:.4f}")
        st.write(f"Recall (sensitivity): {rec:.4f}")
        st.write(f"F1 score: {f1:.4f}")

        st.write("Confusion Matrix:")
        st.write(cm)

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred, zero_division=0))

except Exception as e:
    # If sklearn or other packages aren't installed, show a friendly message but do not crash the app
    st.error(f"Modeling section encountered an error: {e}")
    st.info("If this is due to missing packages, install them: `pip install scikit-learn imbalanced-learn` and restart the app.")

# FINDINGS 

import streamlit as st

st.set_page_config(
    page_title="Covid Data Analysis - Findings & Conclusion",
    page_icon="📊",
    #layout="wide"   
)
#st.markdown("""<style>body {zoom: 1.4;  /* Adjust this value as needed */}</style>""", unsafe_allow_html=True)

st.sidebar.markdown("""<div style="font-size: 17px;">✍️ <strong>Authors:</strong></div> 
\n&nbsp;                                  
<a href="https://www.linkedin.com/in/amralshatnawi/" style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; text-decoration: none; font-size: 15px; border-radius: 4px;">&nbsp;&nbsp;Amr Alshatnawi&nbsp;&nbsp;</a><br>             

<a href="https://www.linkedin.com/in/hailey-pangburn" style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; text-decoration: none; font-size: 15px; border-radius: 4px;">&nbsp;&nbsp;Hailey Pangburn&nbsp;&nbsp;</a><br>             
                    
<a href="mailto:mcmasters@uchicago.edu" style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; text-decoration: none; font-size: 15px; border-radius: 4px;">Richard McMasters</a><br>
""", unsafe_allow_html=True)

st.sidebar.write("---")
st.sidebar.markdown("""📅 March 9th, 2024""")
show_sidebar_logo()


############################# start page content #############################

st.title("Findings & Conclusion")
st.divider()



st.markdown("""

Our analysis aimed to uncover patterns in COVID-19 case counts and mortality outcomes, focusing on the roles of age, gender, and case year.
Initially, we explored the distribution of COVID-19 cases across various age groups. The data revealed a significant deviation in the 18 to 49 age group,
which displayed a disproportionately high number of cases. Utilizing the Chi-square goodness-of-fit test, we determined this variance to be statistically
significant, with a p-value less than 0.05. This finding suggests certain age groups, notably the 18 to 49 demographic, are more susceptible to contracting
COVID-19 relative to their population size, potentially due to factors like social behavior and employment types.

In examining COVID-19 mortality outcomes, logistic regression analysis highlighted that gender, age group, and case year are significant predictors of mortality,
with all predictors showing statistical significance (p-values < 0.05). Despite an initial dataset imbalance, our resampling strategy, which included both
undersampling and oversampling techniques, allowed us to maintain the model's overall significance while revealing an increased baseline probability of death
in a more balanced dataset context. This adjustment suggests a refined understanding of mortality risk factors. However, the model's precision at 10.62%
indicates a high rate of false positives, a challenge balanced by its strong sensitivity (81.44%) in accurately identifying actual deaths. This emphasizes
the model's utility in critical public health scenarios despite its need for further optimization to reduce false positives and improve the F1 score (0.1879).

In conclusion, our findings confirm the significant impact of gender, age, and case year on COVID-19 mortality, underscoring the importance of targeted public
health strategies. While the model presents areas for improvement, its ability to predict true positives remains a valuable asset in managing the pandemic
response, highlighting the potential for further refinement to enhance its predictive accuracy and applicability.


""")

# REFERENCES

import streamlit as st

st.set_page_config(
    page_title="Covid Data Analysis - References",
    page_icon="📊",
    #layout="wide"   
)
#st.markdown("""<style>body {zoom: 1.4;  /* Adjust this value as needed */}</style>""", unsafe_allow_html=True)

st.sidebar.markdown("""<div style="font-size: 17px;">✍️ <strong>Authors:</strong></div> 
\n&nbsp;                                  
<a href="https://www.linkedin.com/in/amralshatnawi/" style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; text-decoration: none; font-size: 15px; border-radius: 4px;">&nbsp;&nbsp;Amr Alshatnawi&nbsp;&nbsp;</a><br>             

<a href="https://www.linkedin.com/in/hailey-pangburn" style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; text-decoration: none; font-size: 15px; border-radius: 4px;">&nbsp;&nbsp;Hailey Pangburn&nbsp;&nbsp;</a><br>             
                    
<a href="mailto:mcmasters@uchicago.edu" style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; text-decoration: none; font-size: 15px; border-radius: 4px;">Richard McMasters</a><br>
""", unsafe_allow_html=True)

st.sidebar.write("---")
st.sidebar.markdown("""📅 March 9th, 2024""")
show_sidebar_logo()

# def add_side_title():
#     st.markdown(
#         """
#         <style>
#             [data-testid="stSidebarNav"]::before {
#                 content:"MSBI 32000 Winter 2024";
#                 margin-left: 20px;
#                 margin-top: 20px;
#                 font-size: 25px;
#                 position: relative;
#                 top: 80px;
#             }
#         </style>
#         """,
#         unsafe_allow_html=True,
#     )

# add_side_title()

############################# start page content #############################

st.title("References")
st.divider()

st.markdown("""
1. Brownlee, J. (2021, March 16). Smote for imbalanced classification with python. MachineLearningMastery.com. https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/ 
            
2. Centers for Disease Control and Prevention. (n.d.-a). About covid-19. Centers for Disease Control and Prevention. https://www.cdc.gov/coronavirus/2019-ncov/your-health/about-covid-19.html#:~:text=COVID%2D19%20(coronavirus%20disease%202019,%2C%20the%20flu%2C%20or%20pneumonia. 
            
3. Centers for Disease Control and Prevention. (n.d.-b). Covid-19 case surveillance public use data with geography. Centers for Disease Control and Prevention. https://data.cdc.gov/Case-Surveillance/COVID-19-Case-Surveillance-Public-Use-Data-with-Ge/n8mc-b4w4/about_data 
            
4. Coronavirus cases:. Worldometer. (n.d.). https://www.worldometers.info/coronavirus/ 
            
5. Randomundersampler#. RandomUnderSampler - Version 0.12.0. (n.d.). https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html 
            
6. United States population by age - 2023 united states age demographics. Neilsberg. (n.d.). https://www.neilsberg.com/insights/united-states-population-by-age/#pop-by-age 
""")