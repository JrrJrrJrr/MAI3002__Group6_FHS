# streamlit_app.py
#imports
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Scikit-learn metrics for model evaluation
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

# --- Statsmodels is optional (some environments don't have it)
try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except Exception:
    sm = None
    HAS_STATSMODELS = False


# --- Global style and color + ONE palette everywhere for consistency
sns.set_style("whitegrid")
PALETTE = sns.color_palette("viridis", 8)
CREST_CMAP = sns.color_palette("viridis", as_cmap=True)

# --- Fixed colors per visit period (used in longitudinal plots)
PERIOD_PALETTE = {1: PALETTE[2], 2: PALETTE[4], 3: PALETTE[6]}


def solid_color(i=4):
    """One consistent 'main' color for single-series plots."""
    return PALETTE[i]


# --- Load raw data (long format) from GitHub (cached)
# --- Streamlit caching is used to prevent unnecessary reloading

@st.cache_data
def load_raw_framingham():
    # --- Load the raw Framingham dataset (long format) from GitHub.
    url = (
        "https://raw.githubusercontent.com/"
        "LUCE-Blockchain/Databases-for-teaching/"
        "refs/heads/main/Framingham%20Dataset.csv"
    )
    return pd.read_csv(url)


# --- Load project outputs exported from Colab (cached)
@st.cache_data
def load_project_outputs():
    """
    analytic_df : pd.DataFrame
    Cleaned analytic dataset (one row per participant).
    model_results : pd.DataFrame
    Summary table comparing model performance.
    all_model_outputs : dict
    Stored test predictions and metrics per model.
    """
    analytic_df = pd.read_csv("analytic_dataset.csv")
    model_results = pd.read_csv("model_results.csv")
    all_model_outputs = joblib.load("all_model_outputs.pkl")
    return analytic_df, model_results, all_model_outputs


# --- Sidebar: project info + vertical segmented control (buttons)
def sidebar_block_and_nav():
    
    # Custom CSS for sidebar styling
    st.markdown(
        """
<style>
/* Sidebar buttons styled as a connected vertical segmented control */
section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    border-radius: 0px;
    padding: 0.6rem 0.75rem;
    margin: 0px !important;
    border: 1px solid rgba(49, 51, 63, 0.22);
    background: rgba(255,255,255,0.0);
    text-align: left;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    border-color: rgba(49, 51, 63, 0.45);
    background: rgba(49, 51, 63, 0.04);
}

/* Rounded corners only for first + last */
section[data-testid="stSidebar"] .nav-first .stButton > button {
    border-top-left-radius: 12px;
    border-top-right-radius: 12px;
}
section[data-testid="stSidebar"] .nav-last .stButton > button {
    border-bottom-left-radius: 12px;
    border-bottom-right-radius: 12px;
}

/* Active state */
section[data-testid="stSidebar"] .nav-active .stButton > button {
    font-weight: 800;
    border-color: rgba(49, 51, 63, 0.60) !important;
    background: rgba(49, 51, 63, 0.08) !important;
}
</style>
""",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Project information")

        # Placeholder image UM Logo
        st.image(
            "https://via.placeholder.com/300x120.png?text=Maastricht+University",
            caption="Maastricht University",
            use_container_width=True,
        )
# Course and student information
        st.markdown(
            """
**Course:** MAI3002  
**Course name:** Introduction to Programming in Python  

**Students:**
- Cleo Habets — `i6337758`  
- Jerrica Pubben — `i6276134`  
- Noura al Sayed — `i6359287`
"""
        )

        st.markdown("---")
        st.markdown("### Navigation")

        pages = [
            "1) Overview",
            "2) Exploratory data analysis",
            "3) ΔPP & analytic exploration",
            "4) Model comparison",
            "5) Model detail + threshold",
            "6) Interaction test (ΔPP × sex)",
            "7) Final RQ recap",
        ]
        # Initialize page state
        if "page" not in st.session_state:
            st.session_state.page = pages[0]
            
        # Render navigation buttens
        for i, p in enumerate(pages):
            classes = []
            if i == 0:
                classes.append("nav-first")
            if i == len(pages) - 1:
                classes.append("nav-last")
            if st.session_state.page == p:
                classes.append("nav-active")

            cls = " ".join(classes)
            st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
            clicked = st.button(p, key=f"nav_{p}")
            st.markdown("</div>", unsafe_allow_html=True)

            if clicked:
                st.session_state.page = p
                st.rerun()

        return st.session_state.page


# --- Helper: recompute confusion matrix + metrics for a custom threshold
def compute_threshold_metrics(y_true, y_proba, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba)
    
    # Convert probabilities to class predictions
    y_pred_thr = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred_thr)
    acc = accuracy_score(y_true, y_pred_thr)

    # AUC uses probabilities (threshold-free)
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


# --- Helper: ROC plot (palette-consistent)
def plot_roc(y_test, y_proba):
    """
    The ROC curve visualizes the trade-off between sensitivity
    and specificity across all classification thresholds.
    """
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_val = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=solid_color(4), label=f"AUC = {auc_val:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("False Positive Rate (1 - specificity)")
    ax.set_ylabel("True Positive Rate (sensitivity)")
    ax.set_title("ROC curve")
    ax.legend()
    st.pyplot(fig)


# --- Helper: PR plot (palette-consistent)
def plot_pr(y_test, y_proba):
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    baseline = float(np.mean(y_test))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color=solid_color(4), label=f"AP = {ap:.3f}")
    ax.axhline(baseline, linestyle="--", color="grey", linewidth=1, label=f"Baseline = {baseline:.3f}")
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title("Precision–Recall curve")
    ax.legend()
    st.pyplot(fig)


# --- Helper: calibration plot (palette-consistent)
def plot_calibration(y_test, y_proba, bins=10):
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=bins, strategy="quantile")
    brier = brier_score_loss(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, marker="o", color=solid_color(4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.set_title(f"Calibration plot (Brier = {brier:.3f})")
    st.pyplot(fig)


# --- Page 1: overview + text + funnel + imbalance context (palette-consistent)
def page_overview(raw_df, analytic_df):
    st.header("Project overview")
    
    # --- Research question & study design explanation
    st.markdown(
        """
**Main research question**  
Can changes in pulse pressure between Visit 1 and Visit 2 (ΔPP) predict whether a participant experiences a CVD event by Visit 3?

**Subquestions**  
1) Does the ΔPP–CVD association differ by sex? (ΔPP × sex interaction)  

**Study design and data flow**  
The Framingham dataset is in **long format** (multiple rows per participant across PERIOD 1–3).  
We constructed an **analytic dataset** with **one row per participant**, containing:
- Pulse pressure per visit: **PP = SYSBP − DIABP**
- Main predictor: **ΔPP = PP₂ − PP₁**
- Baseline covariates from Visit 1 (age, sex, BMI, blood pressure, glucose, cholesterol, smoking)
- Outcome: **CVD at Visit 3** (binary)

**Key preprocessing decisions**  
- Structural missingness: **HDLC/LDLC** are largely absent outside Period 3 → excluded rather than imputed.  
- Remaining missing values: median (numeric) / mode (categorical) imputation.  
- Outliers: clinically motivated capping (winsorizing) to retain participants.  
- Modeling: train/test split; **ROC AUC** as primary discrimination metric due to class imbalance.
"""
    )

    st.markdown("---")

    # --- Quick dataset metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Raw rows (long)", f"{len(raw_df):,}")
    with c2:
        st.metric("Raw participants", f"{raw_df['RANDID'].nunique():,}")
    with c3:
        st.metric("Analytic rows", f"{len(analytic_df):,}")
    with c4:
        if "CVD" in analytic_df.columns:
            st.metric("Analytic CVD (%)", f"{analytic_df['CVD'].mean()*100:.1f}")
        else:
            st.metric("Analytic CVD (%)", "–")

    st.markdown("---")

    # --- Inclusion funnel
    # Shows how many participants remain after each selection step
    st.subheader("Analytic sample construction (inclusion funnel)")

    n_raw = raw_df["RANDID"].nunique()

    # Compute pulse pressure where SBP and DBP are available
    needed_cols = ["RANDID", "PERIOD", "SYSBP", "DIABP"]
    tmp = raw_df[needed_cols].dropna().copy()
    tmp["PP"] = tmp["SYSBP"] - tmp["DIABP"]

    # Reshape to wide format to identify participants with PP at Visit 1 & 2
    pp_wide = (
        tmp.drop_duplicates(["RANDID", "PERIOD"])
           .pivot(index="RANDID", columns="PERIOD", values="PP")
    )
    n_pp12 = pp_wide.dropna(subset=[1, 2]).shape[0]
    
    # Count participants with observed CVD outcome at Visit 3
    if "CVD" in raw_df.columns:
        n_cvd3 = (
            raw_df.loc[raw_df["PERIOD"] == 3, ["RANDID", "CVD"]]
            .dropna()
            ["RANDID"]
            .nunique()
        )
    else:
        n_cvd3 = np.nan
        
    # Final analytic sample size
    if "RANDID" in analytic_df.columns:
        n_analytic = analytic_df["RANDID"].nunique()
    else:
        n_analytic = len(analytic_df)

    funnel = pd.DataFrame({
        "Step": [
            "Unique participants in raw",
            "Has PP at Visit 1 & 2 (for ΔPP)",
            "Has CVD recorded at Visit 3",
            "Final analytic dataset"
        ],
        "N": [n_raw, n_pp12, n_cvd3, n_analytic]
    })

    st.dataframe(funnel, use_container_width=True)

     # Visual funnel representation
    fig, ax = plt.subplots(figsize=(8, 3.2))
    sns.barplot(data=funnel, x="N", y="Step", ax=ax, color=solid_color(4))
    ax.set_title("Participant inclusion funnel")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    # --- Quick look at analytic dataset structure
    st.subheader("Quick snapshot of analytic dataset")
    with st.expander("Show first 10 rows"):
        st.dataframe(analytic_df.head(10), use_container_width=True)

    with st.expander("Show variable list"):
        st.write(list(analytic_df.columns))


# --- Page 2: raw EDA (palette-consistent + added sanity checks)
def page_eda_raw(raw_df):
    st.header("Exploratory data analysis – raw Framingham dataset (long format)")

    st.markdown(
        """
This page summarizes the **raw longitudinal structure** (multiple rows per participant).  
Focus: visit coverage (PERIOD), missingness patterns, and a focused set of distributions.
"""
    )

    st.markdown("---")

    # --- Basic dataset info + PERIOD distribution
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Dataset structure")
        st.write(f"Rows × columns: `{raw_df.shape[0]:,}` × `{raw_df.shape[1]}`")
        st.write(f"Unique participants (RANDID): `{raw_df['RANDID'].nunique():,}`")

    with c2:
        st.subheader("Rows per PERIOD")
        period_counts = raw_df["PERIOD"].value_counts().sort_index()
        st.dataframe(period_counts.rename("Rows").to_frame(), use_container_width=True)

        fig, ax = plt.subplots(figsize=(5, 3))
        sns.barplot(
            x=period_counts.index.astype(str),
            y=period_counts.values,
            ax=ax,
            palette=[PERIOD_PALETTE.get(int(p), solid_color(4)) for p in period_counts.index],
        )
        ax.set_xlabel("PERIOD")
        ax.set_ylabel("Row count")
        ax.set_title("Rows per PERIOD")
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")

    # --- Number of visits per participant
    st.subheader("Visits per participant (longitudinal completeness)")

    visits_per_person = (
        raw_df.groupby("RANDID")["PERIOD"]
        .nunique()
        .value_counts()
        .sort_index()
    )
    visits_df = visits_per_person.rename_axis("Visits attended").reset_index(name="Participants")
    st.dataframe(visits_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(5.5, 3))
    sns.barplot(
        data=visits_df,
        x="Visits attended",
        y="Participants",
        ax=ax,
        color=solid_color(4),
    )
    ax.set_title("How many visits did participants attend?")
    plt.tight_layout()
    st.pyplot(fig)

    st.caption(
        "Note: the dataframe index is not '0 visits'. The table means: N participants with 1 / 2 / 3 observed PERIODs."
    )

    st.markdown("---")

    # --- Visit coverage by PERIOD (sanity check)
    st.subheader("Visit coverage by PERIOD (sanity check)")

    coverage = (
        raw_df.drop_duplicates(["RANDID", "PERIOD"])
              .groupby("PERIOD")["RANDID"]
              .nunique()
              .sort_index()
    )
    coverage_df = coverage.rename("Participants").reset_index()
    coverage_df["% of all participants"] = (
        coverage_df["Participants"] / raw_df["RANDID"].nunique() * 100
    ).round(1)

    st.dataframe(coverage_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(5.5, 3))
    sns.barplot(
        data=coverage_df,
        x="PERIOD",
        y="Participants",
        ax=ax,
        palette=[PERIOD_PALETTE.get(int(p), solid_color(4)) for p in coverage_df["PERIOD"]],
    )
    ax.set_title("Unique participants observed in each PERIOD")
    plt.tight_layout()
    st.pyplot(fig)

    st.caption(
        "This explains why '3 visits' can be the largest group: many participants have complete records across PERIOD 1–3, "
        "while fewer have missing PERIODs."
    )

    st.markdown("---")

    # --- Outcome prevalence across PERIODs
    st.subheader("Outcome signal across PERIODs (sanity check)")

    if "CVD" in raw_df.columns:
        cvd_by_period = (
            raw_df.groupby("PERIOD")["CVD"]
            .mean()
            .mul(100)
            .round(2)
            .reset_index(name="CVD prevalence (%)")
        )
        st.dataframe(cvd_by_period, use_container_width=True)

        fig, ax = plt.subplots(figsize=(5.5, 3))
        sns.barplot(
            data=cvd_by_period,
            x="PERIOD",
            y="CVD prevalence (%)",
            ax=ax,
            palette=[PERIOD_PALETTE.get(int(p), solid_color(4)) for p in cvd_by_period["PERIOD"]],
        )
        ax.set_title("CVD prevalence by PERIOD")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("CVD column not available in the raw dataset.")

    st.markdown("---")

    # --- Missingness overview
    st.subheader("Missingness overview (raw)")

    missing_pct = (raw_df.isna().mean() * 100).round(2)
    missing_nonzero = missing_pct[missing_pct > 0].sort_values(ascending=False)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.write("Top variables by % missing:")
        st.dataframe(missing_nonzero.head(15).to_frame("% missing"), use_container_width=True)

    with c2:
        fig, ax = plt.subplots(figsize=(7, 4))
        top = missing_nonzero.head(15)
        sns.barplot(x=top.values, y=top.index, ax=ax, color=solid_color(4))
        ax.set_xlabel("% missing")
        ax.set_ylabel("")
        ax.set_title("Missingness (top 15)")
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")

    # --- Missingness by PERIOD heatmap (user-selected variables)
    st.subheader("Missingness by PERIOD (selected variables)")

    default_cols = ["LDLC", "HDLC", "GLUCOSE", "BPMEDS", "TOTCHOL", "educ", "CIGPDAY", "BMI", "HEARTRTE"]
    defaults = [c for c in default_cols if c in raw_df.columns]

    selected_cols = st.multiselect(
        "Variables to include",
        options=sorted(raw_df.columns),
        default=defaults,
    )

    if selected_cols:
        miss_by_period = (
            raw_df.groupby("PERIOD")[selected_cols]
            .apply(lambda d: d.isna().mean() * 100)
            .round(1)
        )
        st.dataframe(miss_by_period, use_container_width=True)

        fig, ax = plt.subplots(figsize=(9, 4))
        sns.heatmap(
            miss_by_period.T,
            annot=True,
            fmt=".1f",
            ax=ax,
            cmap=CREST_CMAP,
        )
        ax.set_xlabel("PERIOD")
        ax.set_ylabel("Variable")
        ax.set_title("Missingness by PERIOD (%)")
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")

    # --- Core numeric distributions by PERIOD
    st.subheader("Core numeric distributions by PERIOD (raw)")

    candidate_nums = [c for c in ["AGE", "SYSBP", "DIABP", "BMI", "TOTCHOL", "GLUCOSE", "CIGPDAY"] if c in raw_df.columns]
    chosen = st.multiselect(
        "Pick numeric variables to plot",
        options=candidate_nums,
        default=[c for c in ["SYSBP", "DIABP", "BMI", "TOTCHOL"] if c in candidate_nums],
    )

    if chosen:
        for col in chosen:
            fig, ax = plt.subplots(figsize=(7, 3.5))
            sns.histplot(
                data=raw_df,
                x=col,
                hue="PERIOD",
                palette=PERIOD_PALETTE,
                bins=30,
                element="step",
                common_norm=False,
                ax=ax,
            )
            ax.set_title(f"{col} distribution by PERIOD")
            plt.tight_layout()
            st.pyplot(fig)

    st.markdown("---")

    # --- Patient trajectories (example)
    st.subheader("Individual patient trajectories (example)")

    ids = sorted(raw_df["RANDID"].unique())
    selected_id = st.selectbox("Select RANDID", options=ids)

    user_data = raw_df[raw_df["RANDID"] == selected_id].sort_values("PERIOD")

    with st.expander("Show selected patient's raw rows"):
        cols = [c for c in ["RANDID", "PERIOD", "SYSBP", "DIABP", "BMI", "TOTCHOL", "CVD"] if c in user_data.columns]
        st.dataframe(user_data[cols], use_container_width=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Blood pressure
    if "SYSBP" in user_data.columns and "DIABP" in user_data.columns:
        axes[0].plot(user_data["PERIOD"], user_data["SYSBP"], marker="o", color=solid_color(6), label="SYSBP")
        axes[0].plot(user_data["PERIOD"], user_data["DIABP"], marker="o", color=solid_color(3), label="DIABP")
        axes[0].set_title("Blood pressure")
        axes[0].set_xlabel("PERIOD")
        axes[0].set_ylabel("mmHg")
        axes[0].legend()
        axes[0].grid(True, linestyle="--", alpha=0.4)
    else:
        axes[0].axis("off")

    # BMI
    if "BMI" in user_data.columns and user_data["BMI"].notna().any():
        axes[1].plot(user_data["PERIOD"], user_data["BMI"], marker="s", linestyle="--", color=solid_color(4))
        axes[1].set_title("BMI")
        axes[1].set_xlabel("PERIOD")
        axes[1].set_ylabel("kg/m²")
        axes[1].grid(True, linestyle="--", alpha=0.4)
    else:
        axes[1].axis("off")

    # Cholesterol
    if "TOTCHOL" in user_data.columns and user_data["TOTCHOL"].notna().any():
        axes[2].plot(user_data["PERIOD"], user_data["TOTCHOL"], marker="^", color=solid_color(4))
        axes[2].set_title("Total cholesterol")
        axes[2].set_xlabel("PERIOD")
        axes[2].set_ylabel("mg/dL")
        axes[2].grid(True, linestyle="--", alpha=0.4)
    else:
        axes[2].axis("off")

    has_cvd = "YES" if ("CVD" in user_data.columns and user_data["CVD"].max() == 1) else "NO"
    fig.suptitle(f"Trajectory for RANDID {selected_id} (CVD by Visit 3: {has_cvd})")
    plt.tight_layout()
    st.pyplot(fig)


# --- Page 3: analytic / ΔPP exploration (palette-consistent)
def page_delta_pp(analytic_df):
    st.header("ΔPP & analytic dataset exploration")

    # Safety check: ΔPP must exist in analytic dataset
    if "DELTA_PP" not in analytic_df.columns:
        st.error("Analytic dataset missing DELTA_PP.")
        return
        
     # Conceptual explanation of ΔPP and analytic sample construction
    st.markdown(
        """
The **change in pulse pressure** over time (ΔPP) can be a *predictor* of future CVD events. 
A significant increase or decrease in PP between visits could indicate changes in cardiovascular health status. 
The focus is on the change between visit 1 and visit 2 to predict CVD outcomes at visit 3.

The final analytic dataset represents individuals with:
* Complete BP data at Visits 1 & 2 (to compute ΔPP)
* Observed CVD outcome by Visit 3
"""
    )

    # Basic dataset characteristics
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"Rows × columns: `{analytic_df.shape[0]:,}` × `{analytic_df.shape[1]}`")
    with c2:
        if "CVD" in analytic_df.columns:
            prev = analytic_df["CVD"].mean() * 100
            st.write(f"CVD prevalence: **{prev:.1f}%**")

    # ΔPP distribution and summary statistics
    st.markdown("---")
    st.subheader("ΔPP summary statistics ")
    st.write(analytic_df["DELTA_PP"].describe().round(2))

    st.markdown(
        """
There is a **mean** increase of **3.87** mmHg (and a **median** of **3.0**). 
This indicates that, on average, pulse pressure is widening between visits. 
This is biologically consistent with the aging process, as arteries tend to stiffen over time, naturally increasing pulse pressure in the general population.

Remarkable is a **standard deviation** of **12.23**. This value is large relative to the mean.
It tells us that although the mean increase is modest, individual changes in pulse pressure vary widely.
This high variance suggests that ΔPP is a *discriminating* feature, which is beneficial for the model. 
If ΔPP changed by the same amount for every participant, the variable would have no predictive power.
"""
    )

    # Histogram of ΔPP
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(analytic_df["DELTA_PP"], bins=30, kde=True, ax=ax, color=solid_color(4))
    ax.set_xlabel("ΔPP (PP₂ − PP₁, mmHg)")
    ax.set_title("Distribution of ΔPP")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown(
        """
The distribution is slightly skewed to the right and displays this wide variance. 
"""
    )

    # ΔPP stratified by CVD outcome
    st.markdown("---")
    st.subheader("ΔPP vs CVD")
    if "CVD" in analytic_df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=analytic_df, x="CVD", y="DELTA_PP", ax=ax, palette=[solid_color(3), solid_color(6)])
        ax.set_xlabel("CVD at Visit 3 (0=no, 1=yes)")
        ax.set_ylabel("ΔPP (mmHg)")
        ax.set_title("ΔPP by CVD outcome")
        plt.tight_layout()
        st.pyplot(fig)
        
    st.markdown(
        """
The CVD group shows a slightly higher median increase in pulse pressure and greater variability.
"""
    )
    # ΔPP by sex and CVD
    if "V1_SEX" in analytic_df.columns and "CVD" in analytic_df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(
            data=analytic_df,
            x="V1_SEX",
            y="DELTA_PP",
            hue="CVD",
            ax=ax,
            palette=[solid_color(3), solid_color(6)],
        )
        ax.set_xlabel("Sex at Visit 1 (0=male, 1=female)")
        ax.set_ylabel("ΔPP (mmHg)")
        ax.set_title("ΔPP by sex and CVD")
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown(
        """
This stays consistent if we stratify by gender. 
Independent of sex, participants who developed CVD show slightly larger increases in pulse pressure.
"""
    )

    # Correlation analysis
    st.markdown("---")
    st.subheader("Correlations (baseline + ΔPP)")

    # Select relevant numeric variables if present
    corr_vars = [c for c in [
        "DELTA_PP", "V1_AGE", "V1_BMI", "V1_SYSBP", "V1_DIABP",
        "V1_GLUCOSE", "V1_TOTCHOL", "V1_CIGPDAY"
    ] if c in analytic_df.columns]

    if len(corr_vars) >= 2:
        corr = analytic_df[corr_vars].corr().round(2)
        st.dataframe(corr, use_container_width=True)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, annot=True, center=0, ax=ax, cmap=CREST_CMAP)
        ax.set_title("Correlation matrix")
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown(
        """
This **correlation matrix** shows that ΔPP provides **added value**. 
While baseline systolic and diastolic BP are strongly correlated (as expected), ΔPP shows low correlations with baseline BP. 
This indicates that ΔPP captures an independent physiological change over time, rather than duplicating baseline information, making it a meaningful and informative feature for the model.
"""
    )

    # Interactive scatter exploration
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
            color_by = st.selectbox(
                "Color by",
                options=["None"] + analytic_df.columns.tolist(),
                index=(analytic_df.columns.tolist().index("CVD") + 1) if "CVD" in analytic_df.columns else 0
            )

        fig, ax = plt.subplots(figsize=(7, 5))
        if color_by != "None":
            sns.scatterplot(
                data=analytic_df,
                x=x_var,
                y=y_var,
                hue=color_by,
                alpha=0.6,
                ax=ax,
                palette="viridis",
            )
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        else:
            sns.scatterplot(
                data=analytic_df,
                x=x_var,
                y=y_var,
                alpha=0.6,
                ax=ax,
                color=solid_color(4),
            )

        ax.set_title(f"{x_var} vs {y_var}")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)


# --- Page 4: model comparison (palette-consistent)
def page_models_overview(model_results):
    st.header("Model comparison")

    if model_results is None or model_results.empty:
        st.error("model_results.csv not loaded or empty.")
        return

    # Sort by ROC AUC if available
    if "ROC_AUC" in model_results.columns:
        model_results_sorted = model_results.sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
    else:
        model_results_sorted = model_results.copy()

    st.subheader("Model comparison table (sorted by test ROC AUC)")
    st.dataframe(model_results_sorted, use_container_width=True)

    st.markdown("---")

    c1, c2 = st.columns(2)

    # Accuracy barplot
    with c1:
        st.subheader("Accuracy (test)")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=model_results_sorted, x="Accuracy", y="Model", ax=ax, color=solid_color(4))
        ax.set_xlim(0.45, 1.0)
        plt.tight_layout()
        st.pyplot(fig)

     # ROC AUC barplot
    with c2:
        st.subheader("ROC AUC (test)")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=model_results_sorted, x="ROC_AUC", y="Model", ax=ax, color=solid_color(4))
        ax.set_xlim(0.45, 0.85)
        plt.tight_layout()
        st.pyplot(fig)

     # Accuracy vs ROC AUC scatter
    st.markdown("---")

    if "Type" in model_results_sorted.columns:
        st.subheader("Accuracy vs ROC AUC (by model type)")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(
            data=model_results_sorted,
            x="ROC_AUC",
            y="Accuracy",
            hue="Type",
            style="Type",
            s=90,
            ax=ax,
            palette="viridis",
        )
        ax.set_xlabel("ROC AUC")
        ax.set_ylabel("Accuracy")
        ax.set_title("Trade-off: accuracy vs discrimination")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)


# --- Page 5: model detail + threshold + plots (palette-consistent)
def page_model_detail(all_model_outputs):
    st.header("Model detail + threshold explorer")

     # Select model to inspect
    model_name = st.selectbox("Select model", list(all_model_outputs.keys()))
    res = all_model_outputs[model_name]

    # Extract test labels and predicted probabilities
    y_test = np.asarray(res["y_test"]).astype(int)
    y_proba = np.asarray(res["y_proba"])

    # Stored base performance metrics
    st.subheader("Base performance (stored)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Accuracy (0.5)", f"{res.get('accuracy', np.nan):.3f}")
    with c2:
        st.metric("ROC AUC", f"{res.get('roc_auc', np.nan):.3f}")
    with c3:
        cv = res.get("cv_best_auc", np.nan)
        st.metric("CV Best AUC", f"{cv:.3f}" if pd.notna(cv) else "–")
    # ROC, PR, and calibration 
    st.markdown("---")
    st.subheader("ROC / PR / Calibration (test set)")

    c1, c2 = st.columns(2)
    with c1:
        st.write("ROC curve")
        plot_roc(y_test, y_proba)
    with c2:
        st.write("Precision–Recall curve")
        plot_pr(y_test, y_proba)
        
    # Calibration curve with adjustable binning
    st.markdown("---")
    bins = st.slider("Calibration bins", 5, 20, 10)
    plot_calibration(y_test, y_proba, bins=bins)

# --- Page 6: interaction test (ΔPP × sex)
def page_interaction_test(analytic_df):
    st.header("Subquestion 1: Interaction test (ΔPP × sex)")

    # Conceptual explanation of interaction analysis
    st.markdown(
        """
**Is the association between ΔPP and CVD different for women vs men?**

We tested whether the association between ΔPP and CVD differs by sex
using a logistic regression interaction model.

**Model form**  
`logit(P(CVD = 1)) = β0 + β1·ΔPP + β2·Sex + β3·(ΔPP × Sex) + controls`

The interaction term **β3** tests whether the ΔPP slope differs between women and men.  
This is an **association analysis**, not a causal claim.
"""
    )

    # Check availability of statsmodels for inference
    if not HAS_STATSMODELS:
        st.error("statsmodels is not installed in this environment, so p-values/robust SE cannot be computed here.")
        st.info("Fix: `pip install statsmodels` in your environment, or run this page locally where statsmodels exists.")
        return

    # Required predictors and potential confounders
    required = ["CVD", "DELTA_PP", "V1_SEX"]
    controls = ["V1_AGE", "V1_BMI", "V1_SYSBP", "V1_DIABP", "V1_GLUCOSE", "V1_TOTCHOL", "V1_CIGPDAY"]
    controls = [c for c in controls if c in analytic_df.columns]

    # Safety check for required variables
    missing = [c for c in required if c not in analytic_df.columns]
    if missing:
        st.error(f"Analytic dataset missing required columns: {missing}")
        return

    df = analytic_df[required + controls].dropna().copy()
    df["interaction"] = df["DELTA_PP"] * df["V1_SEX"]

    X = df[["DELTA_PP", "V1_SEX", "interaction"] + controls]
    X = sm.add_constant(X, has_constant="add")
    y = df["CVD"].astype(int)

    # --- Fit interaction model with robust SE
    try:
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

The models, which included ΔPP, achieved ROC AUC scores significantly higher than the baseline (0.5), reaching up to 0.744.
This indicates that ΔPP does contribute to some predictive signal, but overall performance is still moderate.
ΔPP contributes some predictive signal, but overall predictive performance is moderate.
A multivariable approach is more realistic than using ΔPP alone.

### Subquestion 1 (sex differences)
**Is the association between ΔPP and CVD different for women vs men?**

We used an interaction term (ΔPP × sex). 
Non-significant -> it suggested no statistical evidence for sex effect modification in our model.

### Overall conclusion
ΔPP is meaningful as a longitudinal vascular marker, but its practical value
is in combination with baseline covariates inside a risk model (not as a simple cut-off).
"""
    )

    # Highlight best-performing model
    if model_results is not None and not model_results.empty and "ROC_AUC" in model_results.columns:
        best = model_results.sort_values("ROC_AUC", ascending=False).iloc[0]
        st.markdown("---")
        st.subheader("Best model (based on test ROC AUC)")
        st.write(best)
        
        st.markdown(
            """
Although Neural Network (MLP, tuned) achieved the highest ROC AUC (0.7444), we would however prefer Logistic Regression (ROC AUC = 0.7443) over the Neural Network. That is because of its:
* **Interpretability**: Coefficients clearly show feature impact on the outcome.
* **Simplicity**: Easier to implement, train, and debug.
* **Data efficiency**: Performs well with less data, reducing overfitting risk.
* **Lower computational cost**: Requires fewer resources for training and deployment.

Neural Networks perform well with very complex patterns and large amounts of data, but their 'black box' nature and higher demands can be drawbacks.
"""
    )
        

# --- App entry point
st.set_page_config(page_title="Framingham ΔPP & CVD", layout="wide")

raw_df = load_raw_framingham()
analytic_df, model_results, all_model_outputs = load_project_outputs()

page = sidebar_block_and_nav()

if page == "1) Overview":
    page_overview(raw_df, analytic_df)
elif page == "2) Exploratory data analysis":
    page_eda_raw(raw_df)
elif page == "3) ΔPP & analytic exploration":
    page_delta_pp(analytic_df)
elif page == "4) Model comparison":
    page_models_overview(model_results)
elif page == "5) Model detail":
    page_model_detail(all_model_outputs)
elif page == "6) Interaction test (ΔPP × sex)":
    page_interaction_test(analytic_df)
elif page == "7) Final RQ recap":
    page_final_recap(model_results)
