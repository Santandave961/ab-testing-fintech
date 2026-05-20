"""
A/B Testing Framework — Streamlit App
Author : Wisdom (Santandave961)
Stack  : streamlit, pandas, numpy, scipy, matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import chi2_contingency, norm
import warnings

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="A/B Testing | Naija Fintech",
    page_icon=":test_tube:",
    layout="wide",
)

# ── Theme colours ─────────────────────────────────────────────
BG     = "#0F1117"
CARD   = "#1A1D27"
GRID   = "#2C2F3A"
GREEN  = "#2ECC71"
ORANGE = "#E67E22"
PURPLE = "#9B59B6"
RED    = "#E74C3C"

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #0F1117; }
[data-testid="stSidebar"]          { background-color: #1A1D27; }
[data-testid="stHeader"]           { background-color: #0F1117; }
h1, h2, h3, h4, p, li             { color: #FFFFFF !important; }
.metric-card {
    background: #1A1D27;
    border: 1px solid #2C2F3A;
    border-radius: 10px;
    padding: 18px 22px;
    text-align: center;
}
.metric-label { font-size: 13px; color: #888 !important; margin-bottom: 4px; }
.metric-value { font-size: 28px; font-weight: 700; color: #FFFFFF !important; }
.metric-delta { font-size: 13px; margin-top: 4px; }
.verdict-box {
    background: #1A1D27;
    border-radius: 12px;
    padding: 24px 28px;
    border-left: 5px solid #2ECC71;
    margin-top: 16px;
}
.section-divider { border-top: 1px solid #2C2F3A; margin: 28px 0; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def two_prop_ztest(conv_a, n_a, conv_b, n_b, alternative="larger"):
    p_a    = conv_a / n_a
    p_b    = conv_b / n_b
    p_pool = (conv_a + conv_b) / (n_a + n_b)
    se     = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
    z      = (p_b - p_a) / se
    if alternative == "larger":
        p = 1 - norm.cdf(z)
    elif alternative == "smaller":
        p = norm.cdf(z)
    else:
        p = 2 * (1 - norm.cdf(abs(z)))
    return z, p

def cohens_h(p1, p2):
    return 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))

def required_sample_size(p1, p2, alpha=0.05, power=0.80):
    h   = abs(cohens_h(p1, p2))
    if h == 0:
        return 999_999
    z_a = norm.ppf(1 - alpha)
    z_b = norm.ppf(power)
    return int(np.ceil(((z_a + z_b) / h) ** 2))

def achieved_power(p1, p2, n, alpha=0.05):
    h   = abs(cohens_h(p1, p2))
    z_a = norm.ppf(1 - alpha)
    return norm.cdf(h * np.sqrt(n) - z_a)

def label_h(h):
    h = abs(h)
    if h < 0.20: return "Negligible"
    if h < 0.50: return "Small"
    if h < 0.80: return "Medium"
    return "Large"

@st.cache_data
def simulate_data(n_control, n_treatment, cr_a, cr_b, seed):
    rng = np.random.default_rng(seed)
    ctrl = rng.binomial(1, cr_a, n_control)
    trt  = rng.binomial(1, cr_b, n_treatment)
    df   = pd.DataFrame({
        "user_id"   : range(1, n_control + n_treatment + 1),
        "group"     : (["Control"] * n_control) + (["Treatment"] * n_treatment),
        "converted" : np.concatenate([ctrl, trt]),
        "platform"  : rng.choice(["Android","iOS"],
                                  n_control + n_treatment, p=[0.62, 0.38]),
        "region"    : rng.choice(
                          ["Lagos","Abuja","Port Harcourt","Kano"],
                          n_control + n_treatment, p=[0.45, 0.25, 0.18, 0.12]),
    })
    return df


# ═══════════════════════════════════════════════════════════════
# SIDEBAR — EXPERIMENT PARAMETERS
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/test-tube.png", width=60)
    st.title("Experiment Config")
    st.markdown("---")

    st.subheader("Sample Size")
    n_control   = st.slider("Control users",   500, 10_000, 4_800, 100)
    n_treatment = st.slider("Treatment users", 500, 10_000, 4_950, 100)

    st.subheader("True Conversion Rates")
    cr_a = st.slider("Control CR (%)",   1, 99, 32) / 100
    cr_b = st.slider("Treatment CR (%)", 1, 99, 37) / 100

    st.subheader("Test Settings")
    alpha = st.selectbox("Significance level (α)", [0.05, 0.01, 0.10], index=0)
    seed  = st.number_input("Random seed", value=42, step=1)

    st.markdown("---")
    st.caption("Built by Wisdom · [@Santandave961](https://github.com/Santandave961)")


# ═══════════════════════════════════════════════════════════════
# DATA SIMULATION
# ═══════════════════════════════════════════════════════════════

df = simulate_data(n_control, n_treatment, cr_a, cr_b, int(seed))

summary = (
    df.groupby("group")["converted"]
    .agg(users="count", conversions="sum")
    .assign(conversion_rate=lambda x: x["conversions"] / x["users"])
)

platform_sum = (
    df.groupby(["group","platform"])["converted"]
    .agg(users="count", conversions="sum")
    .assign(cr=lambda x: (x["conversions"]/x["users"]).round(4))
    .reset_index()
)

region_sum = (
    df.groupby(["group","region"])["converted"]
    .agg(users="count", conversions="sum")
    .assign(cr=lambda x: (x["conversions"]/x["users"]).round(4))
    .reset_index()
)

conv_a = summary.loc["Control",   "conversions"]
conv_b = summary.loc["Treatment", "conversions"]
n_a    = summary.loc["Control",   "users"]
n_b    = summary.loc["Treatment", "users"]

obs_cr_a  = conv_a / n_a
obs_cr_b  = conv_b / n_b
lift      = obs_cr_b - obs_cr_a
rel_lift  = lift / obs_cr_a * 100

z_stat, p_value = two_prop_ztest(conv_a, n_a, conv_b, n_b, alternative="larger")
significant     = p_value < alpha

se_diff  = np.sqrt(obs_cr_a*(1-obs_cr_a)/n_a + obs_cr_b*(1-obs_cr_b)/n_b)
ci_lower = lift - norm.ppf(0.975) * se_diff
ci_upper = lift + norm.ppf(0.975) * se_diff

h         = cohens_h(obs_cr_a, obs_cr_b)
req_n     = required_sample_size(obs_cr_a, obs_cr_b, alpha=alpha, power=0.80)
ach_power = achieved_power(obs_cr_a, obs_cr_b, n=n_a, alpha=alpha)

chi2_stat, p_chi2, dof, _ = chi2_contingency(
    np.array([[conv_a, n_a-conv_a],[conv_b, n_b-conv_b]])
)


# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════

st.title("🧪 A/B Testing Framework")
st.markdown("#### Naija Fintech Onboarding — CTA Button Experiment")
st.markdown("""
| | Control (A) | Treatment (B) |
|---|---|---|
| **Button colour** | 🟢 Green | 🟠 Orange |
| **CTA text** | *Open Free Account* | *Start Saving Now* |
| **Hypothesis** | Baseline | CR_B > CR_A |
""")
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TOP KPI CARDS
# ═══════════════════════════════════════════════════════════════

c1, c2, c3, c4, c5 = st.columns(5)

def kpi(col, label, value, delta=None, delta_color=GREEN):
    delta_html = f'<div class="metric-delta" style="color:{delta_color}">{delta}</div>' if delta else ""
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)

kpi(c1, "Control CR",   f"{obs_cr_a*100:.2f}%")
kpi(c2, "Treatment CR", f"{obs_cr_b*100:.2f}%",
    delta=f"{lift*100:+.2f}pp vs control",
    delta_color=GREEN if lift > 0 else RED)
kpi(c3, "Z-Statistic",  f"{z_stat:.3f}")
kpi(c4, "P-Value",      f"{p_value:.5f}",
    delta="✅ Significant" if significant else "❌ Not Significant",
    delta_color=GREEN if significant else RED)
kpi(c5, "Achieved Power", f"{ach_power*100:.1f}%")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# CHARTS ROW 1
# ═══════════════════════════════════════════════════════════════

st.subheader("📊 Visualisations")

def style_ax(ax):
    ax.set_facecolor(CARD)
    ax.tick_params(colors="white", labelsize=9)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)

col1, col2, col3 = st.columns(3)

# ── Chart 1: Conversion Rates ─────────────────────────────────
with col1:
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor(BG)
    style_ax(ax)
    bars = ax.bar(["Control\n(Green)", "Treatment\n(Orange)"],
                  [obs_cr_a*100, obs_cr_b*100],
                  color=[GREEN, ORANGE], width=0.5, edgecolor="white", lw=0.5)
    for bar, rate in zip(bars, [obs_cr_a*100, obs_cr_b*100]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f"{rate:.2f}%", ha="center", color="white",
                fontweight="bold", fontsize=11)
    ax.set_title("Conversion Rate by Group")
    ax.set_ylabel("Conversion Rate (%)")
    ax.set_ylim(0, max(obs_cr_a, obs_cr_b)*100 * 1.3)
    ax.yaxis.grid(True, color=GRID, alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Chart 2: Lift + CI ────────────────────────────────────────
with col2:
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor(BG)
    style_ax(ax)
    ax.barh(["Lift (B − A)"], [lift*100], color=PURPLE,
            xerr=[[(lift-ci_lower)*100], [(ci_upper-lift)*100]],
            error_kw=dict(ecolor="white", capsize=10, lw=2))
    ax.axvline(0, color="white", lw=1, ls="--")
    ax.set_title("Lift & 95% Confidence Interval")
    ax.set_xlabel("Percentage points")
    ax.text(lift*100, 0, f"  {lift*100:+.2f}pp",
            color="white", va="center", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Chart 3: P-value Gauge ────────────────────────────────────
with col3:
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)
    theta = np.linspace(np.pi, 0, 500)
    ax.plot(np.cos(theta), np.sin(theta), lw=14, color=GRID)
    alpha_ang = np.pi * alpha
    theta_sig = np.linspace(alpha_ang, 0, 200)
    ax.plot(np.cos(theta_sig), np.sin(theta_sig), lw=14, color=RED, alpha=0.8)
    p_clamp = min(p_value, 0.999)
    p_ang   = np.pi * p_clamp
    ax.annotate("", xy=(0.72*np.cos(p_ang), 0.72*np.sin(p_ang)),
                xytext=(0,0),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=2.5))
    ax.scatter([0],[0], s=60, color="white", zorder=5)
    ax.text(0, -0.22, f"p = {p_value:.5f}", ha="center",
            color="white", fontsize=11, fontweight="bold")
    ax.text(0, -0.42, "✅ Significant!" if significant else "❌ Not Significant",
            ha="center",
            color=GREEN if significant else RED,
            fontsize=10, fontweight="bold")
    ax.text(-1.08, -0.08, "p=1", color="white", fontsize=8)
    ax.text(0.88,  -0.08, "p=0", color="white", fontsize=8)
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-0.6, 1.15)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("P-value Gauge", color="white", pad=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Charts Row 2 ──────────────────────────────────────────────
col4, col5, col6 = st.columns(3)

# ── Chart 4: By Platform ──────────────────────────────────────
with col4:
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor(BG)
    style_ax(ax)
    pp = platform_sum.pivot(index="platform", columns="group", values="cr") * 100
    for col_name in ["Control","Treatment"]:
        if col_name not in pp.columns:
            pp[col_name] = 0
    pp[["Control","Treatment"]].plot(kind="bar", ax=ax,
        color=[GREEN, ORANGE], edgecolor="white", lw=0.5)
    ax.set_title("Conversion by Platform")
    ax.set_ylabel("Conversion Rate (%)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(["Control","Treatment"],
              facecolor=CARD, edgecolor=GRID, labelcolor="white")
    ax.yaxis.grid(True, color=GRID, alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Chart 5: By Region ────────────────────────────────────────
with col5:
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor(BG)
    style_ax(ax)
    rp = region_sum.pivot(index="region", columns="group", values="cr") * 100
    for col_name in ["Control","Treatment"]:
        if col_name not in rp.columns:
            rp[col_name] = 0
    rp[["Control","Treatment"]].plot(kind="bar", ax=ax,
        color=[GREEN, ORANGE], edgecolor="white", lw=0.5)
    ax.set_title("Conversion by Region")
    ax.set_ylabel("Conversion Rate (%)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(["Control","Treatment"],
              facecolor=CARD, edgecolor=GRID, labelcolor="white")
    ax.yaxis.grid(True, color=GRID, alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Chart 6: Power Curve ──────────────────────────────────────
with col6:
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor(BG)
    style_ax(ax)
    sample_sizes = np.arange(200, 10001, 100)
    powers = [achieved_power(obs_cr_a, obs_cr_b, n=n, alpha=alpha)
              for n in sample_sizes]
    ax.plot(sample_sizes, [p*100 for p in powers], color=PURPLE, lw=2)
    ax.axhline(80, color=ORANGE, ls="--", lw=1.5, label="80% target")
    ax.axvline(req_n, color=GREEN, ls="--", lw=1.5,
               label=f"Required n={req_n:,}")
    ax.axvline(n_a, color="white", ls=":", lw=1.5,
               label=f"Actual n={n_a:,}")
    ax.set_title("Power Curve vs Sample Size")
    ax.set_xlabel("n per arm")
    ax.set_ylabel("Power (%)")
    ax.legend(facecolor=CARD, edgecolor=GRID,
              labelcolor="white", fontsize=8)
    ax.yaxis.grid(True, color=GRID, alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# STATISTICAL RESULTS TABLE
# ═══════════════════════════════════════════════════════════════

st.subheader("📋 Statistical Results")

left, right = st.columns(2)

with left:
    st.markdown("**Z-Test (Primary)**")
    results_df = pd.DataFrame({
        "Metric": [
            "Control CR", "Treatment CR",
            "Absolute Lift", "Relative Lift",
            "Z-Statistic", "P-Value (one-sided)",
            "95% CI Lower", "95% CI Upper",
            "Significant?"
        ],
        "Value": [
            f"{obs_cr_a*100:.2f}%",
            f"{obs_cr_b*100:.2f}%",
            f"{lift*100:+.2f}pp",
            f"{rel_lift:+.1f}%",
            f"{z_stat:.4f}",
            f"{p_value:.6f}",
            f"{ci_lower*100:+.2f}pp",
            f"{ci_upper*100:+.2f}pp",
            "✅ Yes" if significant else "❌ No"
        ]
    })
    st.dataframe(results_df, hide_index=True, use_container_width=True)

with right:
    st.markdown("**Chi-Square (Robustness Check)**")
    chi_df = pd.DataFrame({
        "Metric": [
            "Chi² Statistic", "P-Value", "Degrees of Freedom",
            "Significant?"
        ],
        "Value": [
            f"{chi2_stat:.4f}",
            f"{p_chi2:.6f}",
            str(dof),
            "✅ Yes" if p_chi2 < alpha else "❌ No"
        ]
    })
    st.dataframe(chi_df, hide_index=True, use_container_width=True)

    st.markdown("**Effect Size & Power**")
    power_df = pd.DataFrame({
        "Metric": [
            "Cohen's h", "Effect Label",
            "Required n/arm (80% power)",
            "Achieved Power"
        ],
        "Value": [
            f"{h:.4f}",
            label_h(h),
            f"{req_n:,}",
            f"{ach_power*100:.1f}%"
        ]
    })
    st.dataframe(power_df, hide_index=True, use_container_width=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# EDA TABLES
# ═══════════════════════════════════════════════════════════════

with st.expander("🔍 Raw Data & EDA Tables", expanded=False):
    t1, t2, t3 = st.tabs(["Summary", "By Platform", "By Region"])
    with t1:
        st.dataframe(summary.reset_index().round(4), use_container_width=True)
    with t2:
        st.dataframe(platform_sum.round(4), use_container_width=True)
    with t3:
        st.dataframe(region_sum.round(4), use_container_width=True)

    st.markdown("**Sample rows from simulated dataset**")
    st.dataframe(df.sample(10, random_state=1).reset_index(drop=True),
                 use_container_width=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# BUSINESS CONCLUSION
# ═══════════════════════════════════════════════════════════════

st.subheader("💡 Business Conclusion")

verdict_color = "#2ECC71" if significant else "#E74C3C"
verdict_text  = "🚀 SHIP IT — Roll out to 100% of users" if significant else "🚫 DO NOT SHIP — No significant difference detected"

st.markdown(f"""
<div class="verdict-box" style="border-left: 5px solid {verdict_color}">
    <h3 style="color:{verdict_color} !important; margin-top:0">{verdict_text}</h3>
    <p>
        The orange <em>"Start Saving Now"</em> button (Treatment B) achieved a conversion
        rate of <strong>{obs_cr_b*100:.2f}%</strong>, compared to
        <strong>{obs_cr_a*100:.2f}%</strong> for the green control —
        a <strong>{lift*100:+.2f}pp absolute lift</strong> ({rel_lift:+.1f}% relative).
    </p>
    <p>
        The result is {'<strong>statistically significant</strong>' if significant else '<strong>not statistically significant</strong>'}
        (p = {p_value:.5f}, α = {alpha}).
        The 95% confidence interval for the lift is
        <strong>[{ci_lower*100:+.2f}pp, {ci_upper*100:+.2f}pp]</strong>
        {'— entirely above zero, confirming a genuine positive effect.' if ci_lower > 0 else '— crosses zero, meaning the true lift could be neutral or negative.'}
    </p>
    <p>
        <strong>Estimated impact:</strong> ~<strong>{int(lift*10_000):,} additional onboardings</strong>
        per 10,000 future visitors.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.caption("Built by **Wisdom** · Data Science Portfolio · "
           "[@Santandave961](https://github.com/Santandave961) · "
           "Targeting Nigerian Fintech Roles 🇳🇬")