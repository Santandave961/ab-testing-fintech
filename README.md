# 🧪 A/B Testing Framework — Naija Fintech Onboarding

A complete, end-to-end A/B testing project built in Python, framed around a realistic Nigerian fintech product scenario. Covers the full experiment lifecycle — from data simulation and statistical testing to effect size, power analysis, and a business recommendation.

![Dashboard](ab_test_dashboard.png)

---

## 📌 Business Problem

A Nigerian fintech app (modelled after Kuda / PiggyVest) wants to increase onboarding conversion. The product team hypothesises that changing the CTA button copy and colour will drive more sign-ups.

| Variant | Button Colour | CTA Text |
|---|---|---|
| **Control (A)** | 🟢 Green | *"Open Free Account"* |
| **Treatment (B)** | 🟠 Orange | *"Start Saving Now"* |

**Primary metric:** Onboarding completion rate (did the user finish sign-up?)

---

## 🔬 Methods

| Step | Technique |
|---|---|
| Data simulation | `numpy.random.binomial` with realistic conversion rates |
| Exploratory analysis | Conversion breakdown by group, platform, and region |
| Hypothesis test | Two-proportion Z-test (one-sided: H₁: CR_B > CR_A) |
| Robustness check | Pearson Chi-Square test |
| Effect size | Cohen's h |
| Power analysis | Achieved power + required sample size (pure scipy) |
| Visualisation | 6-panel dark-themed Matplotlib dashboard |

---

## 📊 Key Results

| Metric | Value |
|---|---|
| Control conversion rate | 31.25% |
| Treatment conversion rate | 35.05% |
| Absolute lift | **+3.80 percentage points** |
| Relative lift | **+12.2%** |
| Z-statistic | 3.98 |
| P-value (one-sided) | **0.000034** |
| Significance (α = 0.05) | ✅ Yes |
| 95% CI for lift | [+1.93pp, +5.67pp] |
| Cohen's h | 0.081 |
| Achieved power | 100% |

> **Verdict: 🚀 Ship it.** The orange CTA is significantly better.  
> Estimated **+380 additional onboardings per 10,000 visitors**.

---

## 🗂️ Project Structure

```
ab-testing-project/
│
├── ab_testing_project.py   # Main script — simulation, stats, plots
├── ab_test_dashboard.png   # Output dashboard (auto-generated)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## ⚙️ Setup & Usage

**1. Clone the repo**
```bash
git clone https://github.com/Santandave961/ab-testing-project.git
cd ab-testing-project
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the project**
```bash
python ab_testing_project.py
```

The script will print all statistical output to the terminal and save `ab_test_dashboard.png` in the same directory.

---

## 📦 Requirements

```
pandas
numpy
scipy
matplotlib
seaborn
```

> No statsmodels required — power and sample size calculations are implemented from scratch using `scipy.stats.norm`.

---

## 🧠 Concepts Demonstrated

- **Null & alternative hypothesis** formulation for a one-sided test
- **Pooled proportion Z-test** implemented without external stats libraries
- **Chi-Square contingency test** as an independent robustness check
- **Cohen's h** for standardised effect size on proportions
- **Power curve** — visualising how power scales with sample size
- **Confidence intervals** for the difference in proportions
- **Segmentation analysis** — conversion broken down by platform (Android/iOS) and Nigerian region (Lagos, Abuja, Port Harcourt, Kano)

---

## 📈 Dashboard Preview

The auto-generated dashboard contains 6 panels:

1. **Conversion Rate by Group** — side-by-side bar chart
2. **Lift & 95% CI** — horizontal bar with error bars
3. **Conversion by Platform** — grouped bars (Android vs iOS)
4. **Conversion by Region** — grouped bars across 4 Nigerian cities
5. **Power Curve** — statistical power vs sample size
6. **P-value Gauge** — visual significance indicator

---

## 💡 Business Context

This project is designed to reflect the kind of data-driven decision-making used at Nigerian fintech companies like **Kuda**, **Moniepoint**, **PiggyVest**, **Flutterwave**, and **Carbon** — where product, growth, and data science teams run continuous experiments to improve conversion funnels, retention, and user activation.

---

## 👤 Author

**Wisdom** — Data Science & ML Engineer  
📌 NYSC Corper | Abia State, Nigeria  
🐙 GitHub: [@Santandave961](https://github.com/Santandave961)  
🐦 X: [@Santandave961](https://x.com/Santandave961)

---

## 📄 License

MIT License — free to use, adapt, and build on.
