import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Beta Distribution Explorer",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 Beta Distribution Statistical Properties Explorer")
st.markdown("""
The **Beta distribution** is a continuous probability distribution defined on the interval [0,1] and parameterized by two positive shape parameters α (alpha) and β (beta). 
It's extremely versatile and appears in many applications including Bayesian statistics, project management, and modeling proportions.
""")

# Initialize session state
if 'alpha' not in st.session_state:
    st.session_state.alpha = 2.0
if 'beta' not in st.session_state:
    st.session_state.beta = 5.0

# Sidebar for parameters
st.sidebar.header("Distribution Parameters")
alpha = st.sidebar.slider("α (Alpha)", min_value=0.1, max_value=10.0, value=st.session_state.alpha, step=0.1, key="alpha_slider")
beta = st.sidebar.slider("β (Beta)", min_value=0.1, max_value=10.0, value=st.session_state.beta, step=0.1, key="beta_slider")

# Update session state when sliders change
st.session_state.alpha = alpha
st.session_state.beta = beta

# Sample size for simulation
sample_size = st.sidebar.slider("Sample Size", min_value=100, max_value=10000, value=1000, step=100)

# Generate data
x = np.linspace(0, 1, 1000)
rv = stats.beta(alpha, beta)
pdf = rv.pdf(x)
cdf = rv.cdf(x)

# Generate random samples
samples = rv.rvs(size=sample_size, random_state=42)

# Calculate statistics
mean = rv.mean()
variance = rv.var()
std = rv.std()
mode = (alpha - 1) / (alpha + beta - 2) if alpha > 1 and beta > 1 else None
median = rv.median()
skewness = rv.stats(moments='s')
kurtosis = rv.stats(moments='k')

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Probability Density Function', 'Cumulative Distribution Function', 
                       'Sample Histogram', 'Q-Q Plot'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # PDF
    fig.add_trace(
        go.Scatter(x=x, y=pdf, mode='lines', name='PDF', line=dict(color='blue', width=3)),
        row=1, col=1
    )
    
    # CDF  
    fig.add_trace(
        go.Scatter(x=x, y=cdf, mode='lines', name='CDF', line=dict(color='red', width=3)),
        row=1, col=2
    )
    
    # Histogram of samples
    fig.add_trace(
        go.Histogram(x=samples, nbinsx=50, name='Sample Histogram', 
                    marker_color='lightblue', opacity=0.7, histnorm='probability density'),
        row=2, col=1
    )
    
    # Add theoretical PDF overlay on histogram
    fig.add_trace(
        go.Scatter(x=x, y=pdf, mode='lines', name='Theoretical PDF', 
                  line=dict(color='blue', width=2, dash='dash')),
        row=2, col=1
    )
    
    # Q-Q plot
    theoretical_quantiles = np.linspace(0.01, 0.99, len(samples))
    sample_quantiles = np.sort(samples)
    theoretical_values = rv.ppf(theoretical_quantiles)
    
    fig.add_trace(
        go.Scatter(x=theoretical_values, y=sample_quantiles, mode='markers', 
                  name='Q-Q Plot', marker=dict(color='green', size=4)),
        row=2, col=2
    )
    
    # Add reference line for Q-Q plot
    min_val, max_val = min(theoretical_values.min(), sample_quantiles.min()), max(theoretical_values.max(), sample_quantiles.max())
    fig.add_trace(
        go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', 
                  name='Reference Line', line=dict(color='red', dash='dash')),
        row=2, col=2
    )
    
    fig.update_layout(height=700, showlegend=False, title_text="Beta Distribution Visualization")
    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_xaxes(title_text="x", row=1, col=2)
    fig.update_xaxes(title_text="x", row=2, col=1)
    fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2)
    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_yaxes(title_text="Probability", row=1, col=2)
    fig.update_yaxes(title_text="Density", row=2, col=1)
    fig.update_yaxes(title_text="Sample Quantiles", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interactive exploration buttons - right under the visualizations
    st.subheader("🔍 Try These Interesting Cases:")
    col1_btn, col2_btn, col3_btn, col4_btn = st.columns(4)

    with col1_btn:
        if st.button("Uniform (1,1)", key="uniform_btn"):
            st.session_state.alpha = 1.0
            st.session_state.beta = 1.0
            st.rerun()

    with col2_btn:
        if st.button("U-Shape (0.5,0.5)", key="ushape_btn"):
            st.session_state.alpha = 0.5
            st.session_state.beta = 0.5
            st.rerun()

    with col3_btn:
        if st.button("Right Skewed (2,5)", key="right_skew_btn"):
            st.session_state.alpha = 2.0
            st.session_state.beta = 5.0
            st.rerun()

    with col4_btn:
        if st.button("Left Skewed (5,2)", key="left_skew_btn"):
            st.session_state.alpha = 5.0
            st.session_state.beta = 2.0
            st.rerun()

with col2:
    st.subheader("📈 Statistical Properties")
    
    # Display current parameters
    st.metric("Alpha (α)", f"{alpha:.2f}")
    st.metric("Beta (β)", f"{beta:.2f}")
    
    st.subheader("📊 Descriptive Statistics")
    st.metric("Mean", f"{mean:.4f}")
    st.metric("Median", f"{median:.4f}")
    if mode is not None:
        st.metric("Mode", f"{mode:.4f}")
    else:
        st.metric("Mode", "Undefined")
    
    st.metric("Variance", f"{variance:.4f}")
    st.metric("Std Deviation", f"{std:.4f}")
    st.metric("Skewness", f"{skewness:.4f}")
    st.metric("Kurtosis", f"{kurtosis:.4f}")

# Educational section
st.markdown("---")
st.header("🎓 Understanding the Beta Distribution")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Shape Behavior")
    st.markdown("""
    **Effect of Parameters:**
    - **α = β = 1**: Uniform distribution
    - **α < 1, β < 1**: U-shaped (bimodal tendency)
    - **α > 1, β > 1**: Unimodal, bell-shaped
    - **α = β**: Symmetric around 0.5
    - **α > β**: Skewed left (toward 1)
    - **α < β**: Skewed right (toward 0)
    """)

with col2:
    st.subheader("Key Formulas")
    st.markdown("""
    **Mean**: μ = α/(α + β)
    
    **Variance**: σ² = αβ/[(α + β)²(α + β + 1)]
    
    **Mode**: (α - 1)/(α + β - 2) (if α,β > 1)
    
    **PDF**: f(x) = x^(α-1)(1-x)^(β-1)/B(α,β)
    """)

with col3:
    st.subheader("Applications")
    st.markdown("""
    - **Bayesian Statistics**: Conjugate prior for binomial
    - **Project Management**: PERT distributions
    - **Quality Control**: Proportion modeling
    - **Finance**: Risk modeling
    - **A/B Testing**: Conversion rate analysis
    - **Machine Learning**: Probability calibration
    """)

# Interactive exploration section
st.markdown("---")
st.header("🔍 Interactive Exploration")

# Common parameter combinations
st.subheader("Try These Interesting Cases:")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Uniform (1,1)"):
        st.session_state.alpha = 1.0
        st.session_state.beta = 1.0
        st.rerun()

with col2:
    if st.button("U-Shape (0.5,0.5)"):
        st.session_state.alpha = 0.5
        st.session_state.beta = 0.5
        st.rerun()

with col3:
    if st.button("Right Skewed (2,5)"):
        st.session_state.alpha = 2.0
        st.session_state.beta = 5.0
        st.rerun()

with col4:
    if st.button("Left Skewed (5,2)"):
        st.session_state.alpha = 5.0
        st.session_state.beta = 2.0
        st.rerun()

# Statistical tests
st.markdown("---")
st.header("📋 Statistical Tests")

# Goodness of fit test
ks_stat, ks_p_value = stats.kstest(samples, rv.cdf)
ad_stat, ad_critical_values, ad_p_value = stats.anderson(samples, 'norm')

col1, col2 = st.columns(2)

with col1:
    st.subheader("Kolmogorov-Smirnov Test")
    st.write(f"**Test Statistic:** {ks_stat:.4f}")
    st.write(f"**P-value:** {ks_p_value:.4f}")
    if ks_p_value > 0.05:
        st.success("✅ Samples appear to follow the theoretical distribution")
    else:
        st.warning("⚠️ Samples may not follow the theoretical distribution")

with col2:
    st.subheader("Sample Summary")
    st.write(f"**Sample Size:** {sample_size}")
    st.write(f"**Sample Mean:** {np.mean(samples):.4f}")
    st.write(f"**Sample Std:** {np.std(samples, ddof=1):.4f}")
    st.write(f"**Min Value:** {np.min(samples):.4f}")
    st.write(f"**Max Value:** {np.max(samples):.4f}")

# Footer
st.markdown("---")
st.markdown("""
*This interactive dashboard demonstrates key statistical properties of the Beta distribution. 
Adjust the parameters to explore how α and β affect the shape, central tendency, and spread of the distribution.*
""")
