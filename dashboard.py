import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="Nairobi Property Market Dashboard",
    layout="wide"
)

# ── Load data ─────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv('data/clean_listings.csv')

df = load_data()

# ── Header ────────────────────────────────────────────────
st.title(" Nairobi Property Market Intelligence Dashboard")
st.markdown("Real market insights derived from 396 live Property24 listings.")
st.divider()

# ── Top KPI metrics ───────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Listings", f"{len(df):,}")
with col2:
    st.metric("Median Price", f"KES {df['price_kes'].median()/1e6:.1f}M")
with col3:
    st.metric("Avg Price/sqft", f"KES {df['price_per_sqft'].mean():,.0f}")
with col4:
    st.metric("Locations Covered", f"{df['location'].nunique()}")

st.divider()

# ── Page 1: Median price by location ─────────────────────
st.subheader(" Median Price by Location")

location_data = (df.groupby('location')['price_kes']
                 .median()
                 .sort_values(ascending=False)
                 .head(12) / 1e6)

fig1, ax1 = plt.subplots(figsize=(12, 5))
bars = ax1.bar(location_data.index, location_data.values,
               color='steelblue', edgecolor='black', linewidth=0.5)

# Add value labels on top of each bar
for bar, val in zip(bars, location_data.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1f}M', ha='center', va='bottom', fontsize=9)

ax1.set_ylabel('Median Price (KES Millions)')
ax1.set_xlabel('Location')
ax1.set_title('Top 12 Nairobi Locations by Median Property Price')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig1)

insight1 = (f"**Key Insight:** {location_data.index[0]} leads at "
            f"KES {location_data.values[0]:.1f}M median, "
            f"{location_data.values[0]/location_data.values[-1]:.1f}x more expensive "
            f"than {location_data.index[-1]} at KES {location_data.values[-1]:.1f}M.")
st.info(insight1)
st.divider()

# ── Page 2: Price per sqft comparison ────────────────────
st.subheader(" Price Per Sqft by Location")

sqft_data = (df.groupby('location')['price_per_sqft']
             .mean()
             .sort_values(ascending=False)
             .head(12))

fig2, ax2 = plt.subplots(figsize=(12, 5))
colors = ['gold' if i < 3 else 'steelblue' for i in range(len(sqft_data))]
ax2.barh(sqft_data.index, sqft_data.values, color=colors, edgecolor='black')
ax2.set_xlabel('Average Price per sqft (KES)')
ax2.set_title('Price Per Sqft by Location — Top 12')
ax2.invert_yaxis()

for i, (val, name) in enumerate(zip(sqft_data.values, sqft_data.index)):
    ax2.text(val + 100, i, f'KES {val:,.0f}', va='center', fontsize=9)

plt.tight_layout()
st.pyplot(fig2)

st.info(f"**Key Insight:** Gold bars represent the top 3 highest value-per-sqft locations — "
        f"where buyers pay the most per square foot.")
st.divider()

# ── Page 3: Price by bedroom count ───────────────────────
st.subheader(" How Bedrooms Affect Price")

col_left, col_right = st.columns(2)

with col_left:
    bedroom_data = df.groupby('bedrooms')['price_kes'].median() / 1e6
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.bar(bedroom_data.index, bedroom_data.values,
            color='mediumseagreen', edgecolor='black')
    ax3.set_xlabel('Number of Bedrooms')
    ax3.set_ylabel('Median Price (KES Millions)')
    ax3.set_title('Median Price by Bedroom Count')
    for i, val in zip(bedroom_data.index, bedroom_data.values):
        ax3.text(i, val + 0.2, f'{val:.1f}M', ha='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig3)

with col_right:
    # Price distribution by property type
    type_data = df.groupby('property_type')['price_kes'].median() / 1e6
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    ax4.bar(type_data.index, type_data.values,
            color='coral', edgecolor='black')
    ax4.set_xlabel('Property Type')
    ax4.set_ylabel('Median Price (KES Millions)')
    ax4.set_title('Median Price by Property Type')
    plt.xticks(rotation=30, ha='right')
    for i, (name, val) in enumerate(zip(type_data.index, type_data.values)):
        ax4.text(i, val + 0.2, f'{val:.1f}M', ha='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig4)

st.divider()

# ── Page 4: Location tier analysis ───────────────────────
st.subheader(" Location Tier Analysis")

col_t1, col_t2 = st.columns(2)

with col_t1:
    tier_order = ['High End', 'Mid Range', 'Affordable']
    tier_stats = df.groupby('location_tier').agg(
        median_price=('price_kes', 'median'),
        count=('price_kes', 'count'),
        avg_size=('size_sqft', 'mean')
    ).reindex(tier_order)

    tier_stats['median_price_M'] = tier_stats['median_price'] / 1e6

    fig5, ax5 = plt.subplots(figsize=(6, 4))
    bars = ax5.bar(tier_stats.index, tier_stats['median_price_M'],
                   color=['gold', 'skyblue', 'lightcoral'], edgecolor='black')
    ax5.set_ylabel('Median Price (KES Millions)')
    ax5.set_title('Median Price by Location Tier')
    for bar, val in zip(bars, tier_stats['median_price_M']):
        ax5.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.2,
                 f'{val:.1f}M', ha='center', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig5)

with col_t2:
    st.markdown("### Tier Summary")
    for tier in tier_order:
        row = tier_stats.loc[tier]
        st.markdown(
            f"**{tier}**  \n"
            f"Median Price: KES {row['median_price']/1e6:.1f}M  \n"
            f"Listings: {int(row['count'])}  \n"
            f"Avg Size: {row['avg_size']:,.0f} sqft"
        )
        st.markdown

st.divider()

# ── Page 5: Model performance summary ────────────────────
st.subheader(" Model Performance Summary")

results = pd.read_csv('data/model_results.csv')

col_m1, col_m2 = st.columns(2)

with col_m1:
    st.dataframe(results, use_container_width=True)

with col_m2:
    fig6, ax6 = plt.subplots(figsize=(6, 4))
    models = results['model']
    mae_values = results['MAE_kes'] / 1e6
    bars = ax6.bar(models, mae_values,
                   color=['salmon', 'steelblue'], edgecolor='black')
    ax6.set_ylabel('MAE (KES Millions)')
    ax6.set_title('Model Comparison — Lower is Better')
    for bar, val in zip(bars, mae_values):
        ax6.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.02,
                 f'{val:.2f}M', ha='center', fontsize=10)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    st.pyplot(fig6)

improvement = ((results['MAE_kes'][0] - results['MAE_kes'][1]) /
               results['MAE_kes'][0] * 100)
st.success(
    f" Random Forest outperforms baseline by {improvement:.1f}% — "
    f"predicting Nairobi property prices with R² = {results['R2'][1]}"
)

st.divider()
st.caption("Built by JoMatata | Data source: Property24 Kenya | "
           "LTLab Data Engineering Fellowship 2026")