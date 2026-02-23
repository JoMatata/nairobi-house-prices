import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Nairobi House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ── Load model and metadata ───────────────────────────────
@st.cache_resource
def load_model():
    with open('data/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('data/encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('data/model_metadata.json', 'r') as f:
        metadata = json.load(f)
    return model, encoders, metadata

model, encoders, metadata = load_model()

# ── Header ────────────────────────────────────────────────
st.title(" Nairobi House Price Predictor")
st.markdown("Enter property details below to get an instant price estimate.")
st.divider()

# ── Input form ────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    location = st.selectbox(
        " Location",
        options=sorted(metadata['locations'])
    )

    property_type = st.selectbox(
        " Property Type",
        options=sorted(metadata['property_types'])
    )

    bedrooms = st.slider(" Bedrooms", min_value=1, max_value=6, value=3)

with col2:
    bathrooms = st.slider(" Bathrooms", min_value=1, max_value=5, value=2)

    parking = st.slider(" Parking Spaces", min_value=0, max_value=4, value=1)

    size_sqft = st.number_input(
        " Size (sqft)",
        min_value=200,
        max_value=10000,
        value=1200,
        step=50
    )

st.divider()

# ── Predict button ────────────────────────────────────────
if st.button(" Predict Price", use_container_width=True, type="primary"):

    high_end = ['Karen', 'Runda', 'Muthaiga', 'Gigiri', 'Lavington',
                'Westlands', 'Riverside', 'Parklands']
    mid_range = ['Kilimani', 'Kileleshwa', 'Ngong Road', 'South B',
                 'South C', 'Upperhill', 'Syokimau']

    if location in high_end:
        tier = 'High End'
    elif location in mid_range:
        tier = 'Mid Range'
    else:
        tier = 'Affordable'

    try:
        location_encoded = encoders['location'].transform([location])[0]
    except:
        location_encoded = 0

    try:
        type_encoded = encoders['property_type'].transform([property_type])[0]
    except:
        type_encoded = 0

    try:
        tier_encoded = encoders['tier'].transform([tier])[0]
    except:
        tier_encoded = 0

    total_rooms = bedrooms + bathrooms
    listing_month = pd.Timestamp.today().month

    input_data = np.array([[
        bedrooms,
        bathrooms,
        size_sqft,
        parking,
        total_rooms,
        location_encoded,
        type_encoded,
        tier_encoded,
        listing_month
    ]])

    predicted_price = model.predict(input_data)[0]
    mae = metadata['mae']

    low_estimate = predicted_price - mae
    high_estimate = predicted_price + mae

    # ── Display results ───────────────────────────────────
    st.success(" Prediction Complete!")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.metric(
            label="Low Estimate",
            value=f"KES {low_estimate/1e6:.2f}M"
        )
    with col_b:
        st.metric(
            label=" Predicted Price",
            value=f"KES {predicted_price/1e6:.2f}M",
            delta="Best estimate"
        )
    with col_c:
        st.metric(
            label="High Estimate",
            value=f"KES {high_estimate/1e6:.2f}M"
        )

    st.divider()

    # ── Explanation ───────────────────────────────────────
    st.subheader(" What's driving this price?")

    drivers = {
        "Property Size": f"{size_sqft:,} sqft — the single biggest price factor (51.7% of model decisions)",
        "Bathrooms": f"{bathrooms} bathroom(s) — premium signal for Nairobi buyers (23.4%)",
        "Total Rooms": f"{total_rooms} rooms combined — strong value indicator (19.7%)",
        "Location": f"{location} ({tier} tier)",
        "Property Type": property_type
    }

    for driver, explanation in drivers.items():
        st.markdown(f"**{driver}:** {explanation}")

    st.divider()

    # ── Model confidence ──────────────────────────────────
    st.caption(
        f"Model accuracy: R² = {metadata['r2']} | "
        f"Average error: ±KES {mae/1e6:.2f}M | "
        f"Trained on 396 real Nairobi listings from Property24"
    )