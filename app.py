import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.main-header {
    font-size: 40px;
    font-weight: bold;
    color: #1f4e79;
}

.sub-header {
    font-size: 20px;
    color: #555;
}

div[data-testid="metric-container"] {
    background-color: #f7f9fc;
    border: 1px solid #ddd;
    padding: 15px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# ---------------- TITLE ----------------

st.markdown(
    '<p class="main-header">📊 Customer Churn Analytics Dashboard</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-header">Machine Learning powered customer retention analysis</p>',
    unsafe_allow_html=True
)


# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    df = pd.read_csv("data/Telco_customer_Churn1.csv")
    return df


df = load_data()


# ---------------- LOAD MODEL ----------------

model = joblib.load("random_forest_model.joblib")


# ---------------- SIDEBAR ----------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "Customer Analysis",
        "Churn Drivers",
        "Prediction"
    ]
)


# ==================================================
# DASHBOARD PAGE
# ==================================================

if page == "Dashboard":

    st.subheader("Business Overview")


    total_customer = len(df)

    churn_rate = (
        df["Churn"]
        .value_counts(normalize=True)
        .get("Yes",0)
        *100
    )


    avg_charge = df["MonthlyCharges"].mean()


    churn_customer = len(
        df[df["Churn"]=="Yes"]
    )


    col1,col2,col3,col4 = st.columns(4)


    col1.metric(
        "Total Customers",
        total_customer
    )


    col2.metric(
        "Churn Rate",
        f"{churn_rate:.2f}%"
    )


    col3.metric(
        "Average Monthly Charges",
        f"${avg_charge:.2f}"
    )


    col4.metric(
        "Churned Customers",
        churn_customer
    )


    st.divider()


    # Churn Distribution

    st.subheader("Customer Churn Distribution")


    churn_fig = px.pie(
        df,
        names="Churn",
        title="Churn vs Retained Customers",
        hole=0.4
    )


    st.plotly_chart(
        churn_fig,
        use_container_width=True
    )



# ==================================================
# CUSTOMER ANALYSIS
# ==================================================

elif page == "Customer Analysis":


    st.subheader("Customer Segmentation")


    col1,col2 = st.columns(2)


    with col1:

        fig = px.histogram(
            df,
            x="tenure",
            color="Churn",
            title="Tenure vs Churn"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with col2:

        fig = px.box(
            df,
            x="Churn",
            y="MonthlyCharges",
            title="Monthly Charges Impact"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )



# ==================================================
# CHURN DRIVERS
# ==================================================

elif page == "Churn Drivers":


    st.subheader("Important Churn Factors")


    if "Contract" in df.columns:

        contract = (
            df.groupby("Contract")["Churn"]
            .value_counts(normalize=True)
            .rename("Percentage")
            .reset_index()
        )


        fig = px.bar(
            contract,
            x="Contract",
            y="Percentage",
            color="Churn",
            title="Contract Type Impact on Churn"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    st.info(
        """
        Key Business Insights:

        • Month-to-month customers show higher churn risk.

        • Customers with lower tenure are more likely to leave.

        • Higher monthly charges influence churn behaviour.
        """
    )



# ==================================================
# PREDICTION PAGE
# ==================================================

elif page == "Prediction":


    st.subheader("Customer Churn Prediction")

    st.write(
        "Enter customer details to estimate churn risk."
    )


    st.warning(
        "Prediction module will be connected with your trained model features."
    )