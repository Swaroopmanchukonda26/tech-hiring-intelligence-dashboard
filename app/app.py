import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
import pickle

# Load ML model
@st.cache_resource
def load_model():
    with open('attrition_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('feature_columns.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    # Fix compatibility issue
    if not hasattr(model, 'multi_class'):
        model.multi_class = 'auto'
    return model, feature_cols

model, feature_cols = load_model()
# Page configuration
st.set_page_config(
    page_title="Tech Hiring Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    import os
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    layoffs = pd.read_csv(os.path.join(base_path, 'data', 'cleaned', 'layoffs_clean.csv'))
    placement = pd.read_csv(os.path.join(base_path, 'data', 'cleaned', 'placement_clean.csv'))
    hr = pd.read_csv(os.path.join(base_path, 'data', 'cleaned', 'hr_clean.csv'))
    layoffs['date'] = pd.to_datetime(layoffs['date'])
    layoffs['year'] = layoffs['date'].dt.year
    return layoffs, placement, hr
layoffs, placement, hr = load_data()

# Main title
st.markdown('<p class="main-header">🔍 Tech Industry Hiring Intelligence Dashboard</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analyzing Layoffs, Fresher Placement and HR Attrition Trends</p>',
            unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/analytics.png", width=80)
st.sidebar.title("📊 Dashboard Controls")
st.sidebar.markdown("---")

analysis = st.sidebar.selectbox(
    "Select Analysis",
    ["🏠 Overview", "🔴 Tech Layoffs", "🟡 Fresher Placement", "🟢 HR Attrition", "🤖 Attrition Predictor"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Project By:**")
st.sidebar.markdown("Swaroop Manchukonda")
st.sidebar.markdown("Data Analyst | Python | SQL | Power BI")

# ═══════════════════════════════════════
# OVERVIEW PAGE
# ═══════════════════════════════════════
if analysis == "🏠 Overview":
    st.subheader("📈 Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Layoffs Analyzed",
            value=f"{int(layoffs['total_laid_off'].sum()):,}",
            delta="Records from 2020-2026"
        )
    with col2:
        st.metric(
            label="Companies Covered",
            value=f"{layoffs['company'].nunique():,}",
            delta="Across 67 countries"
        )
    with col3:
        placement_rate = (placement['placement_status'] == 1).sum() / len(placement) * 100
        st.metric(
            label="Overall Placement Rate",
            value=f"{placement_rate:.1f}%",
            delta="From 1 lakh students"
        )
    with col4:
        attrition = (hr['Attrition'] == 'Yes').sum() / len(hr) * 100
        st.metric(
            label="HR Attrition Rate",
            value=f"{attrition:.1f}%",
            delta="1470 employees analyzed"
        )

    st.markdown("---")
    st.subheader("🔑 Key Insights from This Project")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.error("🔴 **Layoffs Insight**")
        top_company = layoffs.groupby('company')['total_laid_off'].sum().idxmax()
        top_count = int(layoffs.groupby('company')['total_laid_off'].sum().max())
        st.write(f"**{top_company}** had highest layoffs with **{top_count:,}** employees")
        st.write("**2023** was the worst year for tech layoffs")

    with col2:
        st.warning("🟡 **Placement Insight**")
        avg_sal = placement[placement['placement_status']==1]['salary_package_lpa'].mean()
        st.write(f"Average salary of placed students: **{avg_sal:.2f} LPA**")
        st.write("Students with **2+ internships** have **73% placement rate**")

    with col3:
        st.success("🟢 **HR Insight**")
        high_attr = hr.groupby('Department').apply(
            lambda x: (x['Attrition']=='Yes').sum()/len(x)*100
        ).idxmax()
        st.write(f"**{high_attr}** has highest attrition rate")
        st.write("Higher job level = significantly **higher salary**")

# ═══════════════════════════════════════
# LAYOFFS PAGE
# ═══════════════════════════════════════
elif analysis == "🔴 Tech Layoffs":
    st.subheader("🔴 Tech Industry Layoffs Analysis")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        countries = ['All'] + sorted(layoffs['country'].dropna().unique().tolist())
        selected_country = st.selectbox("Filter by Country", countries)
    with col2:
        years = ['All'] + sorted(layoffs['year'].dropna().unique().astype(int).tolist())
        selected_year = st.selectbox("Filter by Year", years)

    # Apply filters
    filtered = layoffs.copy()
    if selected_country != 'All':
        filtered = filtered[filtered['country'] == selected_country]
    if selected_year != 'All':
        filtered = filtered[filtered['year'] == selected_year]

    # KPI row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Laid Off", f"{int(filtered['total_laid_off'].sum()):,}")
    with col2:
        st.metric("Companies Affected", f"{filtered['company'].nunique():,}")
    with col3:
        st.metric("Industries Affected", f"{filtered['industry'].nunique():,}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # Top companies chart
        top_co = filtered.groupby('company')['total_laid_off'].sum().sort_values(
            ascending=False).head(10).reset_index()
        fig = px.bar(top_co, x='total_laid_off', y='company',
                    orientation='h',
                    title='Top 10 Companies by Layoffs',
                    color='total_laid_off',
                    color_continuous_scale='Reds')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Industry pie chart
        ind = filtered.groupby('industry')['total_laid_off'].sum().reset_index()
        ind = ind[ind['total_laid_off'] > 0].sort_values(
            'total_laid_off', ascending=False).head(8)
        fig2 = px.pie(ind, values='total_laid_off', names='industry',
                     title='Layoffs by Industry')
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

    # Yearly trend
    yearly = filtered.groupby('year')['total_laid_off'].sum().reset_index()
    fig3 = px.line(yearly, x='year', y='total_laid_off',
                  title='Layoff Trend Over Years',
                  markers=True,
                  line_shape='spline')
    fig3.update_traces(line_color='red', line_width=3)
    fig3.update_layout(height=350)
    st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════
# PLACEMENT PAGE
# ═══════════════════════════════════════
elif analysis == "🟡 Fresher Placement":
    st.subheader("🟡 Fresher Placement Intelligence")

    # Filters
    tiers = ['All'] + sorted(placement['college_tier'].unique().tolist())
    selected_tier = st.selectbox("Filter by College Tier", tiers)

    filtered_p = placement.copy()
    if selected_tier != 'All':
        filtered_p = filtered_p[filtered_p['college_tier'] == selected_tier]

    # KPIs
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Students", f"{len(filtered_p):,}")
    with col2:
        rate = (filtered_p['placement_status']==1).sum()/len(filtered_p)*100
        st.metric("Placement Rate", f"{rate:.1f}%")
    with col3:
        avg = filtered_p[filtered_p['placement_status']==1]['salary_package_lpa'].mean()
        st.metric("Avg Salary (Placed)", f"{avg:.2f} LPA")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # CGPA vs Salary
        placed = filtered_p[filtered_p['placement_status']==1]
        fig = px.scatter(placed, x='cgpa', y='salary_package_lpa',
                        title='CGPA vs Salary Package',
                        color='college_tier',
                        opacity=0.6)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Placement by branch
        branch = filtered_p.groupby('branch')['placement_status'].mean()*100
        branch = branch.sort_values(ascending=False).reset_index()
        fig2 = px.bar(branch, x='placement_status', y='branch',
                     orientation='h',
                     title='Placement Rate by Branch (%)',
                     color='placement_status',
                     color_continuous_scale='Blues')
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════
# HR ATTRITION PAGE
# ═══════════════════════════════════════
elif analysis == "🟢 HR Attrition":
    st.subheader("🟢 HR Employee Attrition Analysis")

    # Filter
    depts = ['All'] + sorted(hr['Department'].unique().tolist())
    selected_dept = st.selectbox("Filter by Department", depts)

    filtered_hr = hr.copy()
    if selected_dept != 'All':
        filtered_hr = filtered_hr[filtered_hr['Department'] == selected_dept]

    # KPIs
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Employees", f"{len(filtered_hr):,}")
    with col2:
        attr = (filtered_hr['Attrition']=='Yes').sum()/len(filtered_hr)*100
        st.metric("Attrition Rate", f"{attr:.1f}%")
    with col3:
        st.metric("Avg Monthly Income",
                 f"${filtered_hr['MonthlyIncome'].mean():,.0f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # Attrition by Job Role
        role_attr = filtered_hr.groupby('JobRole').apply(
            lambda x: (x['Attrition']=='Yes').sum()
        ).sort_values(ascending=False).reset_index()
        role_attr.columns = ['JobRole', 'Attritions']
        fig = px.bar(role_attr, x='Attritions', y='JobRole',
                    orientation='h',
                    title='Attrition Count by Job Role',
                    color='Attritions',
                    color_continuous_scale='Reds')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Income by Job Level
        income = filtered_hr.groupby('JobLevel')['MonthlyIncome'].mean().reset_index()
        fig2 = px.line(income, x='JobLevel', y='MonthlyIncome',
                      title='Average Income by Job Level',
                      markers=True)
        fig2.update_traces(line_color='green', line_width=3)
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

    # Satisfaction analysis
    satisfaction = filtered_hr.groupby('JobSatisfaction')['MonthlyIncome'].mean().reset_index()
    fig3 = px.bar(satisfaction, x='JobSatisfaction', y='MonthlyIncome',
                 title='Average Income by Job Satisfaction Level',
                 color='MonthlyIncome',
                 color_continuous_scale='Greens')
    fig3.update_layout(height=350)
    st.plotly_chart(fig3, use_container_width=True)
# ═══════════════════════════════════════
# ML ATTRITION PREDICTOR PAGE
# ═══════════════════════════════════════
elif analysis == "🤖 Attrition Predictor":
    st.subheader("🤖 Employee Attrition Prediction")
    st.write("Enter employee details to predict if they are likely to leave the company")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 60, 30)
        monthly_income = st.number_input("Monthly Income ($)", 1000, 20000, 5000)
        total_working_years = st.slider("Total Working Years", 0, 40, 5)

    with col2:
        overtime = st.selectbox("OverTime", ["Yes", "No"])
        job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
        years_at_company = st.slider("Years At Company", 0, 40, 3)

    with col3:
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        distance_from_home = st.slider("Distance From Home (miles)", 1, 30, 10)
        work_life_balance = st.slider("Work Life Balance (1-4)", 1, 4, 3)

    if st.button("🔮 Predict Attrition Risk"):

        # Create input array matching training features
        input_data = pd.DataFrame(np.zeros((1, len(feature_cols))), columns=feature_cols)

        # Fill known values
        if 'Age' in input_data.columns:
            input_data['Age'] = age
        if 'MonthlyIncome' in input_data.columns:
            input_data['MonthlyIncome'] = monthly_income
        if 'TotalWorkingYears' in input_data.columns:
            input_data['TotalWorkingYears'] = total_working_years
        if 'OverTime' in input_data.columns:
            input_data['OverTime'] = 1 if overtime == "Yes" else 0
        if 'JobSatisfaction' in input_data.columns:
            input_data['JobSatisfaction'] = job_satisfaction
        if 'YearsAtCompany' in input_data.columns:
            input_data['YearsAtCompany'] = years_at_company
        if 'DistanceFromHome' in input_data.columns:
            input_data['DistanceFromHome'] = distance_from_home
        if 'WorkLifeBalance' in input_data.columns:
            input_data['WorkLifeBalance'] = work_life_balance
        if 'MaritalStatus' in input_data.columns:
            marital_map = {"Single": 2, "Married": 1, "Divorced": 0}
            input_data['MaritalStatus'] = marital_map[marital_status]

        # Predict
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        st.markdown("---")

        if prediction == 1:
            st.error(f"⚠️ HIGH RISK: This employee is likely to leave")
            st.write(f"Attrition Probability: {probability[1]*100:.1f}%")
        else:
            st.success(f"✅ LOW RISK: This employee is likely to stay")
            st.write(f"Retention Probability: {probability[0]*100:.1f}%")

        st.markdown("---")
        st.write("**Model Info:** Logistic Regression | Accuracy: 86.05%")
        st.write("**Top Factors:** Monthly Income, OverTime, Hourly Rate, Marital Status")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built by <b>Swaroop Manchukonda</b> | 
    Data Analyst Portfolio Project | 
    Tools: Python, Pandas, SQL, Power BI, Streamlit</p>
</div>
""", unsafe_allow_html=True)