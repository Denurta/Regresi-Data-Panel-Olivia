import streamlit as st
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS, RandomEffects
from linearmodels.panel import compare
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt

# Fungsi untuk mengupload file
def upload_file():
    uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx"])
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        return df
    else:
        return None

# Fungsi untuk menampilkan eksplorasi data
def explore_data(df):
    st.header("Eksplorasi Data")

    # Pilih jenis grafik
    chart_types = st.multiselect("Pilih jenis chart", ["Scatterplot", "Histogram", "Piechart", "Line Chart"])
    
    # Pilih kolom untuk visualisasi
    columns = df.columns.tolist()
    selected_columns = st.multiselect("Pilih kolom", columns, default=columns[:2])
    
    if "Scatterplot" in chart_types:
        st.subheader("Scatterplot")
        if len(selected_columns) >= 2:
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=selected_columns[0], y=selected_columns[1], ax=ax)
            st.pyplot(fig)
    
    if "Histogram" in chart_types:
        st.subheader("Histogram")
        for col in selected_columns:
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            st.pyplot(fig)
    
    if "Piechart" in chart_types:
        st.subheader("Piechart")
        if len(selected_columns) > 0:
            fig, ax = plt.subplots()
            df[selected_columns[0]].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
            st.pyplot(fig)
    
    if "Line Chart" in chart_types:
        st.subheader("Line Chart")
        for col in selected_columns:
            fig, ax = plt.subplots()
            sns.lineplot(data=df, x=df.index, y=col, ax=ax)
            st.pyplot(fig)

# Fungsi untuk memeriksa multikolinearitas
def check_multicollinearity(df, predictors):
    st.subheader("Asumsi Multikolinearitas")
    X = df[predictors]
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    st.write(vif_data)

# Fungsi untuk uji Chow
def chow_test(df, predictors, response):
    st.subheader("Uji Chow")
    # Implementasi uji Chow
    st.write("Uji Chow belum diimplementasikan sepenuhnya.")

# Fungsi untuk uji Hausman
def hausman_test(fixed, random):
    st.subheader("Uji Hausman")
    b = fixed.params
    B = random.params
    v_b = fixed.cov
    v_B = random.cov
    df = b.shape[0]
    chi2 = np.dot((b - B).T, np.linalg.inv(v_b - v_B).dot(b - B))
    p_value = stats.chi2.sf(chi2, df)
    return chi2, p_value

# Fungsi untuk uji Lagrange Multiplier (LM)
def lm_test(pooled_model, random_model):
    st.subheader("Uji Lagrange Multiplier (LM)")
    # Implementasi uji LM
    st.write("Uji LM belum diimplementasikan sepenuhnya.")

# Fungsi untuk model regresi data panel
def panel_regression(df, predictors, response, entity_col, time_col, model_type):
    st.subheader("Regresi Data Panel")
    df.set_index([entity_col, time_col], inplace=True)
    y = df[response]
    X = df[predictors]
    X = sm.add_constant(X)
    
    if model_type == "FEM":
        model = PanelOLS(y, X, entity_effects=True)
        results = model.fit()
        st.write(results.summary)
    elif model_type == "REM":
        model = RandomEffects(y, X)
        results = model.fit()
        st.write(results.summary)
    elif model_type == "CEM":
        model = PanelOLS(y, X)
        results = model.fit()
        st.write(results.summary)
    else:
        st.write("Model tidak dikenali.")

    return results

# Fungsi utama
def main():
    st.title("Aplikasi Regresi Data Panel dengan Streamlit")
    
    df = upload_file()
    if df is not None:
        st.write("Dataframe:")
        st.write(df)
        
        explore_data(df)
        
        predictors = st.multiselect("Pilih variabel prediktor", df.columns.tolist())
        response = st.selectbox("Pilih variabel respon", df.columns.tolist())
        entity_col = st.selectbox("Pilih kolom entitas (misal: perusahaan, negara)", df.columns.tolist())
        time_col = st.selectbox("Pilih kolom waktu (misal: tahun)", df.columns.tolist())
        
        if len(predictors) > 0 and response:
            check_multicollinearity(df, predictors)
            
            model_type = st.selectbox("Pilih jenis model regresi", ["FEM", "REM", "CEM"])
            results = panel_regression(df, predictors, response, entity_col, time_col, model_type)
            
            if model_type == "FEM" or model_type == "REM":
                st.subheader("Uji Hausman")
                fixed_model = PanelOLS(df[response], df[predictors], entity_effects=True).fit()
                random_model = RandomEffects(df[response], df[predictors]).fit()
                chi2, p_value = hausman_test(fixed_model, random_model)
                st.write(f"Chi-Square: {chi2}, p-value: {p_value}")
            
            st.subheader("Pilih Uji Tambahan")
            additional_tests = st.multiselect("Pilih uji tambahan", ["Uji Chow", "Uji LM"])
            
            if "Uji Chow" in additional_tests:
                chow_test(df, predictors, response)
            if "Uji LM" in additional_tests:
                lm_test(fixed_model, random_model)
    
if __name__ == "__main__":
    main()
