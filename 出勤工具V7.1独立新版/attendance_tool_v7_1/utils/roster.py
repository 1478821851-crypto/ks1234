import pandas as pd
import streamlit as st

def read_names(uploaded_file):

    if uploaded_file is None:
        return []

    filename = uploaded_file.name.lower()

    try:

        if filename.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file, header=None)
            values = df.values.flatten()

        elif filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file, header=None)
            values = df.values.flatten()

        elif filename.endswith(".txt"):
            text = uploaded_file.getvalue().decode("utf-8-sig")
            values = text.splitlines()

        else:
            return []

        names = []

        for value in values:

            if pd.isna(value):
                continue

            name = str(value).strip()

            if name:
                names.append(name)

        return list(dict.fromkeys(names))

    except Exception as e:

        st.error(f"读取名录失败：{e}")
        return []
