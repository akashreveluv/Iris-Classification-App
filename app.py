import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Iris Research Analytics",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp{
    background:#F8FAFC;
}
[data-testid="stSidebar"]{
    background:#0F172A;
}
[data-testid="stSidebar"] *{
    color:white !important;
}
.stButton>button{
    width:100%;
    height:48px;
    background:#2563EB;
    color:white !important;
    border:none;
    border-radius:10px;
    font-weight:600;
}
.stButton>button:hover{
    background:#1D4ED8;
}
img{
    border-radius:12px;
}
</style>
""", unsafe_allow_html=True)

df = pd.read_csv("data/Iris.csv")
model = joblib.load("best_iris_model.pkl")
encoder = joblib.load("label_encoder.pkl")

st.sidebar.title("🌸 Iris Explorer 🌸")

page = st.sidebar.radio(
    "Select Module",
    ["Iris Analytics","Iris Prediction"]
)

if page=="Iris Analytics":

    st.title("🌸 Iris Exploratory Analytics 🌸")

    analysis = st.selectbox(
        "Select Analysis",
        [
            "Petal Features",
            "Sepal Features",
            "Species-wise Density Plot"
        ]
    )

    if analysis=="Petal Features":

        fig,ax=plt.subplots(figsize=(8,6))
        sns.scatterplot(
            data=df,
            x="PetalLengthCm",
            y="PetalWidthCm",
            hue="Species",
            s=90,
            ax=ax
        )
        ax.set_title("Petal Length vs Petal Width")
        st.pyplot(fig)

    elif analysis=="Sepal Features":

        fig,ax=plt.subplots(figsize=(8,6))
        sns.scatterplot(
            data=df,
            x="SepalLengthCm",
            y="SepalWidthCm",
            hue="Species",
            s=90,
            ax=ax
        )
        ax.set_title("Sepal Length vs Sepal Width")
        st.pyplot(fig)

    else:

        feature = st.selectbox(
            "Select Feature",
            [
                "SepalLengthCm",
                "SepalWidthCm",
                "PetalLengthCm",
                "PetalWidthCm"
            ]
        )

        fig,ax = plt.subplots(figsize=(8,5))

        sns.kdeplot(
            data=df,
            x=feature,
            hue="Species",
            fill=True,
            ax=ax
        )

        ax.set_title(f"Density Distribution of {feature}")
        st.pyplot(fig)

else:

    st.title("🌸 Iris Species Prediction 🌸")

    c1,c2 = st.columns([1.2,1])

    with c1:

        sl = st.slider("Sepal Length (cm)",
                       float(df["SepalLengthCm"].min()),
                       float(df["SepalLengthCm"].max()),
                       5.1)

        sw = st.slider("Sepal Width (cm)",
                       float(df["SepalWidthCm"].min()),
                       float(df["SepalWidthCm"].max()),
                       3.5)

        pl = st.slider("Petal Length (cm)",
                       float(df["PetalLengthCm"].min()),
                       float(df["PetalLengthCm"].max()),
                       1.4)

        pw = st.slider("Petal Width (cm)",
                       float(df["PetalWidthCm"].min()),
                       float(df["PetalWidthCm"].max()),
                       0.2)

        predict = st.button("Predict Species")

    with c2:
        if predict:

            sample = np.array([[sl,sw,pl,pw]])

            pred = model.predict(sample)
            species = encoder.inverse_transform(pred)[0]

            st.success(f"Predicted Species: {species}")

            image_map = {
                "Iris-setosa":"images/Iris_setosa.jpg",
                "Iris-versicolor":"images/Iris_versicolor.jpg",
                "Iris-virginica":"images/Iris_virginica.jpg"
            }

            if species in image_map:
                st.image(
                    image_map[species],
                    caption=species.replace("Iris-","Iris ").title(),
                    width=450
                )