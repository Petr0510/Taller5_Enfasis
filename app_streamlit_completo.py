
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
import nltk
import re
from transformers import pipeline

nltk.download('stopwords')

st.set_page_config(page_title="Análisis de Opiniones de Clientes", layout="centered")
st.title("🔍 Análisis de Opiniones de Clientes")

@st.cache_resource(show_spinner="Cargando modelo de sentimiento...")
def get_sentiment_pipeline():
    return pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")

@st.cache_resource(show_spinner="Cargando modelo de resumen...")
def get_summarizer_pipeline():
    return pipeline("summarization", model="facebook/bart-large-cnn")

sentiment_pipe = get_sentiment_pipeline()
summarizer = get_summarizer_pipeline()

uploaded_file = st.file_uploader("📁 Sube un archivo CSV con 20 opiniones de clientes", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "opinion" not in df.columns:
        st.error("❌ El archivo debe tener una columna llamada 'opinion'.")
    else:
        opiniones = df["opinion"].astype(str).tolist()
        all_text = " ".join(opiniones).lower()
        stop_words = set(stopwords.words("spanish"))
        words = re.findall(r"\b\w+\b", all_text)
        filtered_words = [word for word in words if word not in stop_words]

        st.subheader("☁️ Nube de Palabras")
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(filtered_words))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

        st.subheader("🔟 Top 10 Palabras Más Frecuentes")
        counter = Counter(filtered_words)
        common_words = counter.most_common(10)
        words_df = pd.DataFrame(common_words, columns=["Palabra", "Frecuencia"])
        fig, ax = plt.subplots()
        ax.bar(words_df["Palabra"], words_df["Frecuencia"], color="skyblue")
        ax.set_ylabel("Frecuencia")
        ax.set_title("Top 10 Palabras")
        st.pyplot(fig)

        st.subheader("📊 Clasificación de Sentimientos")
        sentimientos = sentiment_pipe(opiniones)
        df["sentimiento"] = [s["label"].capitalize() for s in sentimientos]
        st.dataframe(df)

        st.subheader("📈 Porcentaje de Opiniones por Sentimiento")
        sentimiento_counts = df["sentimiento"].value_counts(normalize=True) * 100
        fig, ax = plt.subplots()
        ax.pie(sentimiento_counts, labels=sentimiento_counts.index, autopct="%1.1f%%", colors=["green", "red", "gray"])
        ax.axis("equal")
        st.pyplot(fig)

        st.subheader("🧠 Función Adicional con Modelos de Lenguaje")
        opcion = st.radio("Elige una opción:", ["📬 Analizar nuevo comentario", "📚 Interactuar con comentarios existentes"])

        if opcion == "📬 Analizar nuevo comentario":
            nuevo_comentario = st.text_area("Escribe un nuevo comentario para analizar")
            if st.button("Analizar"):
                resultado = sentiment_pipe(nuevo_comentario)[0]
                resumen = summarizer(nuevo_comentario, max_length=60, min_length=20, do_sample=False)[0]["summary_text"]
                st.markdown(f"**Sentimiento:** {resultado['label'].capitalize()}")
                st.markdown(f"**Resumen:** {resumen}")

        elif opcion == "📚 Interactuar con comentarios existentes":
            pregunta = st.text_input("¿Qué quieres saber de los comentarios?")
            if st.button("Consultar"):
                concatenado = " ".join(opiniones)
                resumen = summarizer(concatenado, max_length=100, min_length=40, do_sample=False)[0]["summary_text"]
                st.markdown("**Resumen de los comentarios:**")
                st.write(resumen)
