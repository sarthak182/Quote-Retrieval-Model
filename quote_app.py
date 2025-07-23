import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

from datasets import load_dataset
ds = load_dataset("Abirate/english_quotes")
data_f=ds["train"].to_pandas()
all_quotes = data_f["quote"].tolist()
author_list = data_f["author"].dropna().unique().tolist()

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')  # Or your trained model path

model = load_model()

@st.cache_data
def get_quote_embeddings():
    return model.encode(all_quotes, convert_to_tensor=True)

quote_embeddings = get_quote_embeddings()

def extract_author_from_query(query, author_list):
    query_lower = query.lower()
    for author in author_list:
        if author.lower() in query_lower:
            return author
    return None

st.title("Quote Retriever")
st.markdown("Enter a query like *'Give me quotes about love by Einstein'* or *'Wisdom quotes'*")

query = st.text_input("Your Query")

top_k = st.slider("How many quotes to show?", 1, 20, 5)

if query:
    with st.spinner("Searching..."):
        matched_author = extract_author_from_query(query, author_list)

        query_embedding = model.encode(query, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_embedding, quote_embeddings)[0]
        top_results = cosine_scores.argsort(descending=True)[:top_k*2]

        shown = 0
        st.subheader("Top Quotes")

        for idx in top_results.tolist():
            author = data_f.iloc[idx]["author"]
            quote = all_quotes[idx]

            if matched_author:
                if matched_author.lower() not in str(author).lower():
                    continue  # skip if doesn't match

            score = cosine_scores[idx].item()
            with st.container():
                st.markdown(f"> *{quote}*")
                st.markdown(f"— **{author}**  \nSimilarity Score: `{score:.4f}`")

            shown += 1
            if shown >= top_k:
                break

        if shown == 0:
            st.warning("No quotes matched the author you asked for. Try rephrasing.")
