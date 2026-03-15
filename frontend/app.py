from __future__ import annotations

import json
from io import BytesIO

import requests
import streamlit as st

BACKEND_URL = "http://backend:8000"

st.set_page_config(page_title="AI Book Finder", page_icon="📚", layout="wide")
st.title("📚 AI Book Finder")
st.caption("Чат-интерфейс для семантического поиска книг по естественному языку")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Привет! Опиши книгу, настроение, жанр или сюжет — и я подберу 5 наиболее релевантных вариантов.",
        }
    ]


def show_error(message: str) -> None:
    st.error(message, icon="🚨")


with st.sidebar:
    st.subheader("Управление каталогом")

    if st.button("Индексировать пример из проекта", use_container_width=True):
        response = requests.post(f"{BACKEND_URL}/catalog/index-default", timeout=120)
        if response.ok:
            payload = response.json()
            st.success(payload["message"])
            st.json(payload)
        else:
            show_error(response.text)

    uploaded_file = st.file_uploader("Загрузить books.json", type=["json"])
    if uploaded_file is not None:
        if st.button("Загрузить и проиндексировать", use_container_width=True):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/json")}
            response = requests.post(f"{BACKEND_URL}/catalog/upload", files=files, timeout=120)
            if response.ok:
                payload = response.json()
                st.success(payload["message"])
                st.json(payload)
            else:
                show_error(response.text)

    st.divider()
    st.subheader("Состояние сервиса")
    try:
        health_response = requests.get(f"{BACKEND_URL}/healthz", timeout=5)
        if health_response.ok:
            st.success("Backend доступен")
            st.json(health_response.json())
        else:
            st.warning("Backend отвечает, но ещё не полностью готов")
    except requests.RequestException:
        st.warning("Backend ещё запускается. Обнови страницу через несколько секунд.")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "results" in message:
            for item in message["results"]:
                with st.container(border=True):
                    st.markdown(f"### {item['title']}")
                    st.markdown(
                        f"**Автор:** {item['author']}  | **Жанр:** {item['category']}  | **Год:** {item['year']}"
                    )
                    st.markdown(
                        f"**Сходство:** {item['score']}"
                    )
                    st.write(item["description"])
                    st.caption(item["explanation"])
                    if item["matched_signals"]:
                        st.write("Сигналы совпадения:", ", ".join(item["matched_signals"]))


prompt = st.chat_input()
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        response = requests.post(
            f"{BACKEND_URL}/search",
            json={"query": prompt, "limit": 5},
            timeout=30,
        )

        if response.status_code >= 400:
            error_text = f"Ошибка backend ({response.status_code}): {response.text}"
            st.session_state.messages.append({"role": "assistant", "content": error_text})
            with st.chat_message("assistant"):
                show_error(error_text)
        else:
            payload = response.json()
            assistant_message = {
                "role": "assistant",
                "content": payload["assistant_message"],
                "results": payload["results"],
            }
            st.session_state.messages.append(assistant_message)
            with st.chat_message("assistant"):
                st.markdown(payload["assistant_message"])
                for item in payload["results"]:
                    with st.container(border=True):
                        st.markdown(f"### {item['title']}")
                        st.markdown(
                            f"**Автор:** {item['author']}  | **Жанр:** {item['category']}  | **Год:** {item['year']}"
                        )
                        st.markdown(
                            f"**Similarity score:** {item['score']}"
                        )
                        st.write(item["description"])
                        st.caption(item["explanation"])
                        if item["matched_signals"]:
                            st.write("Сигналы совпадения:", ", ".join(item["matched_signals"]))
    except requests.RequestException as exc:
        error_text = f"Ошибка поиска: {exc}"
        st.session_state.messages.append({"role": "assistant", "content": error_text})
        with st.chat_message("assistant"):
            show_error(error_text)