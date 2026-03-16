from __future__ import annotations

import requests
import streamlit as st
import time

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


def render_results(results: list[dict]) -> None:
    for item in results:
        with st.container(border=True):
            st.markdown(f"### {item['title']}")
            st.markdown(
                f"**Автор:** {item['author']}  \n**Жанр:** {item['category']}  \n**Год:** {item['year']}  \n**Итоговый score:** {item['score']}"
            )
            st.write(item["description"])
            st.caption(item["explanation"])
            if item["matched_signals"]:
                badges = " · ".join(item["matched_signals"])
                st.write(f"**Совпавшие сигналы:** {badges}")
                
def fetch_backend_health(retries: int = 5, delay: float = 1.0):
    last_error = None

    for attempt in range(retries):
        try:
            response = requests.get(f"{BACKEND_URL}/healthz", timeout=3)
            if response.ok:
                return response.json(), None
            last_error = f"HTTP {response.status_code}: {response.text}"
        except requests.RequestException as exc:
            last_error = str(exc)

        if attempt < retries - 1:
            time.sleep(delay)

    return None, last_error


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
    if uploaded_file is not None and st.button("Загрузить и проиндексировать", use_container_width=True):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/json")}
        response = requests.post(f"{BACKEND_URL}/catalog/upload", files=files, timeout=120)
        if response.ok:
            payload = response.json()
            st.success(payload["message"])
            st.json(payload)
        else:
            show_error(response.text)

    st.divider()
    st.subheader("Что понимает поиск")
    st.markdown(
        "- жанр и формат произведения\n"
        "- тему и исторический контекст\n"
        "- настроение и атмосферу\n"
        "- сюжетные сигналы и тип героя"
    )

    st.divider()
    st.subheader("Состояние сервиса")
    try:
        health_payload, health_error = fetch_backend_health()
        status_response = requests.get(f"{BACKEND_URL}/catalog/status", timeout=5)

        if health_payload is not None:
            st.success("Backend доступен")
            st.json(health_payload)
        else:
            st.info("Backend запускается…")
        if status_response.ok:
            payload = status_response.json()
            if payload["catalog_ready"]:
                st.success(payload["message"])
            else:
                st.info(payload["message"])
            st.json(payload)
        else:
            show_error(status_response.text)
    except requests.RequestException:
        st.info("Backend запускается…")
        time.sleep(2)
        st.rerun()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "interpretation" in message:
            interpretation = message["interpretation"]
            st.caption(
                f"Интерпретация запроса: тип = {interpretation['query_type']}; "
                f"темы = {', '.join(interpretation['themes']) or '—'}; "
                f"жанры = {', '.join(interpretation['genres']) or '—'}; "
                f"настроение = {', '.join(interpretation['moods']) or '—'}."
            )
        if "results" in message:
            render_results(message["results"])


prompt = st.chat_input("Например: историческая проза о революции в России")
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
            try:
                error_payload = response.json()
                detail = error_payload.get("detail", response.text)
            except Exception:
                detail = response.text

            error_text = f"Ошибка backend ({response.status_code}): {detail}"
        else:
            payload = response.json()
            assistant_message = {
                "role": "assistant",
                "content": payload["assistant_message"],
                "results": payload["results"],
                "interpretation": payload["interpretation"],
            }
            st.session_state.messages.append(assistant_message)
            with st.chat_message("assistant"):
                st.markdown(payload["assistant_message"])
                st.caption(
                    f"Интерпретация запроса: тип = {payload['interpretation']['query_type']}; "
                    f"темы = {', '.join(payload['interpretation']['themes']) or '—'}; "
                    f"жанры = {', '.join(payload['interpretation']['genres']) or '—'}; "
                    f"настроение = {', '.join(payload['interpretation']['moods']) or '—'}."
                )
                render_results(payload["results"])
    except requests.RequestException as exc:
        error_text = f"Ошибка поиска: {exc}"
        st.session_state.messages.append({"role": "assistant", "content": error_text})
        with st.chat_message("assistant"):
            show_error(error_text)
