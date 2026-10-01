"""Streamlit UI.  Run:  streamlit run app.py"""
import streamlit as st
from chatbot.rag import Chatbot

st.set_page_config(page_title="Student Support Bot", page_icon="🎓")
st.title("🎓 University Student Support Chatbot")
st.caption("Answers from AKTU and college documents. Always confirm important dates and fees officially.")


@st.cache_resource
def load():
    return Chatbot()


bot = load()
if "msgs" not in st.session_state:
    st.session_state.msgs = []
for m in st.session_state.msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
if q := st.chat_input("Ask about exams, fees, attendance, scholarships..."):
    st.session_state.msgs.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    res = bot.ask(q, [{"role": m["role"], "content": m["content"]} for m in st.session_state.msgs[:-1]])
    text = res["answer"]
    with st.chat_message("assistant"):
        st.markdown(text)
        st.caption(f"Topic: {res['topic']} ({res['confidence']:.0%})")
        if res["sources"]:
            with st.expander("Sources"):
                for s in res["sources"]:
                    st.write(s)
    st.session_state.msgs.append({"role": "assistant", "content": text})
