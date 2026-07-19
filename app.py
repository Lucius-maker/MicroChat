import streamlit as st
import uuid
from auth import init_auth
from database import init_db, save_message, load_history, get_sessions, delete_session

# ---------- 页面配置 ----------
st.set_page_config(page_title="MicroChat", page_icon="💛", layout="wide")

# ---------- DeepSeek 风格 CSS ----------
st.markdown("""
<style>
.stApp { background: #ffffff; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
.css-1d391kg { background: #fafafa; border-right: 1px solid #f0f0f0; }
h1 { font-size: 1.8rem; font-weight: 600; color: #1a1a1a; letter-spacing: -0.5px; margin-bottom: 0; }
h1 span { color: #f5b041; }
.subtitle { font-size: 0.9rem; color: #999; margin-top: -0.2rem; margin-bottom: 1.5rem; font-weight: 400; }
.stButton > button { background: #ffffff; color: #1a1a1a; border: 1px solid #e8e8e8; border-radius: 8px; font-weight: 500; transition: all 0.2s ease; }
.stButton > button:hover { background: #fef9e7; border-color: #f5b041; color: #b8860b; }
.stTextInput > div > div > input { background: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px 16px; font-size: 0.95rem; color: #1a1a1a; }
.stTextInput > div > div > input:focus { border-color: #f5b041; box-shadow: 0 0 0 3px rgba(245, 176, 65, 0.1); outline: none; }
div[data-testid="stChatMessage"]:nth-child(odd) { background: #f7f7f8; border-radius: 16px 16px 16px 4px; padding: 12px 16px; }
div[data-testid="stChatMessage"]:nth-child(even) { background: #fef9e7; border-radius: 16px 16px 4px 16px; padding: 12px 16px; }
div[data-testid="stChatMessage"]:nth-child(odd) .stMarkdown { color: #1a1a1a; }
div[data-testid="stChatMessage"]:nth-child(even) .stMarkdown { color: #5d4037; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1>Micro<span>Chat</span></h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">温暖 · 有记忆 · 懂推理</p>', unsafe_allow_html=True)

# ---------- 占位回复（无模型） ----------
def generate_response(prompt, messages):
    return f"💛 收到：{prompt}\n\n（MicroChat 当前处于无模型模式，仅用于展示界面和功能。模型将在后续版本中接入。）"

# ---------- 会话管理 ----------
def init_session_state(username):
    if "sessions" not in st.session_state:
        st.session_state.sessions = {}
    if "current_session" not in st.session_state:
        existing = get_sessions(username)
        st.session_state.current_session = existing[0] if existing else str(uuid.uuid4())[:8]
    current = st.session_state.current_session
    if current not in st.session_state.sessions:
        history = load_history(username, current)
        st.session_state.sessions[current] = [{"role": r, "content": c} for r, c in history]

def new_chat(username):
    session_id = str(uuid.uuid4())[:8]
    st.session_state.current_session = session_id
    st.session_state.sessions[session_id] = []
    st.rerun()

def switch_chat(username, session_id):
    st.session_state.current_session = session_id
    if session_id not in st.session_state.sessions:
        history = load_history(username, session_id)
        st.session_state.sessions[session_id] = [{"role": r, "content": c} for r, c in history]
    st.rerun()

# ---------- 侧边栏 ----------
def show_sidebar(username):
    with st.sidebar:
        st.markdown("### MicroChat")
        st.caption(f"👤 {username}")
        st.divider()
        st.button("➕ 新对话", on_click=new_chat, args=(username,), use_container_width=True, type="primary")
        st.divider()
        st.markdown("#### 历史对话")
        sessions = get_sessions(username)
        if not sessions:
            st.caption("暂无历史对话")
        else:
            for sid in sessions:
                col1, col2 = st.columns([5, 1])
                with col1:
                    if st.button(f"💬 {sid[:6]}", key=f"chat_{sid}", use_container_width=True):
                        switch_chat(username, sid)
                with col2:
                    if st.button("✕", key=f"del_{sid}", help="删除此会话"):
                        delete_session(username, sid)
                        if sid in st.session_state.sessions:
                            del st.session_state.sessions[sid]
                        if st.session_state.current_session == sid:
                            st.session_state.current_session = None
                        st.rerun()
        st.divider()
        st.caption("v1.2 · 2026.07.19")

# ---------- 聊天界面 ----------
def show_chat_interface(username):
    init_session_state(username)
    show_sidebar(username)
    
    current = st.session_state.current_session
    messages = st.session_state.sessions.get(current, [])
    
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("输入你的问题..."):
        messages.append({"role": "user", "content": prompt})
        save_message(username, current, "user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)
        
        response = generate_response(prompt, messages)
        
        messages.append({"role": "assistant", "content": response})
        save_message(username, current, "assistant", response)
        with st.chat_message("assistant"):
            st.markdown(response)

# ---------- 主入口 ----------
def main():
    init_db()
    authenticator = init_auth()
    authenticator.login()
    
    if st.session_state.get("authentication_status"):
        name = st.session_state.get("name", "用户")
        st.success(f"☀️ 欢迎回来，{name}！")
        authenticator.logout('登出', 'main')
        show_chat_interface(st.session_state.get("username"))
    elif st.session_state.get("authentication_status") is False:
        st.error('用户名或密码错误')
    else:
        st.info('请登录或注册')
    
    try:
        authenticator.register_user()
    except Exception as e:
        st.error(str(e))

if __name__ == "__main__":
    main()
