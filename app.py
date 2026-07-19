import streamlit as st
import uuid
import torch
import os
from huggingface_hub import hf_hub_download
from auth import init_auth, get_user
from database import init_db, save_message, load_history, get_sessions, delete_session
from model.model_minimind import MiniMindForCausalLM, MiniMindConfig

# ---------- 页面配置 ----------
st.set_page_config(page_title="MicroChat", page_icon="🧠", layout="wide")

# ---------- 黄色治愈主题 CSS（省略，与之前相同） ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(145deg, #fdf6e3 0%, #fef9e7 50%, #fff8e1 100%);
    font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
}
h1 {
    text-align: center;
    font-size: 2.8rem;
    font-weight: 600;
    color: #b8860b;
    text-shadow: 2px 2px 8px rgba(184, 134, 11, 0.15);
    letter-spacing: 1px;
}
.stButton > button {
    background: linear-gradient(145deg, #f9e79f, #f7dc6f);
    color: #7d6608;
    border: none;
    border-radius: 30px;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(184, 134, 11, 0.2);
    transition: all 0.25s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    background: linear-gradient(145deg, #f7dc6f, #f5b041);
    box-shadow: 0 8px 24px rgba(184, 134, 11, 0.3);
    color: #4d3800;
}
.stTextInput > div > div > input {
    background: #fffdf5;
    border: 2px solid #f7dc6f;
    border-radius: 30px;
    padding: 12px 20px;
    font-size: 1rem;
    color: #5d4037;
    box-shadow: 0 2px 8px rgba(184, 134, 11, 0.08);
}
.stTextInput > div > div > input:focus {
    border-color: #d4a017;
    box-shadow: 0 4px 16px rgba(184, 134, 11, 0.2);
    outline: none;
}
div[data-testid="stChatMessage"]:nth-child(odd) {
    background: #fdebd0;
    border-left: 4px solid #f5b041;
    border-radius: 18px 18px 4px 18px;
}
div[data-testid="stChatMessage"]:nth-child(even) {
    background: #fffdf7;
    border-left: 4px solid #f7dc6f;
    border-radius: 18px 18px 18px 4px;
}
</style>
""", unsafe_allow_html=True)

# ---------- 模型加载（从 Hugging Face 下载权重） ----------
@st.cache_resource
def load_model():
    # 1. 下载权重文件到临时目录
    model_file = hf_hub_download(
        repo_id="Lucius-maker/MicroChat-Distill",
        filename="full_sft_distill_786.pth",
        cache_dir="/tmp/huggingface_cache"
    )
    # 2. 实例化模型
    model = MiniMindForCausalLM(MiniMindConfig())
    # 3. 加载权重
    state_dict = torch.load(model_file, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

# 加载模型（首次访问时会下载，约 132MB）
try:
    model = load_model()
    model_ready = True
except Exception as e:
    model_ready = False
    st.error(f"模型加载失败：{e}")

# ---------- 会话管理（同前） ----------
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
        st.title("🧠 MicroChat")
        st.caption(f"用户：{username}")
        st.divider()
        if st.button("➕ 新对话", use_container_width=True):
            new_chat(username)
        st.divider()
        st.subheader("📚 历史对话")
        sessions = get_sessions(username)
        if not sessions:
            st.info("暂无历史对话")
        else:
            for sid in sessions:
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(f"💬 {sid[:8]}", key=f"chat_{sid}", use_container_width=True):
                        switch_chat(username, sid)
                with col2:
                    if st.button("✕", key=f"del_{sid}", help="删除此会话"):
                        delete_session(username, sid)
                        if sid in st.session_state.sessions:
                            del st.session_state.sessions[sid]
                        if st.session_state.current_session == sid:
                            st.session_state.current_session = None
                        st.rerun()

# ---------- 生成回复 ----------
def generate_response(prompt, model):
    # 这里调用模型生成回复（需要你根据 MiniMindLM 的接口实现）
    # 示例：model.generate(prompt, max_length=256)
    # 由于不知道具体 API，先返回占位
    return f"🌻 你说了：{prompt}\n（模型已加载，回复功能开发中）"

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
        
        if model_ready:
            response = generate_response(prompt, model)
        else:
            response = "⚠️ 模型未加载，请稍后再试"
        
        messages.append({"role": "assistant", "content": response})
        save_message(username, current, "assistant", response)
        with st.chat_message("assistant"):
            st.markdown(response)

# ---------- 主入口 ----------
def main():
    init_db()
    authenticator = init_auth()
    
    name, authentication_status, username = authenticator.login(
        'Login', 'main',
        fields={'Form name':'登录', 'Username':'用户名', 'Password':'密码', 'Login':'登录'}
    )
    
    if authentication_status:
        st.success(f"☀️ 欢迎回来，{name}！")
        authenticator.logout('登出', 'main')
        show_chat_interface(username)
    elif authentication_status is False:
        st.error('用户名或密码错误')
    else:
        st.info('请登录或注册')
    
    try:
        if authenticator.register_user('注册新用户', preauthorization=False):
            st.success('✅ 注册成功！请登录')
    except Exception as e:
        st.error(str(e))

if __name__ == "__main__":
    main()
