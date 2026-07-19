import streamlit as st
import uuid
import torch
import os
from huggingface_hub import hf_hub_download
from auth import init_auth, get_user
from database import init_db, save_message, load_history, get_sessions, delete_session

# ---------- 页面配置 ----------
st.set_page_config(page_title="MicroChat", page_icon="🧠", layout="wide")

# ---------- 黄色治愈主题 CSS ----------
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

# ---------- 模型加载（带异常处理） ----------
model_ready = False
try:
    # 尝试从 Hugging Face 下载权重（如果仓库不存在，会抛出异常）
    # 这里先注释掉，让网站无模型也能运行
    # model_file = hf_hub_download(
    #     repo_id="Lucius-maker/MicroChat-Distill",
    #     filename="full_sft_distill_786.pth",
    #     cache_dir="/tmp/huggingface_cache"
    # )
    # from model.model_minimind import MiniMindForCausalLM, MiniMindConfig
    # config = MiniMindConfig()
    # model = MiniMindForCausalLM(config)
    # state_dict = torch.load(model_file, map_location='cpu')
    # model.load_state_dict(state_dict)
    # model.eval()
    # model_ready = True
    # st.success("✅ 模型加载成功！")
    pass
except Exception as e:
    st.warning(f"⚠️ 模型加载失败，将使用占位回复。错误信息：{e}")

# ---------- 占位回复 ----------
def generate_response(prompt):
    return f"🌻 你说了：{prompt}\n\n（MicroChat 1.2 已上线，模型正在接入中，请耐心等待）"

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
        
        response = generate_response(prompt)
        
        messages.append({"role": "assistant", "content": response})
        save_message(username, current, "assistant", response)
        with st.chat_message("assistant"):
            st.markdown(response)

# ---------- 主入口 ----------
def main():
    init_db()
    authenticator = init_auth()
    
    # 修正 login 调用：location 参数必须为 'main' 或 'sidebar'
    name, authentication_status, username = authenticator.login(
        location='main',
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
    
    # 修正 register_user 调用
    try:
        if authenticator.register_user(
            preauthorization=False,
            fields={'Form name':'注册', 'Username':'用户名', 'Password':'密码', 'Repeat password':'确认密码', 'Register':'注册'}
        ):
            st.success('✅ 注册成功！请登录')
    except Exception as e:
        st.error(str(e))

if __name__ == "__main__":
    main()
