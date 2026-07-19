import streamlit as st
import uuid
import torch
from huggingface_hub import hf_hub_download
from auth import init_auth, get_user
from database import init_db, save_message, load_history, get_sessions, delete_session

# ---------- 页面配置 ----------
st.set_page_config(
    page_title="MicroChat",
    page_icon="💛",
    layout="wide"
)

# ---------- DeepSeek 风格 CSS（暖黄点缀） ----------
st.markdown("""
<style>
/* 全局 - 极简白 */
.stApp {
    background: #ffffff;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* 侧边栏 - 浅灰 */
.css-1d391kg {
    background: #fafafa;
    border-right: 1px solid #f0f0f0;
}

/* 标题 - 简洁，微黄 */
h1 {
    font-size: 1.8rem;
    font-weight: 600;
    color: #1a1a1a;
    letter-spacing: -0.5px;
    margin-bottom: 0;
}
h1 span {
    color: #f5b041;
}

.subtitle {
    font-size: 0.9rem;
    color: #999;
    margin-top: -0.2rem;
    margin-bottom: 1.5rem;
    font-weight: 400;
}

/* 按钮 - 暖黄点缀 */
.stButton > button {
    background: #ffffff;
    color: #1a1a1a;
    border: 1px solid #e8e8e8;
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: #fef9e7;
    border-color: #f5b041;
    color: #b8860b;
}

/* 输入框 - 干净 */
.stTextInput > div > div > input {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.95rem;
    color: #1a1a1a;
}
.stTextInput > div > div > input:focus {
    border-color: #f5b041;
    box-shadow: 0 0 0 3px rgba(245, 176, 65, 0.1);
    outline: none;
}

/* 聊天气泡 - 左侧灰，右侧暖黄 */
div[data-testid="stChatMessage"]:nth-child(odd) {
    background: #f7f7f8;
    border-radius: 16px 16px 16px 4px;
    padding: 12px 16px;
}
div[data-testid="stChatMessage"]:nth-child(even) {
    background: #fef9e7;
    border-radius: 16px 16px 4px 16px;
    padding: 12px 16px;
}

/* 用户消息中的文本颜色 */
div[data-testid="stChatMessage"]:nth-child(odd) .stMarkdown {
    color: #1a1a1a;
}
div[data-testid="stChatMessage"]:nth-child(even) .stMarkdown {
    color: #5d4037;
}

/* 会话列表 - 简洁 */
.sidebar-chat-item {
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 0.9rem;
    color: #333;
    cursor: pointer;
    transition: background 0.15s ease;
}
.sidebar-chat-item:hover {
    background: #f0f0f0;
}
.sidebar-chat-item.active {
    background: #fef9e7;
    border-left: 3px solid #f5b041;
}

/* 新对话按钮 */
.new-chat-btn {
    background: #f5b041 !important;
    color: #fff !important;
    border: none !important;
    font-weight: 500 !important;
}
.new-chat-btn:hover {
    background: #e0992e !important;
    color: #fff !important;
}

/* 删除按钮 - 浅灰 */
.delete-btn {
    background: transparent !important;
    border: none !important;
    color: #ccc !important;
    font-size: 0.8rem !important;
    padding: 0 6px !important;
}
.delete-btn:hover {
    color: #e74c3c !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- 标题 ----------
st.markdown('<h1>Micro<span>Chat</span></h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">温暖 · 有记忆 · 懂推理</p>', unsafe_allow_html=True)

# ---------- 模型加载（从 Hugging Face 下载） ----------
@st.cache_resource
def load_model():
    try:
        from model.model_minimind import MiniMindForCausalLM, MiniMindConfig
        
        # 从 Hugging Face 下载权重
        model_file = hf_hub_download(
            repo_id="你的用户名/MicroChat-Distill",  # 替换为你的实际仓库
            filename="full_sft_distill_786.pth",
            cache_dir="/tmp/huggingface_cache"
        )
        
        config = MiniMindConfig()
        model = MiniMindForCausalLM(config)
        state_dict = torch.load(model_file, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.warning(f"⚠️ 模型加载失败，将使用占位回复：{e}")
        return None

model = load_model()
model_ready = model is not None

# ---------- 生成回复 ----------
def generate_response(prompt, messages):
    if model_ready:
        # 调用真实模型生成回复
        try:
            # 这里需要根据 MiniMindForCausalLM 的实际生成接口编写
            # 示例：model.generate(prompt, max_length=256)
            # 由于不清楚具体 API，先用占位
            return f"💛 你说了：{prompt}\n\n（模型已加载，生成接口待适配）"
        except Exception as e:
            return f"⚠️ 生成失败：{e}"
    else:
        return f"💛 你说了：{prompt}\n\n（模型正在接入中，请稍候）"

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
                    st.markdown(f"<div class='sidebar-chat-item' onClick=''>💬 {sid[:6]}</div>", unsafe_allow_html=True)
                    # 使用按钮替代点击
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
