import random
import re
import json
import os
import hashlib
from threading import Thread
from pathlib import Path

import torch
import numpy as np
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# ---------- 用户管理 ----------
USER_DB = "users.json"

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if os.path.exists(USER_DB):
        with open(USER_DB, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USER_DB, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

def authenticate(username, password):
    users = load_users()
    return username in users and users[username] == hash_password(password)

def register_user(username, password):
    users = load_users()
    if username in users:
        return False
    users[username] = hash_password(password)
    save_users(users)
    return True

# ---------- 对话管理 ----------
def get_chats_file(username):
    return f"chats_{username}.json"

def load_user_chats(username):
    path = get_chats_file(username)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_user_chats(username, chats):
    with open(get_chats_file(username), "w", encoding="utf-8") as f:
        json.dump(chats, f, ensure_ascii=False, indent=2)

# ---------- 保存当前对话 ----------
def save_current_chat(username):
    if 'messages' in st.session_state and st.session_state.messages:
        chats = load_user_chats(username)
        current_id = st.session_state.get('current_chat_id')
        if current_id is not None:
            for chat in chats:
                if chat['id'] == current_id:
                    chat['messages'] = st.session_state.messages.copy()
                    if st.session_state.messages:
                        chat['title'] = st.session_state.messages[0]['content'][:30]
                    break
        else:
            new_id = max([c['id'] for c in chats], default=-1) + 1
            title = st.session_state.messages[0]['content'][:30] if st.session_state.messages else "新对话"
            chats.append({
                "id": new_id,
                "title": title,
                "messages": st.session_state.messages.copy()
            })
            st.session_state['current_chat_id'] = new_id
        save_user_chats(username, chats)

# ---------- 页面配置 ----------
st.set_page_config(page_title="MiniMind", initial_sidebar_state="expanded")

# ---------- 样式（亮黄色 DeepSeek 风格） ----------
st.markdown("""
<style>
    /* 全局背景 - 浅黄白 */
    .stApp {
        background: #fef9e6;
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    }
    /* 标题 - 亮黄 */
    h1 {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 600;
        color: #e6a800;
        text-shadow: 2px 2px 12px rgba(230, 168, 0, 0.2);
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #d4a373;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    /* 聊天容器 */
    .stChatMessage {
        background: rgba(255, 248, 225, 0.6);
        backdrop-filter: blur(4px);
        border-radius: 20px;
        padding: 12px 18px;
        margin: 8px 0;
        border: 1px solid rgba(247, 201, 72, 0.2);
        box-shadow: 0 4px 12px rgba(247, 201, 72, 0.08);
    }
    /* 用户消息 - 淡黄 */
    div[data-testid="stChatMessage"]:nth-child(odd) {
        background: #fef3c7;
        border-left: 4px solid #f7c948;
        border-radius: 18px 18px 4px 18px;
    }
    /* 助手消息 - 奶油白 */
    div[data-testid="stChatMessage"]:nth-child(even) {
        background: #fffdf7;
        border-left: 4px solid #f7dc6f;
        border-radius: 18px 18px 18px 4px;
    }
    /* 输入框 */
    .stTextInput > div > div > input {
        background: #fffdf5;
        border: 2px solid #f7c948;
        border-radius: 30px;
        padding: 12px 20px;
        font-size: 1rem;
        color: #5d4037;
        box-shadow: 0 2px 8px rgba(247, 201, 72, 0.15);
        transition: all 0.3s ease;
    }
    .stTextInput > div > div > input:focus {
        border-color: #e6a800;
        box-shadow: 0 4px 16px rgba(247, 201, 72, 0.3);
        outline: none;
    }
    /* 按钮 - 亮黄 */
    .stButton > button {
        background: #f7c948;
        color: #333;
        border: none;
        border-radius: 30px;
        padding: 10px 28px;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 12px rgba(247, 201, 72, 0.3);
        transition: all 0.25s ease;
        cursor: pointer;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        background: #f5b700;
        box-shadow: 0 8px 24px rgba(247, 201, 72, 0.5);
        color: #1a1a1a;
    }
    .stButton > button:active {
        transform: translateY(0px);
        box-shadow: 0 2px 8px rgba(247, 201, 72, 0.2);
    }
    /* 侧边栏 */
    .css-1d391kg {
        background: #fef9e7;
        border-right: 1px solid #f7c948;
    }
    /* 滚动条 */
    ::-webkit-scrollbar {
        width: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #fef9e7;
    }
    ::-webkit-scrollbar-thumb {
        background: #f7c948;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #e6a800;
    }
    .footer {
        text-align: center;
        color: #d4a373;
        font-size: 0.8rem;
        margin-top: 2rem;
        opacity: 0.7;
        border-top: 1px solid #f7c948;
        padding-top: 1rem;
    }
    /* 新对话按钮（主界面顶部） */
    .new-chat-top {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ---------- 工具函数和全局变量 ----------
device = "cuda" if torch.cuda.is_available() else "cpu"

LANG_TEXTS = {
    'zh': {
        'settings': '模型设定调整',
        'history_rounds': '历史对话轮次',
        'max_length': '最大生成长度',
        'temperature': '温度',
        'thinking': '思考',
        'tools': '工具',
        'language': '语言',
        'send': '给 MiniMind 发送消息',
        'disclaimer': 'AI 生成内容可能存在错误，请仔细核实',
        'think_tip': '自适应思考，目前多轮对话或Tool Call共存时思考不稳定',
        'tool_select': '工具选择（最多4个）',
    },
    'en': {
        'settings': 'Model Settings',
        'history_rounds': 'History Rounds',
        'max_length': 'Max Length',
        'temperature': 'Temperature',
        'thinking': 'Thinking',
        'tools': 'Tools',
        'language': 'Language',
        'send': 'Send a message to MiniMind',
        'disclaimer': 'AI-generated content may be inaccurate, please verify',
        'think_tip': 'Adaptive thinking; may be unstable with multi-turn or Tool Call',
        'tool_select': 'Tool Selection (max 4)',
    }
}

def get_text(key):
    lang = st.session_state.get('lang', 'en')
    return LANG_TEXTS.get(lang, {}).get(key, LANG_TEXTS['zh'].get(key, key))

TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "数学表达式"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "获取当前时间", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "default": "Asia/Shanghai"}}, "required": []}}},
    {"type": "function", "function": {"name": "random_number", "description": "生成随机数", "parameters": {"type": "object", "properties": {"min": {"type": "integer"}, "max": {"type": "integer"}}, "required": ["min", "max"]}}},
    {"type": "function", "function": {"name": "text_length", "description": "计算文本长度", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "单位转换", "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "获取天气", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "获取汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "翻译文本", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_lang": {"type": "string"}}, "required": ["text", "target_lang"]}}},
]

TOOL_SHORT_NAMES = {
    'calculate_math': '数学', 'get_current_time': '时间', 'random_number': '随机',
    'text_length': '字数', 'unit_converter': '单位', 'get_current_weather': '天气',
    'get_exchange_rate': '汇率', 'translate_text': '翻译'
}

def execute_tool(tool_name, args):
    import datetime
    try:
        if tool_name == 'calculate_math':
            return {"result": eval(args.get('expression', '0'))}
        elif tool_name == 'get_current_time':
            tz = args.get('timezone', 'Asia/Shanghai')
            return {"result": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        elif tool_name == 'random_number':
            return {"result": random.randint(args.get('min', 0), args.get('max', 100))}
        elif tool_name == 'text_length':
            return {"result": len(args.get('text', ''))}
        elif tool_name == 'unit_converter':
            return {"result": f"{args.get('value', 0)} {args.get('from_unit', '')} = ? {args.get('to_unit', '')}"}
        elif tool_name == 'get_current_weather':
            return {"result": f"{args.get('city', 'Unknown')}: 晴, 7~10°C"}
        elif tool_name == 'get_exchange_rate':
            return {"result": f"1 {args.get('from_currency', 'USD')} = 7.2 {args.get('to_currency', 'CNY')}"}
        elif tool_name == 'translate_text':
            return {"result": f"[翻译结果]: hello world"}
        return {"result": "Unknown tool"}
    except Exception as e:
        return {"error": str(e)}

def process_assistant_content(content, is_streaming=False):
    if '<tool_call>' in content:
        def format_tool_call(match):
            try:
                tc = json.loads(match.group(1))
                name = tc.get('name', 'unknown')
                args = tc.get('arguments', {})
                return f'<div style="background: rgba(80, 110, 150, 0.20); border: 1px solid rgba(140, 170, 210, 0.30); padding: 10px 12px; border-radius: 12px; margin: 6px 0;"><div style="font-size:12px;opacity:.75;display:block;margin:0 0 6px 0;line-height:1;">ToolCalling</div><div><b>{name}</b>: {json.dumps(args, ensure_ascii=False)}</div></div>'
            except:
                return match.group(0)
        content = re.sub(r'<tool_call>(.*?)</tool_call>', format_tool_call, content, flags=re.DOTALL)
    
    if is_streaming and st.session_state.get('enable_thinking', False) and '</think>' not in content and '<think>' not in content:
        m = re.search(r'(\n\n(?:我是|您好|你好)[^\n]*)', content)
        if m and m.start(1) > 5:
            i = m.start(1)
            think_part = content[:i]
            answer_part = content[i:]
            return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">已思考</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto;">{think_part.strip()}</div></details>{answer_part}'
        elif len(content) > 5:
            return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">思考中...</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto; display: flex; flex-direction: column-reverse;"><div style="margin-bottom: auto;">{content.strip().replace(chr(10), "<br>")}</div></div></details>'

    if '<think>' in content and '</think>' in content:
        def format_think(match):
            think_content = match.group(2)
            if think_content.replace('\n', '').strip():
                return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">已思考</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto;">{think_content.strip()}</div></details>'
            return ''
        content = re.sub(r'(<think>)(.*?)(</think>)', format_think, content, flags=re.DOTALL)

    if '<think>' in content and '</think>' not in content:
        def format_think_in_progress(match):
            tc = match.group(1)
            return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">思考中...</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto; display: flex; flex-direction: column-reverse;"><div style="margin-bottom: auto;">{tc.strip().replace(chr(10), "<br>")}</div></div></details>'
        content = re.sub(r'<think>(.*?)$', format_think_in_progress, content, flags=re.DOTALL)

    if '<think>' not in content and '</think>' in content:
        def format_think_no_start(match):
            think_content = match.group(1)
            if think_content.replace('\n', '').strip():
                return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">已思考</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto;">{think_content.strip()}</div></details>'
            return ''
        content = re.sub(r'(.*?)</think>', format_think_no_start, content, flags=re.DOTALL)
    return content

@st.cache_resource
def load_model_tokenizer(model_path):
    if model_path is None or not os.path.exists(model_path):
        repo_id = "Qwen/Qwen2.5-0.5B-Instruct"
        print(f"从 Hugging Face 加载模型: {repo_id}")
        model_path = repo_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=False,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=False
    )
    model = model.eval()
    return model, tokenizer

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------- 登录/注册页面 ----------
def login_register_page():
    st.markdown("<h1 style='text-align:center;'>🔐 登录 / 注册</h1>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["登录", "注册"])
    with tab1:
        with st.form("login_form"):
            username = st.text_input("用户名")
            password = st.text_input("密码", type="password")
            submitted = st.form_submit_button("登录")
            if submitted:
                if authenticate(username, password):
                    st.session_state['user'] = username
                    st.session_state['current_chat_id'] = None
                    st.session_state.messages = []
                    st.session_state.chat_messages = []
                    st.rerun()
                else:
                    st.error("用户名或密码错误")
    with tab2:
        with st.form("register_form"):
            new_user = st.text_input("新用户名")
            new_pass = st.text_input("密码", type="password")
            confirm_pass = st.text_input("确认密码", type="password")
            submitted = st.form_submit_button("注册")
            if submitted:
                if new_pass != confirm_pass:
                    st.error("两次密码不一致")
                elif register_user(new_user, new_pass):
                    st.success("注册成功，请登录")
                else:
                    st.error("用户名已存在")

# ---------- 主函数 ----------
def main():
    # 检查登录状态
    if 'user' not in st.session_state:
        login_register_page()
        return

    username = st.session_state['user']

    # ---------- 侧边栏：用户信息 + 对话管理 ----------
    with st.sidebar:
        st.write(f"👤 {username}")
        if st.button("🚪 登出"):
            save_current_chat(username)
            del st.session_state['user']
            del st.session_state['current_chat_id']
            st.session_state.messages = []
            st.session_state.chat_messages = []
            st.rerun()
        
        st.markdown("---")
        if st.button("➕ 新建对话"):
            save_current_chat(username)
            st.session_state.messages = []
            st.session_state.chat_messages = []
            st.session_state['current_chat_id'] = None
            st.rerun()
        
        st.markdown("---")
        st.write("📂 历史对话")
        chats = load_user_chats(username)
        for chat in chats:
            cols = st.columns([5, 1])
            with cols[0]:
                if st.button(chat['title'], key=f"load_{chat['id']}"):
                    save_current_chat(username)
                    st.session_state.messages = chat['messages'].copy()
                    st.session_state.chat_messages = chat['messages'].copy()
                    st.session_state['current_chat_id'] = chat['id']
                    st.rerun()
            with cols[1]:
                if st.button("🗑️", key=f"del_{chat['id']}"):
                    chats = [c for c in chats if c['id'] != chat['id']]
                    save_user_chats(username, chats)
                    if st.session_state.get('current_chat_id') == chat['id']:
                        st.session_state.messages = []
                        st.session_state.chat_messages = []
                        st.session_state['current_chat_id'] = None
                    st.rerun()

        st.markdown("---")
        # 模型参数设置
        selected_model = st.selectbox('Model', list(MODEL_PATHS.keys()), index=0)
        model_path = MODEL_PATHS[selected_model][0]
        st.session_state.history_chat_num = st.slider(get_text('history_rounds'), 0, 8, 0, step=2)
        st.session_state.max_new_tokens = st.slider(get_text('max_length'), 256, 8192, 8192, step=1)
        st.session_state.temperature = st.slider(get_text('temperature'), 0.6, 1.2, 0.90, step=0.01)
        st.session_state.enable_thinking = st.checkbox(get_text('thinking'), value=False, help=get_text('think_tip'))
        st.session_state.selected_tools = []
        with st.expander(get_text('tools')):
            st.caption(get_text('tool_select'))
            selected_count = sum(1 for tool in TOOLS if st.session_state.get(f"tool_{tool['function']['name']}", False))
            for tool in TOOLS:
                name = tool['function']['name']
                short_name = TOOL_SHORT_NAMES.get(name, name)
                checked = st.checkbox(short_name, key=f"tool_{name}", disabled=(selected_count >= 4 and not st.session_state.get(f"tool_{name}", False)))
                if checked and len(st.session_state.selected_tools) < 4:
                    st.session_state.selected_tools.append(name)

    # ---------- 主聊天界面 ----------
    # 顶部显示当前对话标题和“新建对话”按钮（额外放置，确保可见）
    col1, col2 = st.columns([4, 1])
    with col1:
        chat_title = "新对话" if not st.session_state.messages else st.session_state.messages[0]['content'][:30]
        st.markdown(f"### 💬 {chat_title}")
    with col2:
        if st.button("✨ 新建对话", key="new_chat_top"):
            save_current_chat(username)
            st.session_state.messages = []
            st.session_state.chat_messages = []
            st.session_state['current_chat_id'] = None
            st.rerun()

    st.markdown("---")

    # 加载模型
    model, tokenizer = load_model_tokenizer(model_path)

    # 显示消息
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_messages = []

    for message in st.session_state.messages:
        if message["role"] == "assistant":
            st.markdown(process_assistant_content(message["content"]), unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px; background-color: #3d4450; border-radius: 22px; color: white;">{message["content"]}</div></div>',
                unsafe_allow_html=True)

    prompt = st.chat_input(key="input", placeholder=get_text('send'))

    if prompt:
        st.markdown(
            f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px; background-color: #3d4450; border-radius: 22px; color: white;">{prompt}</div></div>',
            unsafe_allow_html=True)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        placeholder = st.empty()

        random_seed = random.randint(0, 2 ** 32 - 1)
        setup_seed(random_seed)

        tools = [t for t in TOOLS if t['function']['name'] in st.session_state.get('selected_tools', [])] or None
        sys_prompt = [] if tools else [{"role": "system", "content": "你是MiniMind，一个乐于助人、知识渊博的AI助手。请用完整且友好的方式回答用户问题。"}]
        st.session_state.chat_messages = sys_prompt + st.session_state.chat_messages[-(st.session_state.history_chat_num + 1):]
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if st.session_state.get('enable_thinking', False):
            template_kwargs["open_thinking"] = True
        if tools:
            template_kwargs["tools"] = tools
        new_prompt = tokenizer.apply_chat_template(st.session_state.chat_messages, **template_kwargs)

        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {
            "input_ids": inputs.input_ids,
            "max_length": inputs.input_ids.shape[1] + st.session_state.max_new_tokens,
            "num_return_sequences": 1,
            "do_sample": True,
            "attention_mask": inputs.attention_mask,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "temperature": st.session_state.temperature,
            "top_p": 0.85,
            "streamer": streamer,
        }

        Thread(target=model.generate, kwargs=generation_kwargs).start()

        answer = ""
        for new_text in streamer:
            answer += new_text
            placeholder.markdown(process_assistant_content(answer, is_streaming=True), unsafe_allow_html=True)

        full_answer = answer
        for _ in range(16):
            tool_calls = re.findall(r'<tool_call>(.*?)</tool_call>', answer, re.DOTALL)
            if not tool_calls:
                break
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            tool_results = []
            for tc_str in tool_calls:
                try:
                    tc = json.loads(tc_str.strip())
                    result = execute_tool(tc.get('name', ''), tc.get('arguments', {}))
                    st.session_state.chat_messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
                    tool_results.append(f'<div style="background: rgba(90, 130, 110, 0.20); border: 1px solid rgba(150, 200, 170, 0.30); padding: 10px 12px; border-radius: 12px; margin: 6px 0;"><div style="font-size:12px;opacity:.75;display:block;margin:0 0 6px 0;line-height:1;">ToolCalled</div><div><b>{tc.get("name", "")}</b>: {json.dumps(result, ensure_ascii=False)}</div></div>')
                except:
                    pass
            full_answer += "\n" + "\n".join(tool_results) + "\n"
            placeholder.markdown(process_assistant_content(full_answer, is_streaming=True), unsafe_allow_html=True)
            new_prompt = tokenizer.apply_chat_template(st.session_state.chat_messages, **template_kwargs)
            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs["input_ids"] = inputs.input_ids
            generation_kwargs["attention_mask"] = inputs.attention_mask
            generation_kwargs["max_length"] = inputs.input_ids.shape[1] + st.session_state.max_new_tokens
            generation_kwargs["streamer"] = streamer
            Thread(target=model.generate, kwargs=generation_kwargs).start()
            answer = ""
            for new_text in streamer:
                answer += new_text
                placeholder.markdown(process_assistant_content(full_answer + answer, is_streaming=True), unsafe_allow_html=True)
            full_answer += answer
        answer = full_answer

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})

        save_current_chat(username)

# ---------- 动态扫描模型目录 ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = {}
for d in sorted(os.listdir(script_dir), reverse=True):
    full_path = os.path.join(script_dir, d)
    if os.path.isdir(full_path) and not d.startswith('.') and not d.startswith('_'):
        if any(f.endswith(('.bin', '.safetensors', '.pt')) or os.path.exists(os.path.join(full_path, 'model.safetensors.index.json')) for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))):
            MODEL_PATHS[d] = [d, d]
if not MODEL_PATHS:
    MODEL_PATHS = {"No models found": ["", "No models"]}

if __name__ == "__main__":
    main()
