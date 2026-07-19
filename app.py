import random
import re
import json
import os
import hashlib
import threading
from threading import Thread
from pathlib import Path

import torch
import numpy as np
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# ---------- 全局并发锁 ----------
model_lock = threading.Lock()

# ---------- 配置 ----------
MAX_DISPLAY_MESSAGES = 40   # 界面最多显示的消息条数，防止内存爆炸

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

def save_current_chat(username):
    if 'messages' in st.session_state and st.session_state.messages:
        chats = load_user_chats(username)
        current_id = st.session_state.get('current_chat_id')
        # 获取完整消息（从chat_messages中恢复完整历史）
        full_messages = st.session_state.get('chat_messages', st.session_state.messages)
        if current_id is not None:
            for chat in chats:
                if chat['id'] == current_id:
                    chat['messages'] = full_messages.copy()
                    if full_messages:
                        first_user = next((m for m in full_messages if m['role'] == 'user'), None)
                        chat['title'] = first_user['content'][:30] if first_user else "New Chat"
                    break
        else:
            new_id = max([c['id'] for c in chats], default=-1) + 1
            first_user = next((m for m in full_messages if m['role'] == 'user'), None)
            title = first_user['content'][:30] if first_user else "New Chat"
            chats.append({
                "id": new_id,
                "title": title,
                "messages": full_messages.copy()
            })
            st.session_state['current_chat_id'] = new_id
        save_user_chats(username, chats)

# ---------- 页面配置 ----------
st.set_page_config(page_title="MicroChat version1.2", initial_sidebar_state="expanded")

# ---------- 样式：浅色 DeepSeek 风格（白色背景 + 黄色强调） ----------
st.markdown("""
<style>
    .stApp {
        background: #ffffff;
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
        color: #333333;
    }
    h1 {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 600;
        color: #f5b800;
        margin-bottom: 0.3rem;
    }
    .subtitle {
        text-align: center;
        color: #7a7a7a;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    /* 聊天气泡 */
    .stChatMessage {
        background: #f7f7f7;
        border-radius: 16px;
        padding: 12px 18px;
        margin: 6px 0;
        border: 1px solid #eaeaea;
        box-shadow: 0 1px 4px rgba(0,0,0,0.03);
    }
    /* 用户消息 */
    div[data-testid="stChatMessage"]:nth-child(odd) {
        background: #fef5d4;
        border-left: 3px solid #fcd535;
        border-radius: 16px 16px 4px 16px;
        color: #4a4a4a;
    }
    /* 助手消息 */
    div[data-testid="stChatMessage"]:nth-child(even) {
        background: #f9f9f9;
        border-left: 3px solid #e0c050;
        border-radius: 16px 16px 16px 4px;
        color: #333333;
    }
    /* 输入框 */
    .stTextInput > div > div > input {
        background: #ffffff;
        border: 2px solid #fcd535;
        border-radius: 30px;
        padding: 12px 20px;
        font-size: 1rem;
        color: #333;
        box-shadow: 0 1px 4px rgba(252, 213, 53, 0.2);
    }
    .stTextInput > div > div > input:focus {
        border-color: #f5b800;
        box-shadow: 0 4px 12px rgba(245, 184, 0, 0.25);
        outline: none;
    }
    /* 按钮 */
    .stButton > button {
        background: #fcd535;
        color: #222;
        border: none;
        border-radius: 30px;
        padding: 8px 22px;
        font-weight: 600;
        font-size: 0.9rem;
        box-shadow: 0 2px 6px rgba(252, 213, 53, 0.3);
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: #f5b800;
        box-shadow: 0 4px 12px rgba(245, 184, 0, 0.4);
        transform: translateY(-1px);
    }
    /* 侧边栏 */
    .css-1d391kg {
        background: #fafafa;
        border-right: 1px solid #f0e2a0;
    }
    /* 历史对话按钮样式 */
    .stButton button[kind="secondary"] {
        background: #f5f5f5;
        border: 1px solid #e0e0e0;
        color: #444;
        border-radius: 20px;
    }
    /* 滚动条 */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #f5f5f5; }
    ::-webkit-scrollbar-thumb { background: #fcd535; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: #e5b800; }
    .footer {
        text-align: center;
        color: #aaaaaa;
        font-size: 0.75rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #f0e2a0;
    }
</style>
""", unsafe_allow_html=True)

# ---------- 工具函数 ----------
device = "cuda" if torch.cuda.is_available() else "cpu"

LANG_TEXTS = {
    'zh': {
        'settings': 'Model Settings',
        'history_rounds': 'History Rounds',
        'max_length': 'Max Length',
        'temperature': 'Temperature',
        'thinking': 'Thinking',
        'tools': 'Tools',
        'language': 'Language',
        'send': 'Send a message to MicroChat',
        'disclaimer': 'AI-generated content may be inaccurate, please verify',
        'think_tip': 'Adaptive thinking; may be unstable with multi-turn or Tool Call',
        'tool_select': 'Tool Selection (max 4)',
    },
    'en': {
        'settings': 'Model Settings',
        'history_rounds': 'History Rounds',
        'max_length': 'Max Length',
        'temperature': 'Temperature',
        'thinking': 'Thinking',
        'tools': 'Tools',
        'language': 'Language',
        'send': 'Send a message to MicroChat',
        'disclaimer': 'AI-generated content may be inaccurate, please verify',
        'think_tip': 'Adaptive thinking; may be unstable with multi-turn or Tool Call',
        'tool_select': 'Tool Selection (max 4)',
    }
}

def get_text(key):
    lang = st.session_state.get('lang', 'en')
    return LANG_TEXTS.get(lang, {}).get(key, LANG_TEXTS['zh'].get(key, key))

TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "Calculate a math expression", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "Get current time", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "default": "Asia/Shanghai"}}, "required": []}}},
    {"type": "function", "function": {"name": "random_number", "description": "Generate a random number", "parameters": {"type": "object", "properties": {"min": {"type": "integer"}, "max": {"type": "integer"}}, "required": ["min", "max"]}}},
    {"type": "function", "function": {"name": "text_length", "description": "Count text length", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "Unit conversion", "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "Get exchange rate", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "Translate text", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_lang": {"type": "string"}}, "required": ["text", "target_lang"]}}},
]

TOOL_SHORT_NAMES = {
    'calculate_math': 'Math', 'get_current_time': 'Time', 'random_number': 'Random',
    'text_length': 'Count', 'unit_converter': 'Unit', 'get_current_weather': 'Weather',
    'get_exchange_rate': 'Rate', 'translate_text': 'Translate'
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
            return {"result": f"{args.get('city', 'Unknown')}: Sunny, 7~10 C"}
        elif tool_name == 'get_exchange_rate':
            return {"result": f"1 {args.get('from_currency', 'USD')} = 7.2 {args.get('to_currency', 'CNY')}"}
        elif tool_name == 'translate_text':
            return {"result": f"[Translation]: hello world"}
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
                return f'<div style="background: rgba(252, 213, 53, 0.1); border: 1px solid #fcd535; padding: 8px 12px; border-radius: 10px; margin: 4px 0; font-size:0.9em;"><b>ToolCalling</b><br>{name}: {json.dumps(args, ensure_ascii=False)}</div>'
            except:
                return match.group(0)
        content = re.sub(r'<tool_call>(.*?)</tool_call>', format_tool_call, content, flags=re.DOTALL)
    # 思考过程渲染（略，保持不变）
    if is_streaming and st.session_state.get('enable_thinking', False) and '</think>' not in content and '<think>' not in content:
        m = re.search(r'(\n\n(?:I am|Hi|Hello)[^\n]*)', content)
        if m and m.start(1) > 5:
            i = m.start(1)
            think_part = content[:i]
            answer_part = content[i:]
            return f'<details open style="border-left: 2px solid #fcd535; padding-left: 12px; margin: 8px 0;"><summary style="color: #b5a050; cursor:pointer;">Thought</summary><div style="color: #999; font-size:0.95em; max-height:100px; overflow-y:auto;">{think_part.strip()}</div></details>{answer_part}'
        elif len(content) > 5:
            return f'<details open style="border-left: 2px solid #fcd535; padding-left: 12px; margin: 8px 0;"><summary style="color: #b5a050; cursor:pointer;">Thinking...</summary><div style="color: #999; font-size:0.95em; max-height:100px; overflow-y:auto;">{content.strip().replace(chr(10), "<br>")}</div></details>'
    if '<think>' in content and '</think>' in content:
        def format_think(match):
            think_content = match.group(2)
            if think_content.replace('\n', '').strip():
                return f'<details open style="border-left:2px solid #fcd535; padding-left:12px; margin:8px 0;"><summary style="color:#b5a050; cursor:pointer;">Thought</summary><div style="color:#999; font-size:0.95em; max-height:100px; overflow-y:auto;">{think_content.strip()}</div></details>'
            return ''
        content = re.sub(r'(<think>)(.*?)(</think>)', format_think, content, flags=re.DOTALL)
    if '<think>' in content and '</think>' not in content:
        def format_think_in_progress(match):
            tc = match.group(1)
            return f'<details open style="border-left:2px solid #fcd535; padding-left:12px; margin:8px 0;"><summary style="color:#b5a050; cursor:pointer;">Thinking...</summary><div style="color:#999; font-size:0.95em; max-height:100px; overflow-y:auto;">{tc.strip().replace(chr(10), "<br>")}</div></details>'
        content = re.sub(r'<think>(.*?)$', format_think_in_progress, content, flags=re.DOTALL)
    if '<think>' not in content and '</think>' in content:
        def format_think_no_start(match):
            think_content = match.group(1)
            if think_content.replace('\n', '').strip():
                return f'<details open style="border-left:2px solid #fcd535; padding-left:12px; margin:8px 0;"><summary style="color:#b5a050; cursor:pointer;">Thought</summary><div style="color:#999; font-size:0.95em; max-height:100px; overflow-y:auto;">{think_content.strip()}</div></details>'
            return ''
        content = re.sub(r'(.*?)</think>', format_think_no_start, content, flags=re.DOTALL)
    return content

@st.cache_resource
def load_model_tokenizer(model_path):
    if model_path is None or not os.path.exists(model_path):
        repo_id = "Qwen/Qwen2.5-0.5B-Instruct"
        print(f"Loading model from Hugging Face: {repo_id}")
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
    st.markdown("<h1 style='text-align:center;'>Login / Register</h1>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Login", "Register"])
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                if authenticate(username, password):
                    st.session_state['user'] = username
                    st.session_state['current_chat_id'] = None
                    st.session_state.messages = []
                    st.session_state.chat_messages = []
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    with tab2:
        with st.form("register_form"):
            new_user = st.text_input("New Username")
            new_pass = st.text_input("Password", type="password")
            confirm_pass = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Register")
            if submitted:
                if new_pass != confirm_pass:
                    st.error("Passwords do not match")
                elif register_user(new_user, new_pass):
                    st.success("Registration successful, please login")
                else:
                    st.error("Username already exists")

# ---------- 主函数 ----------
def main():
    if 'user' not in st.session_state:
        login_register_page()
        return

    username = st.session_state['user']

    # ---------- 侧边栏 ----------
    with st.sidebar:
        st.write(f"User: {username}")
        if st.button("Logout"):
            save_current_chat(username)
            del st.session_state['user']
            del st.session_state['current_chat_id']
            st.session_state.messages = []
            st.session_state.chat_messages = []
            st.rerun()
        
        st.markdown("---")
        if st.button("New Chat"):
            save_current_chat(username)
            st.session_state.messages = []
            st.session_state.chat_messages = []
            st.session_state['current_chat_id'] = None
            st.rerun()
        
        st.markdown("---")
        st.write("History")
        chats = load_user_chats(username)
        for chat in chats:
            cols = st.columns([5, 1])
            with cols[0]:
                if st.button(chat['title'], key=f"load_{chat['id']}"):
                    save_current_chat(username)
                    full_msgs = chat['messages']
                    st.session_state.messages = full_msgs[-MAX_DISPLAY_MESSAGES:]   # 只加载最近若干条
                    st.session_state.chat_messages = full_msgs
                    st.session_state['current_chat_id'] = chat['id']
                    st.rerun()
            with cols[1]:
                if st.button("X", key=f"del_{chat['id']}"):
                    chats = [c for c in chats if c['id'] != chat['id']]
                    save_user_chats(username, chats)
                    if st.session_state.get('current_chat_id') == chat['id']:
                        st.session_state.messages = []
                        st.session_state.chat_messages = []
                        st.session_state['current_chat_id'] = None
                    st.rerun()

        st.markdown("---")
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
    col1, col2 = st.columns([4, 1])
    with col1:
        chat_title = "New Chat"
        if st.session_state.messages:
            first_user = next((m for m in st.session_state.messages if m['role'] == 'user'), None)
            if first_user:
                chat_title = first_user['content'][:30]
        st.markdown(f"### {chat_title}")
    with col2:
        if st.button("New Chat", key="new_chat_top"):
            save_current_chat(username)
            st.session_state.messages = []
            st.session_state.chat_messages = []
            st.session_state['current_chat_id'] = None
            st.rerun()

    st.markdown("---")

    model, tokenizer = load_model_tokenizer(model_path)

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_messages = []

    # 渲染当前显示的消息（已限制条数）
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            st.markdown(process_assistant_content(message["content"]), unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div style="display:flex; justify-content:flex-end;"><div style="display:inline-block; margin:8px 0; padding:8px 14px; background:#fef5d4; border-radius:22px; color:#333; border:1px solid #fcd535;">{message["content"]}</div></div>',
                unsafe_allow_html=True)

    prompt = st.chat_input(key="input", placeholder=get_text('send'))

    if prompt:
        # 显示用户消息
        st.markdown(
            f'<div style="display:flex; justify-content:flex-end;"><div style="display:inline-block; margin:8px 0; padding:8px 14px; background:#fef5d4; border-radius:22px; color:#333; border:1px solid #fcd535;">{prompt}</div></div>',
            unsafe_allow_html=True)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        # 内存控制：保持显示消息不超过 MAX_DISPLAY_MESSAGES 条
        if len(st.session_state.messages) > MAX_DISPLAY_MESSAGES:
            st.session_state.messages = st.session_state.messages[-MAX_DISPLAY_MESSAGES:]

        placeholder = st.empty()

        random_seed = random.randint(0, 2 ** 32 - 1)
        setup_seed(random_seed)

        tools = [t for t in TOOLS if t['function']['name'] in st.session_state.get('selected_tools', [])] or None
        sys_prompt = [] if tools else [{"role": "system", "content": "You are MicroChat version1.2, a helpful AI assistant."}]
        # chat_messages 用于生成，保持完整历史以便工具调用上下文
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

        with model_lock:
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
                    tool_results.append(f'<div style="background: rgba(252,213,53,0.08); border:1px solid #fcd535; padding:8px 12px; border-radius:10px; margin:4px 0; font-size:0.9em;"><b>ToolCalled</b><br>{tc.get("name","")}: {json.dumps(result, ensure_ascii=False)}</div>')
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
            with model_lock:
                Thread(target=model.generate, kwargs=generation_kwargs).start()
                answer = ""
                for new_text in streamer:
                    answer += new_text
                    placeholder.markdown(process_assistant_content(full_answer + answer, is_streaming=True), unsafe_allow_html=True)
            full_answer += answer
        answer = full_answer

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        # 再次裁剪显示消息
        if len(st.session_state.messages) > MAX_DISPLAY_MESSAGES:
            st.session_state.messages = st.session_state.messages[-MAX_DISPLAY_MESSAGES:]

        save_current_chat(username)

# ---------- 模型扫描 ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = {}
for d in sorted(os.listdir(script_dir), reverse=True):
    full_path = os.path.join(script_dir, d)
    if os.path.isdir(full_path) and not d.startswith('.') and not d.startswith('_'):
        if any(f.endswith(('.bin', '.safetensors', '.pt')) or os.path.exists(os.path.join(full_path, 'model.safetensors.index.json')) for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))):
            MODEL_PATHS[d] = [full_path, d]
if not MODEL_PATHS:
    MODEL_PATHS = {"No models found": ["", "No models"]}

if __name__ == "__main__":
    main()
