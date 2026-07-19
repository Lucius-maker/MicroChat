import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

def init_auth():
    """初始化认证系统"""
    if "authenticator" not in st.session_state:
        with open('config.yaml', 'r', encoding='utf-8') as file:
            config = yaml.load(file, Loader=SafeLoader)
        authenticator = stauth.Authenticate(
            config['credentials'],
            config['cookie']['name'],
            config['cookie']['key'],
            config['cookie']['expiry_days']
        )
        st.session_state.authenticator = authenticator
        st.session_state.config = config
    return st.session_state.authenticator

def get_user():
    """获取当前登录用户"""
    if "authentication_status" in st.session_state:
        if st.session_state.authentication_status:
            return st.session_state.username
    return None
