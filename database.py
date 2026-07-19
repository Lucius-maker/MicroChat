from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DB_PATH = "microchat.db"
engine = create_engine(f'sqlite:///{DB_PATH}', echo=False)
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), index=True)
    session_id = Column(String(50), index=True)
    role = Column(String(10))
    message = Column(Text)
    timestamp = Column(DateTime, default=datetime.now)

def init_db():
    Base.metadata.create_all(bind=engine)

def save_message(username, session_id, role, message):
    db = SessionLocal()
    try:
        conv = Conversation(username=username, session_id=session_id, role=role, message=message)
        db.add(conv)
        db.commit()
    finally:
        db.close()

def load_history(username, session_id, limit=50):
    db = SessionLocal()
    try:
        messages = db.query(Conversation).filter(
            Conversation.username == username,
            Conversation.session_id == session_id
        ).order_by(Conversation.timestamp).limit(limit).all()
        return [(m.role, m.message) for m in messages]
    finally:
        db.close()

def get_sessions(username):
    db = SessionLocal()
    try:
        sessions = db.query(Conversation.session_id).filter(
            Conversation.username == username
        ).distinct().all()
        return [s[0] for s in sessions]
    finally:
        db.close()

def delete_session(username, session_id):
    db = SessionLocal()
    try:
        db.query(Conversation).filter(
            Conversation.username == username,
            Conversation.session_id == session_id
        ).delete()
        db.commit()
    finally:
        db.close()
