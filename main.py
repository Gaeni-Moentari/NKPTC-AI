import os
import streamlit as st
from dotenv import load_dotenv
from crewai import Crew, Agent, Task
from crewai_tools import WebsiteSearchTool
from langchain.chat_models import ChatOpenAI
from urlbase import url
load_dotenv()

openai_model = "gpt-3.5-turbo"
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("❌ OPENAI_API_KEY is not set. Please check your .env file.")
elif not openai_model:
    st.error("❌ OPENAI_MODEL_NAME is not set. Please check your .env file.")
else:
    web_rag_tool = WebsiteSearchTool()
    
    llm = ChatOpenAI(
        model=openai_model,
        api_key=openai_api_key,
        temperature=0.2,
    )

    st.set_page_config(
        page_title="AI NKPTC - วิทยาลัยเทคนิคนครพนม",
        page_icon="🎓",
        layout="centered",
    )

    st.markdown("""
        <style>
            [data-testid="stAppViewContainer"] {
                background-color: #ffffff;
            }
            [data-testid="stHeader"] {
                background-color: #ffffff;
            }
            [data-testid="stToolbar"] {
                background-color: #ffffff;
            }
            [data-testid="stSidebar"] {
                background-color: #f8f9fa;
            }
            .stTextInput > div > div > input {
                background-color: #ffffff;
                color: #000000 !important;
                text-align: center;
            }
            h1 {
                color: #000000 !important;
                text-align: center;
            }
            p {
                color: #000000 !important;
                text-align: center;
            }
            .stMarkdown {
                color: #000000;
            }
            .stButton button {
                background-color: #0066CC;
                color: #ffffff;
            }
            .stTextInput label {
                color: #000000 !important;
            }
            .language-selector {
                margin-bottom: 20px;
            }
        </style>
    """, unsafe_allow_html=True)

    left_co, cent_co, last_co = st.columns([1,3,1])
    with cent_co:
        st.image('logo-nptc-full.png', width=500)

    
    st.title("ผู้ช่วยแนะนำการเรียนวิทยาลัยเทคนิคนครพนม")
    st.title("NKPTC Education Assistant")

    st.markdown("""
        ยินดีต้อนรับสู่ผู้ช่วยแนะนำการเรียนวิทยาลัยเทคนิคนครพนม
        กรุณาป้อนคำถามของคุณในช่องด้านล่างนี้ เราจะช่วยคุณหาข้อมูลที่คุณต้องการ
        
        Welcome to NKPTC Education Assistant.
        Please enter your question below and we will help you find the information you need.
    """)

    question = st.text_input("คำถามของคุณ / Your Question:", "")

    search_button = st.button("ค้นหา / Search")

    asisten_pelatihan = Agent(
        role='ผู้ช่วยการเรียนวิทยาลัยเทคนิคนครพนม',
        goal=f'ให้ข้อมูลทั้งหมดเกี่ยวกับ {question}? ที่วิทยาลัยเทคนิคนครพนม',
        backstory='ผู้ช่วยที่มีความรู้กว้างขวางและทุ่มเทเพื่อช่วยผู้ใช้ค้นหาข้อมูลการเรียนที่เหมาะสม',
        tools=[web_rag_tool],
        verbose=True,
        memory=True
    )
    
    riset_program_pelatihan = Task(
        description=(
            f'ค้นหาข้อมูลเกี่ยวกับ {question}? ที่วิทยาลัยเทคนิคนครพนม '
            f"เราจะมุ่งเน้นการสำรวจข้อมูลล่าสุดและถูกต้องจาก {url}"
        ),
        expected_output=(
            f"""
            ✨ ให้รายงานสรุปเป็นภาษาไทย:
            
            📝 สูงสุด 3 บรรทัดต่อย่อหน้า
            🎯 มุ่งเน้นการตอบคำถามเกี่ยวกับ {question}
            ❌ แจ้งว่าไม่พบข้อมูล หากไม่มีข้อมูลที่ถูกต้อง
            
            รูปแบบคำตอบ:
            
            💡 [หัวข้อที่น่าสนใจ]
            
            [ย่อหน้า 1 - สูงสุด 3 บรรทัด]
            [ย่อหน้า 2 - สูงสุด 3 บรรทัด] (ถ้ามี)
            
            🔍 แหล่งที่มา: [ลิงก์ที่ถูกต้อง]
            """
        ),
        agent=asisten_pelatihan,
        tools=[web_rag_tool]
    )
    
    if search_button:
        if not question:
            st.error("❌ กรุณาป้อนคำถามของคุณ")
        else:
            crew = Crew(
                agents=[asisten_pelatihan],
                tasks=[riset_program_pelatihan],
                verbose=True,
                manager_llm=llm
            )

            with st.spinner("🔄..."):
                result = crew.kickoff()

            st.markdown(result)
