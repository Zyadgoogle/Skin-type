import streamlit as st
import time
from typing import Optional
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# --- 1. Pydantic Models ---
class DailyRoutine(BaseModel):
    morning: list[str] = Field(description="Morning skincare steps in order")
    evening: list[str] = Field(description="Evening skincare steps in order")

class ProductRecommendations(BaseModel):
    recommended: list[str] = Field(description="Recommended product types")
    avoid: list[str] = Field(description="Product types to avoid")

class IngredientGuide(BaseModel):
    look_for: list[str] = Field(description="Beneficial ingredients to look for")
    avoid: list[str] = Field(description="Harmful ingredients to avoid")

class SkinRecommendation(BaseModel):
    summary: str = Field(description="Brief summary of the skin analysis")
    daily_routine: DailyRoutine = Field(description="Morning and evening routine steps")
    products: ProductRecommendations = Field(description="Product recommendations")
    ingredients: IngredientGuide = Field(description="Ingredient guidance")
    lifestyle_tips: list[str] = Field(description="Lifestyle tips for skin health")

# --- 2. Agent Class ---
class SkinEAgent:
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.llm = ChatGroq(api_key=api_key, model=model, temperature=0.7)
        self.parser = PydanticOutputParser(pydantic_object=SkinRecommendation)
        
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history", 
            return_messages=True, 
            k=5,
            input_key="input",
            output_key="text"
        )
        
        self.skin_type = None
        self.condition = None
        self.analysis_done = False
        self.recommendation = None

        # Intake Chain
        intake_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are SkinE, a helpful skincare advisor. Ask the user for their skin type and condition if you don't know them."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        self.intake_chain = LLMChain(llm=self.llm, prompt=intake_prompt, memory=self.memory)

        # Analysis Chain
        rec_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("Provide a structured routine.\n{format_instructions}"),
            HumanMessagePromptTemplate.from_template("Skin Type: {skin_type}\nCondition: {condition}")
        ]).partial(format_instructions=self.parser.get_format_instructions())
        self.analysis_chain = LLMChain(llm=self.llm, prompt=rec_prompt)

        # Chat Chain
        chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are SkinE. You already analyzed the user's {skin_type} skin with {condition}. "
                "Context summary: {summary}"
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        self.chat_chain = LLMChain(llm=self.llm, prompt=chat_prompt, memory=self.memory)

    def _extract_info(self, text: str):
        extract_prompt = (
            f"Given: '{text}', does it specify a skin type and condition? "
            "Return ONLY 'YES: [type], [condition]' or 'NO'."
        )
        res = self.llm.invoke(extract_prompt).content
        if "YES" in res.upper():
            try:
                parts = res.split(":")[-1].split(",")
                self.skin_type = parts[0].strip()
                self.condition = parts[1].strip()
                return True
            except: return False
        return False

    def chat(self, user_input: str):
        if not self.analysis_done:
            if self._extract_info(user_input):
                raw_rec = self.analysis_chain.predict(skin_type=self.skin_type, condition=self.condition)
                self.recommendation = self.parser.parse(raw_rec)
                self.analysis_done = True
                
                res = f"Got it! Based on your **{self.skin_type}** skin and **{self.condition}**, here is a routine for you:\n\n"
                res += f"**Summary:** {self.recommendation.summary}\n\n"
                res += "**🌅 Morning Routine:**\n- " + "\n- ".join(self.recommendation.daily_routine.morning) + "\n\n"
                res += "**🌙 Evening Routine:**\n- " + "\n- ".join(self.recommendation.daily_routine.evening) + "\n\n"
                res += "**✅ Recommended Products:** " + ", ".join(self.recommendation.products.recommended) + "\n\n"
                res += "**⚠️ Avoid:** " + ", ".join(self.recommendation.products.avoid) + "\n\n"
                res += "What else would you like to know?"
                
                self.memory.chat_memory.add_user_message(user_input)
                self.memory.chat_memory.add_ai_message(res)
                return res
            else:
                return self.intake_chain.predict(input=user_input)
        else:
            return self.chat_chain.predict(
                input=user_input, 
                skin_type=self.skin_type, 
                condition=self.condition, 
                summary=self.recommendation.summary
            )

# --- 3. Streamlit UI ---
st.set_page_config(page_title="SkinE - AI Skincare Advisor", page_icon="✨", layout="centered")

st.title("✨ SkinE - AI Skincare Advisor")
st.caption("Your personal dermatologist powered by AI")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    model_name = st.selectbox("Select Model", ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"])
    
    st.divider()
    st.markdown("Made with ❤️ using Streamlit & LangChain")

# Main Content Lock
if not api_key:
    st.warning("👈 Please enter your Groq API Key in the sidebar to start!")
    st.stop()

# Initialize Agent & Session State
if "agent" not in st.session_state or st.session_state.get("current_model") != model_name:
    st.session_state.agent = SkinEAgent(api_key=api_key, model=model_name)
    st.session_state.current_model = model_name
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your AI Skincare Advisor. How can I help you today? \n\n*(You can tell me your skin type and what condition you suffer from!)*"}
    ]

# --- 4. Image Uploader (Optional Flow) ---
st.markdown("### 📸 Skin Analysis (Optional)")
uploaded_file = st.file_uploader("Upload a close-up photo of your skin for automatic analysis...", type=["jpg", "png", "jpeg"])

if uploaded_file and not st.session_state.agent.analysis_done:
    with st.spinner("Analyzing image using EfficientNet..."):
        time.sleep(2) # Simulating processing time
        
        # Simulate extracted info from the model
        simulated_input = "I have oily skin and I suffer from acne."
        
        # We pass the simulated text to the chat
        st.session_state.messages.append({"role": "user", "content": "[Image Uploaded] Please analyze my skin."})
        response = st.session_state.agent.chat(simulated_input)
        st.session_state.messages.append({"role": "assistant", "content": f"**Analysis Complete!** I detected oily skin with acne.\n\n{response}"})
        st.rerun() # Refresh page to update chat UI

st.divider()

# --- 5. Chat Interface ---
st.markdown("### 💬 Chat")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input box
if prompt := st.chat_input("Chat with SkinE (e.g., 'I have dry skin and redness')..."):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.agent.chat(prompt)
            st.markdown(response)
            
    # Add assistant message to state
    st.session_state.messages.append({"role": "assistant", "content": response})