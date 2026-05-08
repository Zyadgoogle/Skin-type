# ✨ SkinE - AI-Powered Skincare Advisor

SkinE is an intelligent, conversational AI skincare advisor built with **Python, LangChain, and Streamlit**. It acts as a virtual dermatologist, analyzing a user's skin type and concerns to provide highly structured, personalized skincare routines, product recommendations, and lifestyle advice. 

Unlike standard chatbots, SkinE uses a multi-chain LangChain architecture with conversational memory and strict JSON output parsing to ensure responses are scientifically grounded, consistent, and context-aware.

## 🚀 Features

* **🗣️ Conversational Intake:** If the system doesn't know your skin type, it politely chats with you to figure it out.
* **📋 Structured Recommendations:** Utilizes Pydantic output parsers to force the LLM to generate strict, easy-to-read routines (Morning/Evening, Ingredients to Avoid, Recommended Products).
* **🧠 Contextual Memory:** Remembers your skin profile and the AI's previous advice, allowing you to ask follow-up questions like *"Why should I avoid that specific ingredient?"* without repeating yourself.
* **📸 Image Analysis Simulation:** Includes a UI component for uploading skin photos (simulating visual AI intake).
* **🎨 Beautiful GUI:** A modern, user-friendly chat interface built completely in Streamlit.

## 🛠️ Tech Stack

* **Language:** Python 3.9+
* **Frontend UI:** [Streamlit](https://streamlit.io/)
* **AI Orchestration:** [LangChain](https://www.langchain.com/) (LLMChains, PromptTemplates, ConversationBufferWindowMemory)
* **LLM Provider:** [Groq API](https://groq.com/) (`llama-3.3-70b-versatile`)
* **Data Validation:** Pydantic

## ⚙️ Installation & Setup

Follow these steps to run the project locally on your machine.

### 1. Clone the repository
```bash
git clone [https://github.com/YOUR_USERNAME/SkinE-AI-Advisor.git](https://github.com/YOUR_USERNAME/SkinE-AI-Advisor.git)
cd SkinE-AI-Advisor
