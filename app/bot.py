import os
import asyncio
from telegram.ext import Application, MessageHandler, ContextTypes, CommandHandler, filters
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup
from telegram.constants import ChatAction
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from collections import defaultdict
import concurrent.futures
import json
from datetime import datetime
import logging
import time

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

# --- INITIALIZE VECTOR DB FOR EMBEDDING SEARCH ---
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="chroma_txt_db", embedding_function=embedding)
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# --- PROMPTS FOR CONVERSATIONAL RETRIEVAL CHAIN ---
CONDENSE_PROMPT = PromptTemplate.from_template("""
Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.

Chat History:
{chat_history}

Follow-up question: {question}
Standalone question:
""")

QA_PROMPT = PromptTemplate.from_template("""
You are Antos Pinnacles’ (AP) virtual assistant — a professional and helpful AI trained to support clients of a licensed brokerage firm.

Use the context below to answer the customer's question accurately and concisely. If the information cannot be found in the context, politely admit it.

Context:
{context}

Question: {question}

Instructions:
- Be concise, clear, and professional.
- If the answer is not available in the context, reply with: "I'm sorry, I don't have that information right now."

Only provide the final answer.
""")

# --- INITIALIZE QA CHAIN USING GROQ + LANGCHAIN ---
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=MODEL_NAME)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    condense_question_prompt=CONDENSE_PROMPT,
    combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    return_source_documents=False
)

# --- PREDEFINED PDF GUIDE LINKS ---
pdf_descriptions = {
    "Change member account password": "https://ik.imagekit.io/ddddnzy/CHANGE%20MEMBER%20ACCOUNT%20PASSWORD.pdf",
    "Change MT5 password": "https://ik.imagekit.io/ddddnzy/CHANGE%20MT5%20PASSWORD.pdf",
    "Create bank account": "https://ik.imagekit.io/ddddnzy/CREATE%20BANK%20ACCOUNT.pdf",
    "Create new MT5": "https://ik.imagekit.io/ddddnzy/CREATE%20NEW%20MT5.pdf",
    "Deposit and withdrawal": "https://ik.imagekit.io/ddddnzy/DEPOSIT%20AND%20WITHDRAWAL.pdf",
    "Forgot password": "https://ik.imagekit.io/ddddnzy/FORGOT%20PASSWORD.pdf",
    "Internal transfer": "https://ik.imagekit.io/ddddnzy/INTERNAL%20TRANSFER.pdf",
    "Introduce new client": "https://ik.imagekit.io/ddddnzy/INTRODUCE%20NEW%20CLIENT.pdf",
    "KYC": "https://ik.imagekit.io/ddddnzy/KYC.pdf",
    "Language setting": "https://ik.imagekit.io/ddddnzy/LANGUAGE.pdf",
    "PAMM system": "https://ik.imagekit.io/ddddnzy/PAMM%20SYSTEM.pdf",
    "Report": "https://ik.imagekit.io/ddddnzy/REPORT.pdf",
    "Sign up": "https://ik.imagekit.io/ddddnzy/SIGN%20UP.pdf",
    "Type of Comm Plan Setting": "https://ik.imagekit.io/ddddnzy/Type%20of%20Comm%20Plan%20Setting.pdf",
}

# --- GENERATE TELEGRAM KEYBOARD FOR PDF OPTIONS ---
def generate_pdf_keyboard():
    pdf_buttons = []
    temp_row = []
    for i, title in enumerate(pdf_descriptions.keys(), 1):
        temp_row.append(KeyboardButton(title))
        if i % 2 == 0:
            pdf_buttons.append(temp_row)
            temp_row = []
    if temp_row:
        pdf_buttons.append(temp_row)
    pdf_buttons.append([KeyboardButton("💬 Chat with AI Assistant")])
    return ReplyKeyboardMarkup(pdf_buttons, resize_keyboard=True)

# --- SETUP EMBEDDING MODEL FOR PDF TITLE MATCHING ---
model = SentenceTransformer("all-MiniLM-L6-v2")
pdf_titles = list(pdf_descriptions.keys())

# --- CONTAINERS FOR USER CONTEXT ---
pdf_embeddings = model.encode(pdf_titles, convert_to_tensor=True)
user_chat_history = defaultdict(list)
executor = concurrent.futures.ThreadPoolExecutor(max_workers=20)
chat_enabled_users = set()
COOLDOWN_SECONDS = 5
user_last_seen = {}
user_warned = {}

# --- INITIALIZE TELEGRAM BOT APPLICATION ---
telegram_app = Application.builder().token(TELEGRAM_BOT_TOKEN).concurrent_updates(True).build()

# --- SAVE CHAT TO LOCAL FILE (LOGGING) ---
def save_message_to_local_db(user_id, message, role="user"):
    path = "chat_history.json"
    entry = {
        "user_id": user_id,
        "message": message,
        "role": role,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    history = []
    if os.path.exists(path):
        with open(path, "r") as f:
            try:
                history = json.load(f)
            except:
                pass
    history.append(entry)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)

# --- FIND PDF LINK BASED ON USER QUESTION ---
def get_relevant_pdf_link(question, threshold=0.6):
    q_embed = model.encode(question, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(q_embed, pdf_embeddings)[0]
    best_score = float(scores.max())
    if best_score >= threshold:
        return pdf_descriptions[pdf_titles[int(scores.argmax())]]
    return None

# --- WRAPPER TO CALL QA CHAIN SAFELY WITH TIMEOUT ---
async def safe_invoke_qa(query, user_id, timeout=10):
    loop = asyncio.get_event_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(executor, qa_chain.invoke, {
                "question": query,
                "chat_history": user_chat_history[user_id]
            }), timeout=timeout
        )
    except asyncio.TimeoutError:
        return {"answer": "I'm sorry, the server is taking too long to respond. Please try again later."}
    except Exception:
        return {"answer": "Something went wrong while processing your request."}

# --- /start COMMAND HANDLER ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    welcome_message = (
        "👋 Welcome to Antos Pinnacles’ (AP) virtual assistant!\n\n"
        "Here are some quick PDF guides to get you started."
    )
    await update.message.reply_text(welcome_message, reply_markup=generate_pdf_keyboard())
    await context.bot.send_message(chat_id=update.effective_chat.id, text=welcome_message)
    if user_id in chat_enabled_users:
        chat_enabled_users.remove(user_id)

# --- HANDLE INCOMING TEXT MESSAGES ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    user_query = update.message.text

    # If user taps "Chat with AI Assistant", enable chat mode
    if user_query == "💬 Chat with AI Assistant":
        chat_enabled_users.add(user_id)
        await context.bot.send_message(chat_id=chat_id, text="💬 AI Assistant is now enabled. 👋 Hello and welcome to Antos Pinnacles’ (AP) virtual assistant!")
        return

    # If chat not enabled, only try to match PDF guide
    if user_id not in chat_enabled_users:
        pdf_link = get_relevant_pdf_link(user_query)
        if pdf_link:
            await context.bot.send_message(chat_id=chat_id, text=f"📄 Here's the guide:\n{pdf_link}")
        else:
            await context.bot.send_message(chat_id=chat_id, text="❌ Please click '💬 Chat with AI Assistant'(Bottom of the table) before sending questions.")
        return

    # If chat is enabled, proceed with AI + PDF response
    try:
        save_message_to_local_db(user_id, user_query, "user")
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        await asyncio.sleep(0.5)

        pdf_link = get_relevant_pdf_link(user_query)
        result = await safe_invoke_qa(user_query, user_id)
        answer = result.get("answer", "Sorry, I couldn't find a response.")

        user_chat_history[user_id].append((user_query, answer))
        await context.bot.send_message(chat_id=chat_id, text=answer)
        save_message_to_local_db(user_id, answer, "bot")

        if pdf_link:
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            await asyncio.sleep(0.5)
            await context.bot.send_message(chat_id=chat_id, text=f"📄 Related guide:\n{pdf_link}")
            save_message_to_local_db(user_id, f"📄 Related guide:\n{pdf_link}", "bot")

    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"⚠️ Error: {str(e)}")

# --- REGISTER TELEGRAM HANDLERS ---
telegram_app.add_handler(CommandHandler("start", start_command))
telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

# --- START TELEGRAM BOT WITH WEBHOOK ---
async def start_telegram():
    await telegram_app.initialize()
    await telegram_app.start()
    await telegram_app.bot.set_webhook(url=f"{WEBHOOK_URL}/webhook/{TELEGRAM_BOT_TOKEN}")