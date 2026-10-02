import os
import re
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, APIStatusError, AuthenticationError, RateLimitError

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ThinkWise AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling and clean code/math rendering
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1000px;
    }
    
    /* Header styling */
    .tutor-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .tutor-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .tutor-header p {
        margin-top: 0.3rem;
        margin-bottom: 0;
        opacity: 0.9;
    }

    /* Quick Action Button styling tweak */
    div.stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Environment & Client Setup
# -----------------------------------------------------------------------------
load_dotenv()
api_key = os.getenv("API_KEY") or os.getenv("GROQ_API_KEY")

@st.cache_resource
def get_groq_client(key):
    return OpenAI(
        api_key=key,
        base_url="https://api.groq.com/openai/v1",
    )

def clean_latex(text: str) -> str:
    """
    Fixes common escaping issues with LLM-generated LaTeX in Streamlit.
    Ensures math blocks maintain clean double/single dollar signs and properly
    escaped control sequences.
    """
    if not text:
        return text
    
    # Ensure display math $$...$$ is surrounded by newlines for clean KaTeX parsing
    text = re.sub(r'([^\n])\$\$(.*?)\$\$', r'\1\n\n$$\2$$\n\n', text, flags=re.DOTALL)
    
    return text


# -----------------------------------------------------------------------------
# 3. Session State Initialization
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "learned_concepts" not in st.session_state:
    st.session_state.learned_concepts = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# -----------------------------------------------------------------------------
# 4. Sidebar Controls & Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Tutor Settings")
    
    selected_model = st.selectbox(
        "Select AI Model:",
        [
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-120b"
        ],
        index=2
    )
    
    subject = st.selectbox(
        "Learning Domain:",
        ["Computer Science & Coding", "Mathematics & Calculus", "Physics & Sciences", "General Logic & Reasoning", "Other Subjects"]
    )
    
    guidance_level = st.select_slider(
        "Guidance Level:",
        options=["Strict (Questions Only)", "Balanced (Hints + Questions)", "Slightly Direct"],
        value="Balanced (Hints + Questions)"
    )
    
    st.divider()
    
    # Concept Progress Tracker
    st.subheader("💡 Session Progress")
    new_concept = st.text_input("Add mastered concept:", placeholder="e.g., Integration by Parts")
    if st.button("➕ Log Concept") and new_concept:
        if new_concept not in st.session_state.learned_concepts:
            st.session_state.learned_concepts.append(new_concept)
            st.rerun()

    if st.session_state.learned_concepts:
        st.write("**Mastered Topics:**")
        for concept in st.session_state.learned_concepts:
            st.markdown(f"- ✅ `{concept}`")
    else:
        st.caption("No topics logged yet. Track your progress here!")

    st.divider()
    
    if st.button("🗑️ Clear Chat History", type="secondary"):
        st.session_state.messages = []
        st.session_state.learned_concepts = []
        st.rerun()


# -----------------------------------------------------------------------------
# 5. Socratic Prompting & Response Handler
# -----------------------------------------------------------------------------
def construct_system_prompt(subject_name, style):
    return f"""You are an expert, supportive Socratic AI Tutor specializing in {subject_name}.
Your goal is to lead the user to discover solutions on their own through guided questioning.

PEDAGOGICAL RULES ({style}):
1. NEVER directly solve the user's homework, write full production code solutions, or give final numerical answers.
2. Structure your response in two parts:
   - Part 1: Acknowledge their response or provide a minimal hint/context (1-2 sentences max).
   - Part 2: End with EXACTLY ONE clear, open-ended question that guides them to the next logical step.
3. If the user asks for direct answers or complete code, politely decline and ask a simpler prerequisite question.

FORMATTING RULES FOR MATHEMATICS (CRITICAL):
- Use standard LaTeX delimiters WITHOUT extra escaping:
  * Inline math: single dollar signs `$ ... $` (e.g., $f(x) = x^2 + 2x + 1$).
  * Display math: standalone blocks surrounded by double dollar signs on separate lines:
    $$\\int_0^1 x^2 \\, dx$$
- Do NOT use `\\[ ... \\]` or `\\( ... \\)` brackets. Use ONLY `$` and `$$`.
- Always write explicit multiplication and complete mathematical operators.
"""

def generate_ai_response(client, model, messages, subject_name, style):
    system_message = {"role": "system", "content": construct_system_prompt(subject_name, style)}
    full_conversation = [system_message] + messages

    try:
        response = client.chat.completions.create(
            model=model,
            messages=full_conversation,
            temperature=0.3,
            max_tokens=450,
        )
        raw_content = response.choices[0].message.content
        cleaned_content = clean_latex(raw_content)
        return cleaned_content, None

    except AuthenticationError:
        return None, "🔑 **Authentication Error:** Invalid Groq API key. Check your key in `.env` or sidebar."
    except RateLimitError:
        return None, "⏳ **Rate Limit Exceeded:** Too many requests in a short time. Please wait a moment."
    except APIConnectionError:
        return None, "🌐 **Network Error:** Could not connect to Groq servers. Check your internet connection."
    except APIStatusError as e:
        return None, f"⚠️ **Groq API Error ({e.status_code}):** {e.message}"
    except Exception as e:
        return None, f"❌ **Unexpected Error:** {str(e)}"


# -----------------------------------------------------------------------------
# 6. Main UI Layout
# -----------------------------------------------------------------------------
st.markdown("""
<div class="tutor-header">
    <h1>🎓 ThinkWise AI Tutor</h1>
    <p>Guided learning through critical thinking — Helps you analyze and solve problems independently.</p>
</div>
""", unsafe_allow_html=True)

# Quick Socratic Tool Buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("💡 Give me a Micro-Hint"):
        st.session_state.pending_prompt = "I'm stuck. Can you give me a subtle micro-hint or ask a simpler question to help me move forward?"
        st.rerun()

with col2:
    if st.button("🧩 Break Down Step-by-Step"):
        st.session_state.pending_prompt = "Can you break down the current step into smaller, simpler sub-questions?"
        st.rerun()

with col3:
    if st.button("🎯 Practice Question"):
        st.session_state.pending_prompt = f"Give me a beginner-level conceptual practice question related to {subject} with step-by-step math framing."
        st.rerun()

st.divider()

# Render Existing Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(clean_latex(msg["content"]))

# Process User Input
user_input = st.chat_input("Ask a question, enter code/math, or explain your thinking...")

prompt_to_send = None
if st.session_state.pending_prompt:
    prompt_to_send = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
elif user_input:
    prompt_to_send = user_input

if prompt_to_send:
    if not api_key:
        st.error("Please provide a valid Groq API Key in the sidebar or `.env` file to continue.")
    else:
        client = get_groq_client(api_key)

        # Append User Message
        st.session_state.messages.append({"role": "user", "content": prompt_to_send})
        with st.chat_message("user"):
            st.markdown(clean_latex(prompt_to_send))

        # Generate Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply, error_msg = generate_ai_response(
                    client, 
                    selected_model, 
                    st.session_state.messages, 
                    subject, 
                    guidance_level
                )

                if error_msg:
                    st.error(error_msg)
                else:
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})