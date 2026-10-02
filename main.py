import os
import re
import json
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
api_key = os.getenv("GROQ_API_KEY") or os.getenv("API_KEY")

AVAILABLE_MODELS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b"
]

@st.cache_resource
def get_groq_client(key):
    return OpenAI(
        api_key=key,
        base_url="https://api.groq.com/openai/v1",
    )

def clean_latex(text: str) -> str:
    """
    Normalize common LLM math delimiters so Streamlit renders the expressions.
    """
    if not text:
        return text

    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)

    # Some models put LaTeX in ordinary parentheses instead of math delimiters.
    pairs = []
    open_parens = []
    for index, character in enumerate(text):
        if character == "(":
            open_parens.append(index)
        elif character == ")" and open_parens:
            start = open_parens.pop()
            expression = text[start + 1:index]
            commands = re.sub(r'\\[A-Za-z]+', '', expression)
            if (
                re.search(r'\\[A-Za-z]+', expression)
                and not re.search(r'[A-Za-z]{3,}', commands)
                and text[:start].count("$") % 2 == 0
            ):
                pairs.append((start, index))

    outer_pairs = [
        pair for pair in pairs
        if not any(start < pair[0] and end > pair[1] for start, end in pairs)
    ]
    for start, end in reversed(outer_pairs):
        text = text[:start] + "$" + text[start + 1:end] + "$" + text[end + 1:]

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
        "Primary AI Model:",
        AVAILABLE_MODELS,
        index=0,
        help="Main model used for Socratic dialogue generation."
    )

    fallback_options = [m for m in AVAILABLE_MODELS if m != selected_model]
    fallback_model = st.selectbox(
        "Fallback AI Model:",
        fallback_options,
        index=0,
        help="Automatically engaged if the primary model faces rate limits or downtime."
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
    
    # Concept Progress Tracker (AI-Driven)
    st.subheader("💡 Session Progress")
    st.caption("🤖 Concepts are automatically extracted by AI as you converse.")

    if st.session_state.learned_concepts:
        st.write(f"**Mastered Topics ({len(st.session_state.learned_concepts)}):**")
        for concept in st.session_state.learned_concepts:
            st.markdown(f"- ✅ `{concept}`")
    else:
        st.caption("No topics tracked yet. Start learning to see AI-extracted concepts here!")

    with st.expander("➕ Add concept manually (optional)"):
        manual_concept = st.text_input("Concept title:", placeholder="e.g., Integration by Parts", key="manual_concept_input")
        if st.button("➕ Log Manually") and manual_concept:
            clean_concept = manual_concept.strip()
            if clean_concept and clean_concept.lower() not in [c.lower() for c in st.session_state.learned_concepts]:
                st.session_state.learned_concepts.append(clean_concept)
                st.rerun()

    st.divider()
    
    if st.button("🗑️ Clear Chat History", type="secondary"):
        st.session_state.messages = []
        st.session_state.learned_concepts = []
        st.rerun()


# -----------------------------------------------------------------------------
# 5. Socratic Prompting, Fallback Handler & AI Concept Extraction
# -----------------------------------------------------------------------------
def construct_system_prompt(subject_name, style):
    return f"""You are an expert, supportive Socratic AI Tutor specializing in {subject_name}.
Your goal is to lead the user to discover solutions on their own through guided questioning.

PEDAGOGICAL RULES ({style}):
1. NEVER directly solve the user's homework, write full production code solutions, or give final numerical answers.
2. Structure your response in two parts:
   - Part 1: Acknowledge their response or provide a minimal hint/context.
   - Part 2: End with EXACTLY ONE clear, open-ended question that guides them to the next logical step.
3. If the user asks for direct answers or complete code, politely decline and ask a simpler prerequisite question.

FORMATTING RULES FOR MATHEMATICS (CRITICAL):
- Use standard LaTeX delimiters WITHOUT extra escaping:
  * Inline math: single dollar signs `$ ... $` (e.g., $f(x) = x^2 + 2x + 1$).
  * Display math: standalone blocks surrounded by double dollar signs on separate lines:
    $$\\int_0^1 x^2 \\, dx$$
- Do NOT use `\\[ ... \\]` or `\\( ... \\)` brackets. Use ONLY `$` and `$$`.
- Never wrap LaTeX in ordinary parentheses; every mathematical expression must use `$...$` or `$$...$$`.
- Always write explicit multiplication and complete mathematical operators.
"""

def generate_ai_response(client, primary_model, fallback_model, messages, subject_name, style):
    """
    Generates a tutor response using the primary model with automatic fallback
    resilience if rate limits, status errors, or server connection issues arise.
    """
    system_message = {"role": "system", "content": construct_system_prompt(subject_name, style)}
    full_conversation = [system_message] + messages

    def _call_model(model_name):
        return client.chat.completions.create(
            model=model_name,
            messages=full_conversation,
            temperature=0.3,
            max_tokens=450,
        )

    # 1. Attempt generation with primary model
    try:
        response = _call_model(primary_model)
        raw_content = response.choices[0].message.content
        return clean_latex(raw_content), None, False
    except (RateLimitError, APIStatusError, APIConnectionError) as primary_err:
        # 2. Seamlessly failover to fallback model
        if fallback_model and fallback_model != primary_model:
            try:
                response = _call_model(fallback_model)
                raw_content = response.choices[0].message.content
                return clean_latex(raw_content), None, True
            except Exception as fallback_err:
                return None, f"⚠️ **Model Service Failure:**\n- Primary ({primary_model}): {primary_err}\n- Fallback ({fallback_model}): {fallback_err}", False
        else:
            return None, f"⚠️ **Groq API Error ({primary_model}):** {str(primary_err)}", False
    except AuthenticationError:
        return None, "🔑 **Authentication Error:** Invalid Groq API key. Please check `GROQ_API_KEY` in your `.env` file.", False
    except Exception as e:
        return None, f"❌ **Unexpected Error:** {str(e)}", False

def extract_concepts_ai(client, primary_model, fallback_model, messages, existing_concepts):
    """
    Automated AI Concept Extraction:
    Analyzes the recent conversation turns and automatically detects 1-3 specific
    academic/technical concepts or skills the student is practicing or understanding.
    """
    if not messages:
        return []

    recent_dialogue = messages[-4:]
    formatted_chat = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in recent_dialogue])
    existing_str = ", ".join(existing_concepts) if existing_concepts else "None"

    extraction_prompt = f"""You are an educational curriculum assessor.
Analyze the following tutor-student dialogue. Identify 1 to 2 specific educational concepts, principles, techniques, or topics that the student is exploring, practicing, or has demonstrated understanding of.

Rules:
- Concept names should be concise (2 to 4 words, e.g., "Chain Rule", "Binary Search Trees", "Newton's Second Law").
- Do NOT include generic conversational terms like "Question", "Help", "Homework", or "Code".
- Do NOT return concepts already in this tracked list: [{existing_str}].
- Output MUST be a valid JSON array of strings only. Example: ["Breadth-First Search", "Queue Data Structure"]
- If no specific academic or technical concept is identified or newly introduced, return an empty array: []

Dialogue:
{formatted_chat}

JSON Array:"""

    def _call_extraction(model_name):
        return client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0.1,
            max_tokens=80,
        )

    response_text = None
    try:
        resp = _call_extraction(primary_model)
        response_text = resp.choices[0].message.content
    except Exception:
        if fallback_model and fallback_model != primary_model:
            try:
                resp = _call_extraction(fallback_model)
                response_text = resp.choices[0].message.content
            except Exception:
                return []
        else:
            return []

    if not response_text:
        return []

    try:
        match = re.search(r'\[.*?\]', response_text, re.DOTALL)
        if match:
            extracted = json.loads(match.group(0))
            if isinstance(extracted, list):
                cleaned = []
                for c in extracted:
                    if isinstance(c, str):
                        clean_c = c.strip()
                        if (
                            clean_c
                            and clean_c.lower() not in [e.lower() for e in existing_concepts]
                            and clean_c.lower() not in ["none", "null", "[]"]
                        ):
                            cleaned.append(clean_c)
                return cleaned
    except Exception:
        pass

    return []


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
        st.error("🔑 **Groq API Key Missing:** Please configure `GROQ_API_KEY` (or `API_KEY`) in your `.env` file to start tutoring.")
    else:
        client = get_groq_client(api_key)

        # Append User Message
        st.session_state.messages.append({"role": "user", "content": prompt_to_send})
        with st.chat_message("user"):
            st.markdown(clean_latex(prompt_to_send))

        # Generate Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply, error_msg, used_fallback = generate_ai_response(
                    client, 
                    selected_model, 
                    fallback_model,
                    st.session_state.messages, 
                    subject, 
                    guidance_level
                )

                if error_msg:
                    st.error(error_msg)
                else:
                    if used_fallback:
                        st.info(f"ℹ️ Primary model was unavailable; seamlessly switched to fallback model `{fallback_model}`.")
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})

                    # Automated Concept Extraction
                    new_concepts = extract_concepts_ai(
                        client,
                        selected_model,
                        fallback_model,
                        st.session_state.messages,
                        st.session_state.learned_concepts
                    )
                    if new_concepts:
                        st.session_state.learned_concepts.extend(new_concepts)
                        st.toast(f"💡 AI tracked new concept: {', '.join(new_concepts)}")
                        st.rerun()
