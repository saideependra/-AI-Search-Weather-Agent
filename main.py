import os
import requests
import certifi
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_classic.agents import create_react_agent, AgentExecutor

# Keep HTTPS certificate verification configured for Python requests.
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Search + Weather Agent", page_icon="🌦️", layout="centered")
st.title("🌦️ AI Search + Weather Agent")
st.caption("Ask a question. The agent can search the web and check live weather.")

# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")

missing = [
    name
    for name, value in {
        "GROQ_API_KEY": GROQ_API_KEY,
        "TAVILY_API_KEY": TAVILY_API_KEY,
        "WEATHERSTACK_API_KEY": WEATHERSTACK_API_KEY,
    }.items()
    if not value
]
if missing:
    st.error(f"Missing in your .env file: {', '.join(missing)}")
    st.stop()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@tool
def get_weather_data(city: str) -> str:
    """Get current temperature, weather description, and humidity for a city."""
    try:
        response = requests.get(
            "https://api.weatherstack.com/current",
            params={"access_key": WEATHERSTACK_API_KEY, "query": city},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        return f"Weather request failed for {city}: {exc}"
    except ValueError:
        return "Weather service returned an invalid response."

    if "current" not in data:
        error_info = data.get("error", {}).get("info", "No current-weather data returned.")
        return f"Could not fetch weather for {city}: {error_info}"

    current = data["current"]
    descriptions = current.get("weather_descriptions") or ["Description unavailable"]
    location = data.get("location", {}).get("name", city)
    return (
        f"City: {location}\n"
        f"Temperature: {current.get('temperature', 'N/A')}°C\n"
        f"Weather: {descriptions[0]}\n"
        f"Humidity: {current.get('humidity', 'N/A')}%"
    )


# ---------------------------------------------------------------------------
# Agent (built once, cached across Streamlit reruns)
# ---------------------------------------------------------------------------
GROQ_MODEL = "openai/gpt-oss-120b"

REACT_TEMPLATE = """You are a helpful assistant that can use tools to answer questions accurately.

You have access to these tools:
{tools}

Use exactly this format:

Question: the question to answer
Thought: briefly decide what to do next
Action: one of [{tool_names}]
Action Input: the input for that action
Observation: the tool result
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the answer to the original question

Rules:
- When a tool is needed, output one Action and one Action Input.
- Action must exactly match one of [{tool_names}].
- Do not invent tool results.
- After receiving an Observation, decide whether another tool is needed.
- When finished, output Final Answer.

Question: {input}
Thought:{agent_scratchpad}"""


@st.cache_resource(show_spinner="Starting agent...")
def build_agent_executor() -> AgentExecutor:
    search_tool = TavilySearchResults(max_results=3)

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        api_key=GROQ_API_KEY,
        max_retries=2,
    )

    prompt = PromptTemplate.from_template(REACT_TEMPLATE)
    tools = [search_tool, get_weather_data]

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
        stop_sequence=False,
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=(
            "Your previous response did not follow the required ReAct format. "
            "Use `Action: <tool name>` and `Action Input: <input>` for a tool call, "
            "or `Final Answer: <answer>` when finished."
        ),
        max_iterations=6,
        return_intermediate_steps=True,
    )


agent_executor = build_agent_executor()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.write(f"**Model:** `{GROQ_MODEL}`")
    st.write("**Tools:**")
    st.write("- Tavily web search")
    st.write("- Weatherstack weather")
    st.divider()
    st.write("**Try these:**")
    examples = [
        "What is the capital of India? Also tell me its current weather.",
        "What is the weather in Mumbai right now?",
        "Who is the current Prime Minister of India?",
    ]
    for i, example in enumerate(examples):
        if st.button(example, key=f"example_{i}", use_container_width=True):
            st.session_state["pending_prompt"] = example
    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()


# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []


def render_steps(steps):
    """Show the tools the agent used, inside an expander."""
    if not steps:
        return
    with st.expander(f"🔧 Tools used ({len(steps)})"):
        for action, observation in steps:
            st.markdown(f"**{action.tool}** → `{action.tool_input}`")
            st.code(str(observation), language="text")


for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_steps(message.get("steps"))


# ---------------------------------------------------------------------------
# Handle new input
# ---------------------------------------------------------------------------
typed_prompt = st.chat_input("Ask something...")
user_prompt = typed_prompt or st.session_state.pop("pending_prompt", None)

if user_prompt:
    st.session_state["messages"].append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = agent_executor.invoke({"input": user_prompt})
                answer = result["output"]
                steps = result.get("intermediate_steps", [])
            except Exception as exc:
                answer = f"Something went wrong: {exc}"
                steps = []

        st.markdown(answer)
        render_steps(steps)

    st.session_state["messages"].append(
        {"role": "assistant", "content": answer, "steps": steps}
    )