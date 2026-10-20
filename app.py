import os
import requests
import certifi

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_classic.agents import create_react_agent, AgentExecutor

# Keep HTTPS certificate verification configured for Python requests.
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is missing. Add it to your .env file.")

if not WEATHERSTACK_API_KEY:
    raise ValueError("WEATHERSTACK_API_KEY is missing. Add it to your .env file.")

print("API keys found (values are hidden).")


search_tool = TavilySearchResults(max_results=3)

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


# Web-search smoke test
search_result = search_tool.invoke({"query": "capital of India"})
print(search_result)

# Weather smoke test
weather_result = get_weather_data.invoke({"city": "New Delhi"})
print(weather_result)


GROQ_MODEL = "openai/gpt-oss-120b"

llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0,
    api_key=GROQ_API_KEY,
    max_retries=2,
)

# Simple connectivity/model-ID test
test_response = llm.invoke("Reply with exactly: Groq connection works.")
print(test_response.content)


react_template = """You are a helpful assistant that can use tools to answer questions accurately.

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

prompt = PromptTemplate.from_template(react_template)
print("Prompt variables:", prompt.input_variables)



tools = [search_tool, get_weather_data]

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence=False,
)

agent_executor = AgentExecutor(
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



result = agent_executor.invoke({
    "input": (
        "Use web search to verify the capital of India. Then call the "
        "get_weather_data tool for New Delhi and report the current weather. "
        "Clearly separate the search result from the weather result."
    )
})

print("\nFINAL ANSWER:\n")
print(result["output"])

print("\nTOOLS USED:")
for action, observation in result.get("intermediate_steps", []):
    print(f"- {action.tool}: {action.tool_input}")