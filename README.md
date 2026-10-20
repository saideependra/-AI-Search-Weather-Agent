# AI Search + Weather Agent

A simple AI agent with a Streamlit chat UI. It can search the web and check live weather to answer your questions.

## What it does

- Answers questions using live web search (Tavily)
- Gets current weather for any city (Weatherstack)
- Uses a ReAct agent (think, act, observe, repeat) powered by Groq
- Shows which tools the agent used for each answer

## Tech used

- **Python 3.10+**
- **Streamlit** for the UI
- **LangChain** for the agent
- **Groq** (`openai/gpt-oss-120b`) as the language model
- **Tavily** for web search
- **Weatherstack** for weather data

## Files

- `main.py` - the Streamlit app (run this one)
- `app.py` - the original command-line version with smoke tests
- `.env` - your API keys (you create this, never commit it)

## Setup

### 1. Install packages

```bash
pip install streamlit langchain langchain-classic langchain-community langchain-groq tavily-python python-dotenv requests certifi
```

### 2. Get API keys

- Groq: https://console.groq.com
- Tavily: https://tavily.com
- Weatherstack: https://weatherstack.com

### 3. Create a `.env` file

Put it in the same folder as `main.py`:

```
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
WEATHERSTACK_API_KEY=your_weatherstack_key
```

### 4. Run the app

```bash
streamlit run main.py
```

It opens at http://localhost:8501.

## How to use

- Type a question in the chat box
- Or click an example question in the sidebar
- Open the "Tools used" section under an answer to see what the agent did
- Click "Clear chat" in the sidebar to start over

Example questions:

- What is the capital of India? Also tell me its current weather.
- What is the weather in Mumbai right now?
- Who is the current Prime Minister of India?

## How it works

1. You send a question.
2. The agent decides if it needs a tool.
3. It calls web search or the weather tool.
4. It reads the result and decides what to do next.
5. It stops after giving a final answer, or after 6 steps at most.

## Known limits

- **No memory.** Each question is handled alone. Follow-ups like "and what about Mumbai?" will not work.
- **Weatherstack free plan** only allows `http://`, not `https://`. If the weather tool fails with a plan or access error, change the URL in `main.py` to `http://api.weatherstack.com/current` or upgrade your plan.
- **ReAct can break.** The model sometimes does not follow the exact format. The app retries, but some runs may still fail.
- **API limits.** Free plans on Groq, Tavily and Weatherstack have usage limits. If you hit them, you will see a "Something went wrong" message.

## Troubleshooting

- **"Missing in your .env file"** - check the key names and that `.env` is in the same folder you run the command from.
- **Weather errors** - see the Weatherstack note above.
- **SSL errors** - the app already points to `certifi` certificates. Update it with `pip install -U certifi`.
- **Agent stops without an answer** - it hit the 6-step limit. Ask a simpler question or raise `max_iterations` in `main.py`.

## Security

- Never commit your `.env` file. Add it to `.gitignore`:

```
.env
```