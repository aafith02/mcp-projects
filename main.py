import os
import asyncio
import streamlit as st
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import warnings

# Suppress specific warnings related to schema properties
warnings.filterwarnings(
    "ignore",
    message="Key 'additionalProperties' is not supported in schema, ignoring",
    category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message="Key '$schema' is not supported in schema, ignoring",
    category=UserWarning
)

# Load environment variables from .env file
load_dotenv()

# --- Initialize Agent (Cached to avoid re-initialization on every rerun) ---
@st.cache_resource
def initialize_agent():
    bright_api_token = os.getenv("BRIGHT_DATA_API_TOKEN")
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not bright_api_token:
        st.error("Error: Missing BRIGHT_DATA_API_TOKEN environment variable. Please set it in your .env file.")
        return None
    if not google_api_key:
        st.error("Error: Missing GOOGLE_API_KEY environment variable. Please set it in your .env file.")
        return None

    # Initialize MCP client for Bright Data
    client = MultiServerMCPClient(
        {
            "Bright_data": {
                "command": "npx",
                "args": ["@brightdata/mcp"],
                "transport": "stdio",
                "env": {"API_TOKEN": bright_api_token},
            }
        }
    )

    try:
        tools = asyncio.run(client.get_tools())
    except Exception as e:
        st.error(f"Error getting tools from Bright Data: {e}. Please ensure npx and @brightdata/mcp are correctly set up.")
        return None

    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        api_key=google_api_key
    )

    agent_prompt = (
        "You are a web scraping agent. Your primary task is to find specific train information for a given date. "
        "First, you must use your search tool to find the most relevant train ticketing website for the specified route and date. "
        "Then, you must use your scraping tool to extract and return the following details for each available train: "
        "**train name**, **ticket price**, **departure and arrival times**, and **available seats**. "
        "If direct ticket availability is not found, you must specifically look for and return any **waiting list (WL) numbers** "
        "and their **confirmation chances**. Your final output should be a clear, structured summary of all this information "
        "for the requested date."
    )

    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=agent_prompt
    )
    return agent

# --- Streamlit App Layout ---
st.set_page_config(page_title="Transportation Search Agent", layout="wide")
st.title("🚂✈️🚌 Transportation Search Agent")
st.markdown("Enter your query to find train details (names, prices, times, availability, or waitlist) for a specific route and date.")

# Initialize the agent
agent_instance = initialize_agent()

if agent_instance:
    user_query = st.text_input(
        "**Your Query:**",
        placeholder="e.g., What are the trains available from Chennai to tenkasi on August 22, 2025?"
    )

    if st.button("Search Transportation"):
        if user_query:
            with st.spinner("Searching for transportation details... This might take a moment."):
                try:
                    # Run the agent asynchronously
                    response = asyncio.run(agent_instance.ainvoke({"messages": [HumanMessage(content=user_query)]}))
                    st.subheader("Results:")
                    st.markdown(response["messages"][-1].content)
                except Exception as e:
                    st.error(f"An error occurred while running the agent: {e}")
        else:
            st.warning("Please enter a query to search for transportation.")

st.markdown("---")
st.info("Powered by Bright Data for web scraping and Google Gemini for AI reasoning.")

