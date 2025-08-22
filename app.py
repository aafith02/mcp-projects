import os
import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
load_dotenv()

async def main():
    bright_api_token = os.getenv("BRIGHT_DATA_API_TOKEN") 
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not bright_api_token:
        print("Error: Missing BRIGHT_DATA_API_TOKEN environment variable.")
        return
    if not google_api_key:
        print("Error: Missing GOOGLE_API_KEY environment variable.")
        return

    client = MultiServerMCPClient(
        {
            "Bright_data": { # Ensure this matches the key expected by MultiServerMCPClient
                "command": "npx",
                "args": ["@brightdata/mcp"],
                "transport": "stdio",
                "env": {"API_TOKEN": bright_api_token}, # Use bright_api_token
            }
        }
    )

    tools = await client.get_tools()

    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        api_key=google_api_key
    )
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt="You are a web scraping agent. Your primary task is to find specific train information for a given date. First, you must use your search tool to find the most relevant train ticketing website for the specified route and date. Then, you must use your scraping tool to extract and return the following details for each available train: **train name**, **ticket price**, **departure and arrival times**, and **available seats**. If direct ticket availability is not found, you must specifically look for and return any **waiting list (WL) numbers** and their **confirmation chances**. Your final output should be a clear, structured summary of all this information for the requested date."
    )
    response = await agent.ainvoke({"messages": [HumanMessage(content="What are the trains available from Chennai to tenkasi on August 22, 2025?")]})
    print(response["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())