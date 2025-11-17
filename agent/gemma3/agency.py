from dotenv import load_dotenv
import requests
import json

load_dotenv()


# -- Part1 agent's tool to search web
def search_web(query: str) -> str:
    """
    Use the Serper.dev API to perform a web search
    Returns a formatted string of search results.
    """
    pass


def browse_website(url: str) -> str:
    """
        scrapes the text content of a given URL.
        returns the cleaned text content or an error message if it fails
    """
    pass


# Part2: The Agent's "Brain" (Ollama interaction)
def call_gemma_ollama(prompt: str, output_format: str = "json") -> str: 
    """
    A helper function to call the local ollama API and get a response
    """
    pass

# Part3: The agentic Chain logical with memory and Robustnes
def run_agent(goal: str, history: list) -> str: 
    """
    Runs the main logic of the agent, now with conversation history and robust multi-site broswing.
    Returns the final summary to be added to the history.
    """
    pass

def main():
    pass


if __file__ == "__main__":
    main()



