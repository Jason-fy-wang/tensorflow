from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import json
import os

load_dotenv()


# -- Part1 agent's tool to search web
def search_web(query: str) -> str:
    """
    Use the Serper.dev API to perform a web search
    Returns a formatted string of search results.
    """
    print(f"----Tool: Searching web for: {query}----")
    SERPER_API_KEY=os.getenv("SERP_KEY")
    if not SERPER_API_KEY:
        print(f"please set SERPER_API_KEY")
        return f"Error: SERPER_API_KEY is not set. Cannot perform web search."

    payload = json.dumps({"q": query,"api_key":SERPER_API_KEY})
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.get("https://serpapi.com/search", headers=headers,data=payload)
        response.raise_for_status()
        results = response.json()
        #print(f"results: {results}")

        if not results.get("organic_results"):
            return "No results found."
        output = "Search Results:\n"
        for idx, result in enumerate(results["organic_results"][:5], start=1):
            title = result.get("title", "No Title")
            snippet = result.get("snippet", "No Snippet")
            link = result.get("link", "No Link")
            output += f"{idx}. {title}\n{snippet}\nLink: {link}\n\n"
        return output.strip()

    except requests.exceptions.RequestException as e:
        return f"Error: during web search: {e}"



def browse_website(url: str) -> str:
    """
        scrapes the text content of a given URL.
        returns the cleaned text content or an error message if it fails
    """
    print(f"----Tool: Browsing website at {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        # Remove scripts and styles
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()
        text = soup.get_text(separator='\n')
        # Clean up the text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)

        if not text:
            return "No text content found on the webpage."

        return text[:8000]  # Limit to first 8000 characters
    except requests.exceptions.RequestException as e:
        return f"Error: Unable to browse website: {e}"


# Part2: The Agent's "Brain" (Ollama interaction)
def call_gemma_ollama(prompt: str, output_format: str = "json") -> str: 
    """
    A helper function to call the local ollama API and get a response
    """
    load_dotenv()
    HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
    url = f"{HOST}/api/generate"
    print(f"Prompt to Ollama: {url}")
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
    }
    if output_format == "json":
        payload["output_format"] = "json"
    try:
        print(f"requesting {url}, model: {MODEL}")
        response = requests.post(url=url, data=json.dumps(payload), timeout=300)
        response.raise_for_status()
        resp_text = json.loads(response.text)
        #print(f"response text: {resp_text}")
        return resp_text["response"]
    except requests.Timeout as e:
        return f"Error: time out exception: {e}"
    except Exception as e:
        return f"Error: other exception: {e}"

# Part3: The agentic Chain logical with memory and Robustnes
def run_agent(goal: str, history: list) -> str: 
    """
    Runs the main logic of the agent, now with conversation history and robust multi-site broswing.
    Returns the final summary to be added to the history.
    """
    # step -1: Extract email address from the goal if it exists
    prompt_extract_email = f"""
    You are an expert at extracting email address from text.
    Analyse the following user requests and extract the email address if one is present.
    Return the email address only. If no email address is present, return with the word "none".

    User Request: {goal}
    """
    recipient_email = call_gemma_ollama(prompt_extract_email, output_format="text").strip()
    print(f"Extracted email: {recipient_email}")
    if "@" not in recipient_email:
        recipient_email = "none"
    
    print(f"\n: Goal: {goal}\n")
    format_history = "\n".join(history)

    # 1. Decide what to search
    prompt_decide_search = f"""
    You are a helpful AI agent. Your task is to understand a User's request and generate a concise , effective search query to find the necessary information to fulfill the request.

    Consider the following conversion history:
    ---
    {format_history}
    ---
    User's latest request: "{goal}"

    Based on the request, what is the best, simple search query for Google? The query should be 3 to 5 words.
    Response with ONLY the search query itself.
    """
    search_query = call_gemma_ollama(prompt_decide_search, output_format="text").strip()
    # 2. search the web
    search_results = search_web(search_query)
    print(f"\nSearch Results:\n{search_results}\n")

    # 3. Choose which sites to browse
    prompt2 = f"""
    You are a smart web navigator. Your task is to analyze Google search results and select the most promising URLs to find the answer to the User's request.
    Avoid generic homepages(like yelp.com or google.com) and perfer specific artificles, lists and maps.

    User's goal: "{goal}"

    Search Results:
    ---
    {search_results}
    ---

    Based on the User's goal and search results, which are the top 2-3 most promising and specific URLs to browse for details ?
    Response with ONLY the URLs , one per line.
    """

    browse_urls_str = call_gemma_ollama(prompt2, output_format="text").strip()
    browse_urls = [url.strip() for url in browse_urls_str.split("\n") if url.strip().startswith("http")]
    print(f"\nChosen URLs to browse:\n{browse_urls}\n")

    if not browse_urls:
        print(f"--Could not identify promising URLs to browse. Trying to summarize from search results. ---")

        # if no URLs found, summarize from search results.
        prompt_summarize_snippets = f"""
        You are a helpful AI agent. The web browser is not working.  but you have search result snippets.
        User's goal: "{goal}"
        Search Results:
        ---
        {search_results}
        ---
        Please provide a summary based "only" on the search result snippets. Do not suggest broswing URLs.
        """
        final_summary = call_gemma_ollama(prompt_summarize_snippets, output_format="text").strip()
        print(f"\nFinal Summary from snippets:\n{final_summary}\n")
        return final_summary

     # 4. Browse the chosen URLs and collect content
    all_website_texts = []
    for url in browse_urls:
        website_text = browse_website(url)
        if not website_text.startswith("Error:"):
            all_website_texts.append(f"URL: {url}\nContent:\n{website_text}\n")
        else:
            print(f"Skipping {url} due to error: {website_text}")

    if not all_website_texts:
        return "I tried to browse several websites but was blocked or could not find any information. Please try again later."

    aggregated_text = "\n".join(all_website_texts)

    # 5. Summarize everything for user
    prompt3 = f"""
    You are a meticulous and trustworthy AI agent. Your primary goal is to provide a clear,concise, and above all, ACCURATE answer to the User's request by synthesizing information from multiple sources.

    User's latest goal: "{goal}"

    You have gathered the following text from one or more websites:
    ---
    {aggregated_text}
    ---

    Fact-Check and Synthesize:
    Based on the information above, provide a comprehensive summary that directly answers the User's request.
    Before including any business or item in your summary, you MUST verify that it meets ALL the specific criteria from the User's request.(e.g., hours of operation,localtion, specific features).
    If you cannot find explicit confirmation that a business meets a criterion, DO NOT include it in the summary.
    It is better to provide fewer , accurate results than more, inaccurate ones.

    Format your response clearly for the user. If listing places, use bullet points.
    """
    final_summary = call_gemma_ollama(prompt3, output_format="text").strip()
    print(f"\nFinal Summary:\n{final_summary}\n")

    # 6. Decide if an email should be send and generate its content
    prompt4 = f"""
    You are a highly capable assistant, responsible for drafting clear and detailed emails based on a resarch summary.
    User's original goal: "{goal}"
    Here is the final summary of the research, which has been fact-checked to meet the user's criteria:
    ---
    {final_summary}
    ---

    Here is the reminder of the raw text gathered from the websites, which you can use to find details like reservation links:
    ---
    {aggregated_text}
    ---

    Your task is to decide if an email is appropriate to send to the user with this information. If it is, you must draft the email. 
    - If the summary contains useful, actionable information (like a list of places, contact info, etc.) then an email should be sent.
    - If the summary is short, conversational, or indicates no results were fond, an email is not needed.

    Instructions for the email draft:
    1. Create a clear subject line that summarizes the content
    2. The email body should be a list of the places mentioned in the final summary.
    3. For each place, provide a brief summary of what it offers and, if you can find one in the raw text, the direct link for reservations.
    4. Ensure that ONLY information that strictly matches the user's request. (e.g., open on a specific day) is included.

    Respond with json format:
    If sending , the JSON should be {{"send_email": true, "Subject": "Your requested information", "Body": "....."}}
    If not sending, the JSON should be {{"send_email": false}}

    Example for sending:
    {
        {"send_email": True, 
         "Subject": "Your requested list of Sushi restaurants in Seattle.", 
         "Body": "Hello, \n\nHere are the sushi restaurants that match your criteria: \n\n **Shiro Sushi:** A classic spot known for its traditional sushi. Reservations: [https://www.shiros.com/reservations](https://www.shiros.com/reservations) \n\n **Sushi Zen:** Offers a modern take on sushi with a focus on fresh, local ingredients. Reservations: [https://www.sushizen.com/book](https://www.sushizen.com/book) \n\nBest regards,\nYour AI Assistant"
         }
    }
    """

    email_decision_json = call_gemma_ollama(prompt4, output_format="json")
    print(f"\nEmail Decision JSON:\n{email_decision_json}\n")
    try:
        email_decision = json.loads(email_decision_json)
        if email_decision.get("send_email"):
            email_subject = email_decision.get("Subject", "No Subject")
            email_body = email_decision.get("Body", "No Body")
            if all([email_subject,email_body]):
                # Here you would integrate with an email sending service
                print(f"Preparing to send email to {recipient_email} with subject: {email_subject}")
                # send_email(recipient_email, email_subject, email_body)
                recipient = "none"
                if recipient_email != "none":
                    confirm = input(f"Should i send email to {recipient_email}(y/n)")
                    if confirm.lower() == "y":
                        recipient = recipient_email
                else:
                    confirm = input(f"Would you like me to email this summary to you? (y/n)").lower()
                    if confirm == "y":
                        recipient = input("Please provide your email address: ").strip()
                if recipient != "none":
                    result = f"send email success"
                    print(f"Email sent to {recipient}:\nSubject: {email_subject}\nBody:\n{email_body}\n")            
                else:
                    print(f"---Okay, i will not send the email. ---")
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Error parsing email decision JSON: {e}")
    return final_summary

# print(search_web("why sky is blue? "))

# browse_website("https://spaceplace.nasa.gov/blue-sky/en/")


run_agent("Find me 3 sushi restaurants in Seattle that are open on Sunday and accept reservations online.", [])
#print(call_gemma_ollama("What is the capital of France?", output_format="text"))


# def main():
#     print("searching")
#     search_web("why sky is blue? ")


# if __file__ == "__main__":
#     main()



