
## data flow 

The basic data flow for this agent:

User Input question ---> Ollama (summary the client input for google search) ---> use the ollama output to search (serpApi) ---> parse the search response ---> Ollama(summary the search response for client and format the result for client query) ---> send the response to client via email




