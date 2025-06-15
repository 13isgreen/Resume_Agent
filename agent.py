from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model="llama3.2")

template = """
You are a Lab Technician responsible for monitoring the lab equipment 

Here is the status of the printer: {status}

Here is the question to answer: {question}
"""

# Instantiate this template above as a promt
promt = ChatPromptTemplate.from_template(template)
# Pass variables into promt then promt into model
chain = promt | model                

result = chain.invoke({"status":[],"question":"How much longer on the print?"})
print(result)