from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
load_dotenv(r"C:\Users\SOHAIL\phase7-llm-apis\.env")
llm=ChatGroq(
    model='llama-3.3-70b-versatile',
    groq_api_key=os.getenv('GROQ_API_KEY')
    

)
parser = StrOutputParser()
coding_prompt=PromptTemplate(
    input_variables=['question'],
    template='you are an expert programmer answer this coding question:\n{question}'
)
science_prompt=PromptTemplate(
    input_variables=['question'],
    template='You are a science expert. Answer this science question:\n{question}'

)
general_prompt = PromptTemplate(
    input_variables=['question'],
    template='Answer this question:\n{question}'
)
coding_chain = coding_prompt | llm | parser

science_chain = science_prompt | llm | parser

general_chain = general_prompt | llm | parser
from langchain_core.runnables import RunnableLambda
def route(input):
    topic=input['topic']
    if 'code' in topic.lower() or 'programming' in topic.lower():
        return coding_chain
    elif 'science' in topic.lower() or 'biology' in topic.lower():
        return science_chain
    else:
        return general_chain
final_chain=RunnableLambda(route)
result=final_chain.invoke({"topic":"coding","question":"what is a for loop"})
print(result)

result = final_chain.invoke({"topic": "science", "question": "What is photosynthesis?"})
print(result)

# Test 3
result = final_chain.invoke({"topic": "love", "question": "What is love?"})
print(result)

