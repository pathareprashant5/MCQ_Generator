import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from mcqgenerator.utils import read_file,get_table_data
from mcqgenerator.logger import logging

# Importing necessary packages from langchain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

# Load environment variables form the .env file
load_dotenv()

# Access the enviroment varaibles just like you would with os.environ
key=os.getenv("OPENAI_API_KEY")


llm = ChatOpenAI(openai_api_key=key,model_name="gpt-3.5-turbo", temperature = 0.7)

template="""
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz of {number} multiple choice questions for {subject} students in {tone} tone.
Make sure the questions are not repeated and check all the questions to be confirming the text as well.
Make sure to format your response like RESPONS_JSON below and use it as a guide.\
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}
"""

qgp=PromptTemplate(      #qgp-quiz generation prompt
    input_variables=["text ","number","subject","tone","response_json"],
    template=template
)

quiz_chain=LLMChain(llm=llm,prompt=qgp,output_key="quiz",verbose=True)


template2="""
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use max 50 words for complexity.
If the quiz is not per the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the students ability.
Quiz_MCQs:
{quiz}

Check from an expert English writer of the above quiz:
"""

qep=PromptTemplate(      #qep - quiz evaluation template
    input_variable=["subject","quiz"], template=template2
)

review_chain=LLMChain(llm=llm,prompt=qep,output_key="review",verbose=True)

generate_evaluate_chain=SequentialChain(chains=[quiz_chain, review_chain], input_variables=["text", "number", "subject", "tone", "response_json"],
                                        output_variables=["quiz", "review"], verbose=True,)


