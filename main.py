import os  # helps to access environment variables and perform operating system related tasks

from dotenv import \
    load_dotenv  # used to load environment variables from a .env file
from langchain_core.prompts import \
    PromptTemplate  # used to create prompt templates for language models
from langchain_google_genai import \
    ChatGoogleGenerativeAI  # used to interact with Google's generative AI models
from langchain_ollama import \
    ChatOllama  # used to interact with Ollama's chat models

load_dotenv()  # loads the environment variables from the .env file into the program's environment

def main():  # function that serves as the entry point of the program

    # Text about a person later can be given as an input
    info = """  
        Avul Pakir Jainulabdeen Abdul Kalam; 15 October 1931 – 27 July 2015) was an Indian aerospace engineer and science administrator. He served as president of India from 2002 to 2007.
        Born and raised in a Muslim family in Rameswaram, Tamil Nadu, Kalam studied physics and aerospace engineering. He spent the next four decades as a scientist and science administrator, mainly at the Defence Research and Development Organisation (DRDO) and Indian Space Research Organisation (ISRO) and was intimately involved in India's civilian space programme and military missile development efforts. He is popularly known as the "Missile Man of India" for his work on the development of ballistic missile and launch vehicle technology. He also played a pivotal organisational, technical, and political role in Pokhran-II nuclear tests in 1998, India's second such test after the first test in 1974.
        Kalam was elected as the president of India in 2002 with the support of both the ruling Bharatiya Janata Party and the then-opposition Indian National Congress. He was widely referred to as the "People's President". He engaged in teaching, writing and public service after his presidency. He was a recipient of several awards, including the Bharat Ratna, India's highest civilian honour.
        While delivering a lecture at IIM Shillong, Kalam collapsed and died from an apparent cardiac arrest on 27 July 2015, aged 83. Thousands attended the funeral ceremony held in his hometown of Rameswaram, where he was buried with full state honours. A memorial was inaugurated near his home town in 2017.
    """
    # formatting the prompt using the parameters
    System_prompt_Template = """
    given the information {info} about a person I want to create:
    1. A Short Summary
    2.Two interesting facts about the person
    """
    # passing paramters to the system prompt template
    system_prompt = PromptTemplate(
        input_variables=["info"],
        template=System_prompt_Template,
    )
    # llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=os.environ.get("GOOGLE_API_KEY"))

    llm = ChatOllama(model="gemma3:270m", temperature=0)
    chain = (
        system_prompt | llm
    )  # chaining the system prompt and the language model together

    response = chain.invoke({"info": info})
    print(response.content)


if (
    __name__ == "__main__"
):  # checks if the script is being run directly (not imported as a module) and calls the main function
    main()
