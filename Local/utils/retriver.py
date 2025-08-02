from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are a personal AI assistant. Create 2-3 diverse rephrasings of the user's question:
Original: {question}"""
)


def get_multiquery_retriever(vectordb, llm):
    return MultiQueryRetriever.from_llm(
        retriever=vectordb.as_retriever(search_kwargs={"k": 6}),
        llm=llm,
        prompt=QUERY_PROMPT
    )
