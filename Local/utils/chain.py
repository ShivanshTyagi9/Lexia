from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

template = """Your task is to respond to the user query according to the context retrieved from the vector database.
    Instructions:
    1. Be concise and to the point.
    2. Use the context provided to answer the question.
    3. Don't use phrases like "According to the context" or "Based on the information provided", like that.

    Instructions for the valid output JSON format:
    {{
        "response": "Your answer here"
    }}

Use the following context only:
{context}

Question: {question}
"""

def build_rag_chain(retriever, llm):
    prompt = ChatPromptTemplate.from_template(template)
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# def build_rag_chain_with_context(retriever, llm):
#     prompt = ChatPromptTemplate.from_template(template)

#     def format_contexts(docs):
#         return "\n\n".join([doc.page_content for doc in docs])

#     def rag_pipeline(question):
#         retrieved_docs = retriever.invoke(question)
#         formatted_context = format_contexts(retrieved_docs)
#         prompt_input = prompt.invoke({"context": formatted_context, "question": question})
#         response = llm.invoke(prompt_input)
#         answer = response.content if hasattr(response, "content") else str(response)
#         return {
#             "answer": answer,
#             "contexts": [doc.page_content for doc in retrieved_docs]
#         }

#     return rag_pipeline