
from langchain_core.prompts import PromptTemplate
from src.retriever import retrieve_docs
from src.config import MODEL_PROVIDER, OPENAI_API_KEY, GEMINI_API_KEY

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful travel assistant.
Use the following travel guide context to answer the question.
If the answer is not found, say you don't know — don't make it up.

Context:
{context}

Question:
{question}

Answer:"""
)

# Initialize LLM based on MODEL_PROVIDER
if MODEL_PROVIDER == "huggingface":
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    from langchain_huggingface import HuggingFacePipeline

    model_id = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False,
    )

    llm = HuggingFacePipeline(pipeline=pipe)

elif MODEL_PROVIDER == "openai":
    from langchain_openai import ChatOpenAI

    if not OPENAI_API_KEY:
        raise ValueError("Missing OPENAI_API_KEY in .env")

    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4o-mini",
        temperature=0.2,
    )

elif MODEL_PROVIDER == "gemini":
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not GEMINI_API_KEY:
        raise ValueError("Missing GEMINI_API_KEY in .env")

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=GEMINI_API_KEY,
        temperature=0.2,
        max_retries=3,
    )

else:
    raise ValueError(
        f"Unknown MODEL_PROVIDER: {MODEL_PROVIDER}. "
        "Choose 'huggingface', 'openai', or 'gemini'."
    )


# Create RAG chain
qa_chain = prompt_template | llm


def generate_answer(query: str) -> str:
    docs = retrieve_docs(query)
    context = "\n\n".join(docs)

    result = qa_chain.invoke({
        "context": context,
        "question": query
    })

    content = result.content if hasattr(result, "content") else result

    # Chuyển nội dung phản hồi thành chuỗi văn bản
    if isinstance(content, list):
        return "\n".join(
            part["text"]
            for part in content
            if isinstance(part, dict)
            and part.get("type") == "text"
            and part.get("text")
        )

    return content if isinstance(content, str) else str(content)