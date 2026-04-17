from typing import cast

from dotenv import load_dotenv

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from ingestion import retriever

load_dotenv()


def test_retrieval_grader_document_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    # get highest document
    doc_txt = docs[0].page_content

    res: GradeDocuments = cast(
        GradeDocuments,
        retrieval_grader.invoke({"question": question, "document": doc_txt}),
    )

    assert res.binary_score == "yes"


def test_retrieval_grader_document_no() -> None:
    question = "agent memory"
    new_question = "how to make pizza"

    docs = retriever.invoke(question)

    # get highest document
    doc_txt = docs[-1].page_content

    res: GradeDocuments = cast(
        GradeDocuments,
        retrieval_grader.invoke({"question": new_question, "document": doc_txt}),
    )

    assert res.binary_score == "no"
