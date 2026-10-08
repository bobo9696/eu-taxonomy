"""Evaluate a RAG pipeline with ragas on a golden set of answerable and unanswerable questions."""

import asyncio

import numpy as np
from pydantic import BaseModel, Field
from ragas.embeddings import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

from rag.query.chain import PromptType, RagType, ask_question
from .config import CHAT_MODEL, EMBEDDING_MODEL, get_async_client, get_llm
from rag.index.methods.basic import FAQ_PATH, FAQChunk, parse_taxonomy_faqs

# Answerable questions paraphrase an FAQ entry, whose official answer is the ragas reference.
# Buried-answer questions test whether retrieval works when the answer is in the body, not the title.
# Unanswerable questions (faq_question=None) are absent from the FAQ: the RAG must decline to answer them.
TEST_SET = [
    # Climate Delegated Act – paraphrased
    {
        "question": "Are the technical screening criteria under the Climate Delegated Act going to be tightened and reviewed periodically?",
        "faq_question": "Will the technical screening criteria set out in the Climate Delegated Act be made stricter and updated over time?",
    },
    # Climate Delegated Act – buried answer
    {
        "question": "How frequently should forest management plans be renewed to remain compliant with the Taxonomy?",
        "faq_question": "What is the meaning of a “continuously updated” forest management plan? Does this mean renewed after the end of its validity, in the occurrence of significant events, or yearly? Or is it about having evidence of the work done in the forest?",
    },
    # Taxonomy-Alignment Reporting – paraphrased
    {
        "question": "If a parent company already reports consolidated Taxonomy figures, can its NFRD subsidiary skip its own disclosures?",
        "faq_question": "Can a subsidiary falling within scope of the NFRD/CSRD be exempted from making Taxonomy related disclosures if its parent undertaking discloses Taxonomy-related information at consolidated level where required or on a voluntary basis?",
    },
    # Taxonomy-Alignment Reporting – buried answer
    {
        "question": "When does the 18-month completion window for individual CapEx measures start?",
        "faq_question": "What is the beginning of the 18 month-period for the completion of individual measures referred to in point (c) of Section 1.1.2.2 of Annex I?",
    },
    # Taxonomy-Eligibility reporting (part 2) – paraphrased
    {
        "question": "Can companies publish extra voluntary information alongside their mandatory Taxonomy-eligibility figures?",
        "faq_question": "Can undertakings report voluntary information in the context of Taxonomy eligibility reporting?",
    },
    # Taxonomy-Eligibility reporting (part 2) – buried answer
    {
        "question": "How should a bank that also holds a MiFID investment-firm licence report its Taxonomy-eligible activities?",
        "faq_question": "How should a credit institution with a Markets in Financial Instruments Directive (MifiD) investment firm license report its Taxonomy-eligible economic activities?",
    },
    # Taxonomy-Eligibility reporting (part 1) – paraphrased
    {
        "question": "What should a company disclose if none of its economic activities are Taxonomy-eligible?",
        "faq_question": "What should undertakings with no Taxonomy-eligible economic activities report?",
    },
    # Taxonomy-Eligibility reporting (part 1) – buried answer
    {
        "question": "Are derivative contracts part of the mandatory Taxonomy-eligibility reporting for financial firms?",
        "faq_question": "Are derivative instruments in scope of Taxonomy-eligibility reporting?",
    },
    # Complementary Climate Delegated Act – paraphrased
    {
        "question": "Which new activities and technical criteria are introduced by the Complementary Climate Delegated Act?",
        "faq_question": "What activities and accompanying criteria does the Complementary Climate Delegated Act introduce?",
    },
    # Complementary Climate Delegated Act – buried answer
    {
        "question": "Why did the EU decide to label certain gas-related activities as substantially contributing to climate objectives?",
        "faq_question": "What is the rationale for defining certain gas-related activities as “substantially contributing to” the EU climate and environmental objectives?",
    },
    # Disclosures Delegated Act - General – paraphrased
    {
        "question": "What advantages do companies gain from the disclosures delegated act?",
        "faq_question": "What are the main benefits for financial and non-financial companies associated with this disclosures delegated act?",
    },
    # Disclosures Delegated Act - General – buried answer
    {
        "question": "How should financial companies treat their exposures to government entities for Taxonomy reporting?",
        "faq_question": "How does this disclosures delegated act treat exposures to government entities in the reporting of financial companies?",
    },
    # EU Taxonomy - General – paraphrased
    {
        "question": "How does the Taxonomy determine whether an economic activity counts as green?",
        "faq_question": "How does the EU Taxonomy define green economic activities?",
    },
    # EU Taxonomy - General – buried answer
    {
        "question": "If a company's line of business is not listed in the EU Taxonomy, what does that imply?",
        "faq_question": "My company’s activity is not covered in the EU Taxonomy. What does that mean for my company?",
    },
    # Near-miss unanswerable (topic is covered, but the specific detail is not in the FAQ)
    {
        "question": "What fines can national regulators impose on a company that knowingly misreports its Taxonomy KPIs?",
        "faq_question": None,
    },
    {
        "question": "What is the exact CO2 emission limit in grammes per kWh for electricity generation from geothermal energy under the Climate Delegated Act?",
        "faq_question": None,
    },
    {
        "question": "What was the total amount of green finance raised in the EU that was explicitly aligned with the Taxonomy in 2022?",
        "faq_question": None,
    },
    {
        "question": "By what exact deadline must EU Member States transpose the Taxonomy Regulation into national law?",
        "faq_question": None,
    },
    # Off-topic.
    {
        "question": "Who won the 2018 FIFA World Cup?",
        "faq_question": None,
    },
    {
        "question": "What is the recipe for a traditional ratatouille?",
        "faq_question": None,
    },
]

METRICS = (
    "retrieval_hit",
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "answer_rate",
    "refusal_rate",
)

REFUSAL_PROMPT = """Decide whether the response below declines to answer because the assistant lacks the information (e.g. "I don't know", "the provided FAQs do not cover this").
It still counts as declining if related context is added alongside that statement.
A substantive answer is not a refusal, even when the answer is negative (e.g. "No", "there is no universal method", "the act does not classify technologies").

Question: {question}

Response: {response}"""


class RefusalVerdict(BaseModel):
    refused: bool = Field(description="True only if the response declines to answer for lack of information")


def _faq_entries(faq_question: str, chunks: list[FAQChunk]) -> list[FAQChunk]:
    matches = [chunk for chunk in chunks if chunk.question == faq_question]
    if not matches:
        raise ValueError(f"Expected at least one FAQ entry titled {faq_question!r}, found 0")
    return matches


async def evaluate_rag(rag_type: RagType, prompt_type: PromptType, use_rerank: bool = False) -> dict:
    chunks = parse_taxonomy_faqs(FAQ_PATH)
    # Resolve every reference up front so a broken test set fails before any LLM call.
    entries = {t["faq_question"]: _faq_entries(t["faq_question"], chunks) for t in TEST_SET if t["faq_question"]}
    refusal_judge = get_llm().with_structured_output(RefusalVerdict)
    client = get_async_client()
    llm = llm_factory(CHAT_MODEL, client=client, max_tokens=4096)
    faithfulness = Faithfulness(llm=llm)
    context_precision = ContextPrecision(llm=llm)
    context_recall = ContextRecall(llm=llm)
    answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embedding_factory("openai", model=EMBEDDING_MODEL, client=client))

    async def evaluate_sample(test: dict) -> dict:
        question = test["question"]
        answerable = test["faq_question"] is not None
        response = await asyncio.to_thread(ask_question, question, prompt_type, rag_type, use_rerank)
        answer, contexts = response["answer"], response["contexts"]
        refused = (await refusal_judge.ainvoke(REFUSAL_PROMPT.format(question=question, response=answer))).refused

        # Metrics that do not apply to this kind of question stay None and are left out of the averages.
        scores: dict[str, float | None] = dict.fromkeys(METRICS)
        if not answerable:
            scores["refusal_rate"] = float(refused)
            return {"question": question, "answerable": False, "answer": answer, "sources": response["sources"], "scores": scores}

        entry_list = entries[test["faq_question"]]
        target_ids = {entry.id for entry in entry_list}
        scores |= {
            # Penalises over-cautious pipelines that decline questions the FAQ does answer.
            "answer_rate": float(not refused),
            # Deterministic retrieval check: was the target FAQ entry retrieved at all?
            "retrieval_hit": float(any(source["chunk_id"] in target_ids for source in response["sources"])),
            "answer_relevancy": (await answer_relevancy.ascore(user_input=question, response=answer)).value,
        }
        # All duplicate entries share the same answer text, so the first is sufficient for ragas.
        entry = entry_list[0]
        if contexts:
            faith, precision, recall = await asyncio.gather(
                faithfulness.ascore(user_input=question, response=answer, retrieved_contexts=contexts),
                context_precision.ascore(user_input=question, reference=entry.answer, retrieved_contexts=contexts),
                context_recall.ascore(user_input=question, reference=entry.answer, retrieved_contexts=contexts),
            )
            scores |= {"faithfulness": faith.value, "context_precision": precision.value, "context_recall": recall.value}
        else:
            # Nothing retrieved: retrieval failed outright, and faithfulness to an empty context is undefined.
            scores |= {"context_precision": 0.0, "context_recall": 0.0}

        return {"question": question, "answerable": True, "answer": answer, "sources": response["sources"], "scores": scores}

    samples = await asyncio.gather(*(evaluate_sample(test) for test in TEST_SET))

    metrics = {}
    for name in METRICS:
        values = [s["scores"][name] for s in samples if s["scores"][name] is not None]
        metrics[name] = round(float(np.mean(values)), 3) if values else None

    return {"rag_type": rag_type, "prompt_type": prompt_type, "use_rerank": use_rerank, "samples": samples, "metrics": metrics}


async def score_single_response(question: str, answer: str, contexts: list[str]) -> dict:
    """Score a single response with ragas faithfulness + answer relevancy (no reference needed)."""
    client = get_async_client()
    llm = llm_factory(CHAT_MODEL, client=client, max_tokens=4096)
    faithfulness = Faithfulness(llm=llm)
    answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embedding_factory("openai", model=EMBEDDING_MODEL, client=client))

    faith = await faithfulness.ascore(user_input=question, response=answer, retrieved_contexts=contexts)
    relevancy = await answer_relevancy.ascore(user_input=question, response=answer)

    return {
        "faithfulness": round(float(faith.value), 3) if faith.value is not None else None,
        "answer_relevancy": round(float(relevancy.value), 3) if relevancy.value is not None else None,
    }
