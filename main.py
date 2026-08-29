import os
import streamlit as st
from dotenv import load_dotenv

from pypdf import PdfReader
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
#from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
#from langchain.agents import create_tool_calling_agent,AgentExecutor

load_dotenv()
st.set_page_config(
    page_title="Research Paper Analyzer",
    page_icon="📄",
    layout="wide"
)


class ResearchResponse(BaseModel):
    title: str = Field(
        description="Title of the paper"
    )

    authors: list[str] = Field(
        description=" Names of author"
    )
    publication_date: str | None = Field(
        default=None,
        description="Publication date or year"
    )
    venue: str | None = Field(
        default=None,
        description="Publication venue (journal or conference name)"
    )
    doi: str | None = Field(
        default=None,
        description="DOI of the paper"
    )

    # Research Overview

    research_problem: str = Field(
        description="The main problem or challenge the paper wants to address"
    )
    research_questions: str | None = Field(
        default=None,
        description="Main Objective of the paper"
    )
    summary: str = Field(
        description=" Summary of the paper with results and methodsology"
    )
    methodology: str = Field(
        description="Detailed explanation of the methods, models, algorithms, experiments, procedures, or theoretical approaches used in the research."
    )

    datasets: list[str] = Field(
        description="Names and descriptions of datasets used in the research. Return an empty list if no dataset is used or mentioned."
    )

    # ─── Results & Contributions ───

    key_findings: list[str] = Field(
        description="The most important findings, experimental results, observations, or conclusions supported by the paper."
    )

    contributions: list[str] = Field(
        description="The main scientific, technical, theoretical, or practical contributions claimed or demonstrated by the paper."
    )

    # ─── Critical Analysis ───

    limitations: list[str] = Field(
        description="Limitations explicitly acknowledged by the authors. Do not invent limitations."
    )

    strengths: list[str] = Field(
        description="Important strengths of the research, based on the evidence and methodology presented in the paper."
    )

    weaknesses: list[str] = Field(
        description="Reasonable weaknesses identified through critical analysis of the methodology, experiments, datasets, or conclusions."
    )

    # ─── Research Gaps ───

    explicit_research_gaps: list[str] = Field(
        description="Research gaps, unanswered questions, or limitations explicitly identified by the authors."
    )

    inferred_research_gaps: list[str] = Field(
        description="Additional research gaps identified through critical analysis, clearly distinct from author-stated gaps. Return an empty list if none are inferred."
    )

    # ─── Future Research ───

    future_work: list[str] = Field(
        description="Future research directions explicitly suggested by the authors or strongly supported by the identified limitations and research gaps."
    )

    # ─── Final Takeaway ───

    conclusion: str = Field(
        description="what does the paper say"
    )


# Setting Up an LLM ( either claude or gpt)
llm = ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# calling a llm response
# response = llm.invoke("Whats  the meaning of life?")
# print(response)


parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [

        # ----------------------------------------------------
        # SYSTEM PROMPT
        # ----------------------------------------------------

        (
            "system",
            """
You are an expert research paper analysis assistant.

Your task is to analyze the complete research paper provided
by the user and produce an accurate structured research analysis.

IMPORTANT RULES:

1. Base your analysis ONLY on the provided research paper.

2. Do NOT invent, assume, or hallucinate information.

3. Extract:
   - Title
   - Authors
   - Publication date
   - Publication venue
   - DOI
   - Research problem
   - Research question/objective
   - Summary
   - Methodology
   - Datasets
   - Key findings
   - Contributions
   - Limitations
   - Strengths
   - Weaknesses
   - Research gaps
   - Future work
   - Conclusion

4. Provide a comprehensive but understandable summary.

5. Preserve important technical information such as:
   - Model names
   - Algorithms
   - Datasets
   - Evaluation metrics
   - Numerical results
   - Experimental settings

6. LIMITATIONS:
   Only identify limitations explicitly mentioned by the authors.

7. EXPLICIT RESEARCH GAPS:
   Only include research gaps or open problems explicitly
   identified by the authors.

8. INFERRED RESEARCH GAPS:
   You may identify additional research gaps through critical
   analysis, but clearly distinguish these from author-stated gaps.

9. Never present an inferred research gap as something
   explicitly stated by the authors.

10. STRENGTHS:
    Identify strengths based on evidence in the paper.

11. WEAKNESSES:
    Identify reasonable weaknesses based on critical analysis
    of the methodology, experiments, datasets, or conclusions.

12. If information is unavailable:
    - Use null for optional fields.
    - Use an empty list for list fields.

13. Do not invent:
    - Authors
    - Publication dates
    - DOI
    - Datasets
    - Results
    - Citations
    - Metrics

14. Return ONLY the requested structured format.

{format_instructions}
            """
        ),

        # ----------------------------------------------------
        # PAPER TEXT
        # ----------------------------------------------------

        (
            "human",
            """
Analyze the following research paper.

================ PAPER START ================

{paper_text}

================= PAPER END =================

Provide the complete structured research analysis.
            """
        ),
    ]
).partial(
    format_instructions=parser.get_format_instructions()
)
# creating agent

chain = prompt | llm | parser

st.title("📄 Research Paper Analyzer")

st.write(
    "Upload a research paper PDF and get a structured "
    "analysis including summary, methodology, findings, "
    "research gaps, limitations, and future work."
)


# ============================================================
# PDF UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your research paper",
    type=["pdf"]
)


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file is not None:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    if st.button("🔍 Analyze Research Paper"):

        with st.spinner(
            "Extracting and analyzing the research paper..."
        ):

            try:

                # --------------------------------------------
                # Read PDF
                # --------------------------------------------

                reader = PdfReader(
                    uploaded_file
                )

                paper_text = ""

                for page in reader.pages:

                    text = page.extract_text()

                    if text:
                        paper_text += text + "\n"

                # --------------------------------------------
                # Check extraction
                # --------------------------------------------

                if not paper_text.strip():

                    st.error(
                        "Could not extract text from this PDF. "
                        "The PDF may contain scanned images instead "
                        "of selectable text."
                    )

                    st.stop()

                # --------------------------------------------
                # Analyze paper
                # --------------------------------------------

                result = chain.invoke(
                    {
                        "paper_text": paper_text
                    }
                )

                # --------------------------------------------
                # DISPLAY RESULTS
                # --------------------------------------------

                st.header("📋 Research Paper Analysis")

                # ------------------------------------------------
                # Paper Information
                # ------------------------------------------------

                st.subheader("📄 Paper Information")

                st.write(
                    f"**Title:** {result.title}"
                )

                st.write(
                    f"**Authors:** {', '.join(result.authors)}"
                )

                st.write(
                    f"**Publication Date:** "
                    f"{result.publication_date or 'Not available'}"
                )

                st.write(
                    f"**Venue:** "
                    f"{result.venue or 'Not available'}"
                )

                st.write(
                    f"**DOI:** "
                    f"{result.doi or 'Not available'}"
                )

                # ------------------------------------------------
                # Research Problem
                # ------------------------------------------------

                st.subheader("🎯 Research Problem")

                st.write(
                    result.research_problem
                )

                # ------------------------------------------------
                # Research Question
                # ------------------------------------------------

                st.subheader("❓ Research Question")

                st.write(
                    result.research_questions
                    or "Not explicitly mentioned."
                )

                # ------------------------------------------------
                # Summary
                # ------------------------------------------------

                st.subheader("📝 Summary")

                st.write(
                    result.summary
                )

                # ------------------------------------------------
                # Methodology
                # ------------------------------------------------

                st.subheader("🔬 Methodology")

                st.write(
                    result.methodology
                )

                # ------------------------------------------------
                # Datasets
                # ------------------------------------------------

                st.subheader("📊 Datasets")

                if result.datasets:

                    for dataset in result.datasets:

                        st.write(
                            f"- {dataset}"
                        )

                else:

                    st.write(
                        "No datasets mentioned."
                    )

                # ------------------------------------------------
                # Key Findings
                # ------------------------------------------------

                st.subheader("💡 Key Findings")

                for finding in result.key_findings:

                    st.write(
                        f"- {finding}"
                    )

                # ------------------------------------------------
                # Contributions
                # ------------------------------------------------

                st.subheader("🏆 Contributions")

                for contribution in result.contributions:

                    st.write(
                        f"- {contribution}"
                    )

                # ------------------------------------------------
                # Limitations
                # ------------------------------------------------

                st.subheader("⚠️ Limitations")

                for limitation in result.limitations:

                    st.write(
                        f"- {limitation}"
                    )

                # ------------------------------------------------
                # Strengths
                # ------------------------------------------------

                st.subheader("💪 Strengths")

                for strength in result.strengths:

                    st.write(
                        f"- {strength}"
                    )

                # ------------------------------------------------
                # Weaknesses
                # ------------------------------------------------

                st.subheader("🔎 Weaknesses")

                for weakness in result.weaknesses:

                    st.write(
                        f"- {weakness}"
                    )

                # ------------------------------------------------
                # Explicit Research Gaps
                # ------------------------------------------------

                st.subheader(
                    "🔴 Research Gaps Explicitly Mentioned"
                )

                if result.explicit_research_gaps:

                    for gap in result.explicit_research_gaps:

                        st.write(
                            f"- {gap}"
                        )

                else:

                    st.write(
                        "No explicit research gaps were identified."
                    )

                # ------------------------------------------------
                # Inferred Research Gaps
                # ------------------------------------------------

                st.subheader(
                    "🟠 Research Gaps Inferred by Analysis"
                )

                if result.inferred_research_gaps:

                    for gap in result.inferred_research_gaps:

                        st.write(
                            f"- {gap}"
                        )

                else:

                    st.write(
                        "No additional research gaps inferred."
                    )

                # ------------------------------------------------
                # Future Work
                # ------------------------------------------------

                st.subheader("🚀 Future Work")

                if result.future_work:

                    for work in result.future_work:

                        st.write(
                            f"- {work}"
                        )

                else:

                    st.write(
                        "No future work mentioned."
                    )

                # ------------------------------------------------
                # Conclusion
                # ------------------------------------------------

                st.subheader("🎓 Conclusion")

                st.write(
                    result.conclusion
                )

                # ------------------------------------------------
                # SUCCESS
                # ------------------------------------------------

                st.success(
                    "Research paper analysis completed successfully!"
                )

            except Exception as e:

                st.error(
                    "An error occurred while analyzing the paper."
                )

                st.exception(e)