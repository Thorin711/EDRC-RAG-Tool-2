# -*- coding: utf-8 -*-
"""
Planned but not-yet-shipped features for app.py, pulled out of the live app on
2026-09-14 because they were commented out and rendering as empty tabs in the UI.

This file is not imported anywhere — it's a holding area. To revive a feature,
copy the relevant block back into app.py, re-add it to the `st.tabs(...)` call,
and restore any now-unused imports it depends on (see notes per block).

--------------------------------------------------------------------------------
FEATURE 1: "Author Explorer" tab
--------------------------------------------------------------------------------
Finds the top authors publishing on a given subject by scanning the Full
Database for the most similar documents and counting author frequency.

Needs these imports back in app.py: `import collections`, `import pandas as pd`,
`import altair as alt`.

    st.subheader("Find Top Authors by Subject")
    st.info("This tool searches the **Full Database** to find authors who have published most frequently on a given subject.")

    with st.form("author_search_form"):
        author_query = st.text_input(
            "Search Subject:",
            placeholder="e.g., carbon capture"
        )
        author_search_submitted = st.form_submit_button("Find Authors", type="primary", use_container_width=True)

    if author_search_submitted and author_query:
        DOC_SCAN_K = 1000
        with st.spinner(f"Searching Full Database for authors on '{author_query}' (scanning top {DOC_SCAN_K} docs)..."):
            try:
                # Pass the secure URL/KEY here too
                full_author_store = load_full_store(embeddings, qdrant_url, qdrant_api_key)

                search_results = full_author_store.similarity_search(
                    author_query,
                    k=DOC_SCAN_K
                )

                if not search_results:
                    st.warning("No documents found for this subject.")
                else:
                    author_counts = collections.Counter()
                    grouped_docs = group_results(search_results)

                    for group in grouped_docs:
                        meta = group['metadata']
                        authors_string = meta.get('authors', 'No Authors Found')

                        if authors_string != 'No Authors Found' and authors_string is not None:
                            individual_authors = [
                                name.strip() for name in authors_string.split(',')
                                if name.strip() and name.strip().lower() not in ['not available', 'n/a']
                            ]
                            author_counts.update(individual_authors)

                    if not author_counts:
                        st.info("Documents were found, but no author information was attached to them.")
                    else:
                        top_10_authors = author_counts.most_common(10)

                        st.subheader(f"Top 10 Authors on '{author_query}'")
                        st.write(f"(From {len(grouped_docs)} unique documents found in the top {DOC_SCAN_K} relevant docs)")

                        df = pd.DataFrame(top_10_authors, columns=["Author", "Relevant Publications Found"])

                        max_val = df["Relevant Publications Found"].max()
                        ylim_top = (max_val // 5 + 1) * 5

                        chart = alt.Chart(df).mark_bar().encode(
                            x=alt.X('Author', sort=None, axis=alt.Axis(labelAngle=0)),
                            y=alt.Y('Relevant Publications Found', scale=alt.Scale(domain=[0, ylim_top])),
                            tooltip=['Author', 'Relevant Publications Found']
                        ).interactive()

                        st.altair_chart(chart, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred during the author search: {e}")


# --------------------------------------------------------------------------------
# FEATURE 2: "Query Analyzer" tab
# --------------------------------------------------------------------------------
# Pastes in consultation text, uses an LLM to extract the individual questions,
# then checks each one against the Full Database with the reranker to flag
# which questions the corpus can actually answer.
#
# Needs `import json` back in app.py, plus the `extract_questions_with_llm`
# helper function below (it was removed from app.py along with this tab).

def extract_questions_with_llm(consultation_text):
    """Uses an LLM to extract a list of questions from a block of text."""
    try:
        prompt = f"""
        You are a text-processing bot. Your task is to read the following text and extract a list of all explicit and implicit questions.
        - Ignore headings, numbering, and introductory text.
        - Focus only on the questions themselves.
        - Return *only* a valid JSON list of strings. Do not include any preamble, markdown, or other text.

        Example output:
        ["What is the national vision?", "Why do you disagree?", "How can ATE support local authorities?"]

        --- Text to Analyze ---
        {consultation_text}
        """

        openai.api_key = get_secret("OPENAI_API_KEY")

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a text-processing bot that returns only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )

        response_text = response.choices[0].message.content.strip()

        usage = response.usage
        token_info = {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }

        try:
            data = json.loads(response_text)
            if isinstance(data, list):
                return data, token_info
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, list):
                        return value, token_info

            st.error(f"LLM returned valid JSON, but not in the expected format: {response_text}")
            return None, token_info

        except json.JSONDecodeError:
            st.error(f"Failed to decode JSON from LLM response: {response_text}")
            return None, token_info

    except Exception as e:
        st.warning(f"Could not extract questions due to an API error: {e}.")
        return None, None


"""
Tab body:

    st.subheader("Query Relevance Analyzer")
    st.info("Paste in a consultation or text, and this tool will extract the questions and check if the **Full Database** has relevant answers.")

    if not api_key_present:
        st.warning("`OPENAI_API_KEY` not found in secrets. This feature requires an API key to extract questions.", icon="⚠️")

    with st.form("query_analyzer_form"):
        consultation_text = st.text_area(
            "Paste Consultation Text Here:",
            placeholder="Paste your full consultation text... e.g., 'Question 1: Do you agree...?'",
            height=300
        )
        relevance_threshold = st.slider(
            "Relevance Threshold (Reranker Score):",
            min_value=-5.0, max_value=5.0, value=1.0, step=0.5,
            help="The minimum score from the reranker model to be considered 'Relevant'. A good starting point is 1.0."
        )
        analyze_submitted = st.form_submit_button("Analyze Questions", type="primary", use_container_width=True, disabled=not api_key_present)

    if analyze_submitted and consultation_text:
        extracted_questions = None
        extraction_token_info = None

        with st.spinner("Step 1/2: Extracting questions from text..."):
            openai.api_key = get_secret("OPENAI_API_KEY")
            extracted_questions, extraction_token_info = extract_questions_with_llm(consultation_text)

        if extraction_token_info:
            display_token_usage(extraction_token_info, "gpt-4o-mini", "Question Extraction")

        if extracted_questions:
            st.write(f"Found {len(extracted_questions)} questions. Now analyzing relevance against the **Full Database**...")

            with st.spinner(f"Step 2/2: Analyzing {len(extracted_questions)} questions... This may take a moment."):
                try:
                    reranker_model = load_reranker_model()
                    full_store = load_full_store(embeddings, qdrant_url, qdrant_api_key)

                    analysis_results = []
                    progress_bar = st.progress(0, text="Analyzing...")

                    for i, question in enumerate(extracted_questions):
                        initial_results = full_store.similarity_search(question, k=10)

                        top_score = -10.0

                        if initial_results:
                            reranker_pairs = [(question, d.page_content) for d in initial_results]
                            reranker_scores = reranker_model.predict(reranker_pairs)
                            top_score = max(reranker_scores)

                        is_relevant = top_score > relevance_threshold
                        analysis_results.append({
                            "question": question,
                            "top_score": top_score,
                            "is_relevant": is_relevant
                        })
                        progress_bar.progress((i + 1) / len(extracted_questions), text=f"Analyzing question {i+1}/{len(extracted_questions)}")

                    progress_bar.empty()
                    st.subheader("Analysis Complete")

                    for result in analysis_results:
                        with st.container(border=True):
                            st.markdown(f"**Question:** {result['question']}")

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(label="Top Relevance Score", value=f"{result['top_score']:.2f}")
                            with col2:
                                if result['is_relevant']:
                                    st.success("✅ Relevant")
                                else:
                                    st.error("❌ Not Relevant")

                except Exception as e:
                    st.error(f"An error occurred during relevance analysis: {e}")

        elif extracted_questions is None:
            pass
        else:
            st.info("No questions were found in the provided text.")
"""
