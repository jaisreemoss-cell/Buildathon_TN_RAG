# Project note: Tamil Nadu paddy schemes RAG app

1. **Scrape with Playwright.** `PlaywrightURLLoader` opens the TNAU page in a headless browser and extracts its text, so content rendered by JavaScript is captured too. The trade-off is that it's slower and heavier than a simple HTML scraper. It relies on `unstructured` to turn the rendered page into plain text.

2. **Filter to Tamil Nadu only.** We keep only the text between the "Tamil Nadu" and "Kerala" headings. Keyword filtering was rejected because lines inside the Kerala and Karnataka sections don't repeat the state name, so they slipped through and would have been mistaken for Tamil Nadu schemes.

3. **Save to a file.** The filtered text is saved as `scraped_schemes.txt`. The app builds its knowledge base from this file, so the website doesn't need to be scraped again on every run.

4. **Load the API key.** `python-dotenv` reads `OPENAI_API_KEY` from the `.env` file so the key never appears in the code. The app stops with a clear message if the key is missing.

5. **Load the data.** `TextLoader` reads `scraped_schemes.txt` into a LangChain `Document`, the standard format the rest of the pipeline expects.

6. **Clean the text.** Extra spaces and leftover "Top" links are removed. They add nothing to the meaning and only waste tokens in the embeddings.

7. **Split the text (chunking).**
   - **Why split at all:** the model answers better from small, focused pieces than from one large document, and retrieval works by comparing small pieces against the question.
   - **Why not `chunk_size=1000, chunk_overlap=200`:** fixed-size splitting cut schemes in half, so one scheme's subsidy amount could end up next to another scheme's district list. Raising the overlap to 400 only created near-duplicate chunks.
   - **What we did instead:** we split at each scheme heading, giving 12 chunks of one complete scheme each, so no overlap is needed. `RecursiveCharacterTextSplitter` (2000/200) stays only as a safety net for any unusually long scheme.

8. **Add metadata.** Each chunk is tagged with `state`, `source`, `chunk_id` and `heading`. The UI uses the heading to show which schemes an answer came from.

9. **Create embeddings.** `text-embedding-3-small` turns each chunk into a vector of numbers that captures its meaning, so similar questions and chunks end up close together. This costs about 3,000 tokens in total, once.

10. **Store vectors in FAISS.** FAISS is a local vector database that finds the chunks closest to a question. The index is saved to `faiss_index` and reloaded on later runs, and rebuilt automatically when the text file changes.

11. **Retrieve (the `k` value).** For each question, the retriever returns the top `k` most similar chunks. We use `k=8` out of 12 chunks: a small `k` like 3 misses schemes on "list all" questions, and with such a small corpus, 8 costs only about 2,000 tokens per question. The Streamlit slider lets you adjust it from 3 to 12.

12. **Write a strict prompt.** The prompt tells the model to answer only from the retrieved context, to say when information isn't available, and to name the scheme for every amount it mentions. This is what prevents made-up answers.

13. **Choose the LLM.** `gpt-4o-mini` with `temperature=0` gives consistent, factual answers instead of creative ones.

14. **Parse the output.** `StrOutputParser` turns the model's response object into a plain string ready to display.

15. **Build the chain (LCEL).** The pieces are connected as `prompt | llm | StrOutputParser()`. The pipe syntax passes each step's output to the next. Retrieval runs just before the chain so the UI can also show the sources.

16. **Add the Streamlit UI.** The app provides a chat interface with streamed answers, a "Scheme sections used" list under each answer, a `k` slider, example questions and a rebuild button. `@st.cache_resource` loads the index and model once instead of on every click.

17. **Manage dependencies.** `requirements.txt` lists all packages. Run `pip install -r requirements.txt`, then `playwright install chromium` once for the browser.
