import "dotenv/config";

const COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank";
const RERANK_MODEL = "rerank-v3.5";
const MAX_RETRIES = 3;
const INITIAL_BACKOFF_MS = 800;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isRetryableStatus(status) {
  return status === 429 || (status >= 500 && status <= 599);
}

/**
 * Reranks candidate chunks against the query using Cohere's cross-encoder
 * rerank model. Falls back to the original vector-ranked order (with a
 * console warning) if the rerank API is unavailable, so a transient API
 * outage never breaks retrieval.
 *
 * @param {string} query
 * @param {Array<object>} chunks - rows from retrieveChunks (must have .content)
 * @param {number} topN - how many reranked chunks to keep
 */
export async function rerankChunks(query, chunks, topN = 5) {
  if (chunks.length === 0) return chunks;
  if (!process.env.COHERE_API_KEY) {
    console.warn("COHERE_API_KEY not set — skipping rerank, using vector order.");
    return chunks.slice(0, topN);
  }

  let retryCount = 0;

  while (true) {
    try {
      const response = await fetch(COHERE_RERANK_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${process.env.COHERE_API_KEY}`,
        },
        body: JSON.stringify({
          model: RERANK_MODEL,
          query,
          documents: chunks.map((c) => c.content),
          top_n: Math.min(topN, chunks.length),
        }),
      });

      if (!response.ok) {
        const errText = await response.text();
        const error = new Error(`Cohere rerank failed (${response.status}): ${errText}`);
        error.status = response.status;
        throw error;
      }

      const { results } = await response.json();

      // results: [{ index, relevance_score }], already sorted desc by relevance_score
      return results.map((r) => ({
        ...chunks[r.index],
        rerank_score: r.relevance_score,
        vector_similarity: chunks[r.index].similarity,
      }));
    } catch (error) {
      const retryable = isRetryableStatus(error.status) || error.name === "TypeError";
      if (!retryable || retryCount >= MAX_RETRIES) {
        console.error(`Rerank failed after ${retryCount + 1} attempt(s): ${error.message}`);
        console.warn("Falling back to vector-similarity order.");
        return chunks.slice(0, topN);
      }

      const backoffMs = INITIAL_BACKOFF_MS * 2 ** retryCount;
      retryCount += 1;
      console.warn(`Retrying rerank (attempt ${retryCount + 1}/${MAX_RETRIES + 1}) in ${backoffMs}ms: ${error.message}`);
      await sleep(backoffMs);
    }
  }
}
