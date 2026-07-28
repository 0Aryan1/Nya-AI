import { pool } from "./db.js";
import { embedder } from "./embeddings.js";
import { rerankChunks } from "./rerank.js";
import "dotenv/config";

const RERANK_FETCH_MULTIPLIER = 4;

/**
 * @param {string} question
 * @param {object} opts
 * @param {number} opts.topK       - number of chunks to return (default 5)
 * @param {string} opts.topic      - filter by topic (optional)
 * @param {string} opts.sourceId   - filter by specific PDF (optional)
 * @param {number} opts.threshold  - min similarity 0–1 (default 0.5)
 * @param {boolean} opts.rerank    - cross-encoder rerank the candidates (default false)
 * @param {number} opts.fetchK     - candidate pool size fetched before reranking
 *                                   (default topK * 4, only used when rerank=true)
 */
export async function retrieveChunks(question, {
  topK = 5,
  topic = null,
  sourceId = null,
  threshold = 0.5,
  rerank = false,
  fetchK = null,
} = {}) {
  const candidateLimit = rerank ? (fetchK ?? topK * RERANK_FETCH_MULTIPLIER) : topK;
  const queryVector = await embedder.embedQuery(question);
  const vectorStr = `[${queryVector.join(",")}]`;

  // Build WHERE clauses dynamically
  const conditions = [
    `1 - (kc.embedding <=> $1::vector) >= $2`  // similarity threshold
  ];
  const params = [vectorStr, threshold];
  let paramIdx = 3;

  if (topic) {
    conditions.push(`kc.topic = $${paramIdx++}`);
    params.push(topic);
  }
  if (sourceId) {
    conditions.push(`kc.source_id = $${paramIdx++}`);
    params.push(sourceId);
  }

  params.push(candidateLimit);

  const sql = `
    SELECT
      kc.id          AS chunk_id,
      kc.content,
      kc.topic,
      kc.source_id,
      ks.title       AS source_title,
      ks.source_type,
      1 - (kc.embedding <=> $1::vector) AS similarity
    FROM knowledge_chunks kc
    JOIN knowledge_sources ks ON ks.id = kc.source_id
    WHERE ${conditions.join(" AND ")}
    ORDER BY kc.embedding <=> $1::vector
    LIMIT $${paramIdx}
  `;

  const client = await pool.connect();
  let rows;
  try {
    ({ rows } = await client.query(sql, params));
  } finally {
    client.release();
  }

  if (!rerank || rows.length === 0) return rows;

  return rerankChunks(question, rows, topK);
}

// ─── CLI test ─────────────────────────────────────────────────────────────── 

// node src/query.js "what is gradient descent"        -> vector search only
// node src/query.js "what is gradient descent" --rerank -> vector search + cross-encoder rerank
const args = process.argv.slice(2);
const useRerank = args.includes("--rerank");
const question = args.filter((a) => a !== "--rerank").join(" ");
if (question) {
  const chunks = await retrieveChunks(question, { topK: 3, rerank: useRerank });
  chunks.forEach((c, i) => {
    const scoreLabel = useRerank
      ? `rerank: ${(c.rerank_score * 100).toFixed(1)}% | vector: ${(c.vector_similarity * 100).toFixed(1)}%`
      : `similarity: ${(c.similarity * 100).toFixed(1)}%`;
    console.log(`\n[${i + 1}] ${c.source_title} | ${scoreLabel}`);
    console.log(c.content.slice(0, 300) + "...");
  });
  await pool.end();
}