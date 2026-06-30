import { pool } from "./db.js";
import { embedder } from "./embeddings.js";
import "dotenv/config";

/**
 * @param {string} question
 * @param {object} opts
 * @param {number} opts.topK       - number of chunks to return (default 5)
 * @param {string} opts.topic      - filter by topic (optional)
 * @param {string} opts.sourceId   - filter by specific PDF (optional)
 * @param {number} opts.threshold  - min similarity 0–1 (default 0.5)
 */
export async function retrieveChunks(question, {
  topK = 5,
  topic = null,
  sourceId = null,
  threshold = 0.5,
} = {}) {
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

  params.push(topK);

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
  try {
    const { rows } = await client.query(sql, params);
    return rows;
  } finally {
    client.release();
  }
}

// ─── CLI test ─────────────────────────────────────────────────────────────── 

// node src/query.js "what is gradient descent"
const question = process.argv.slice(2).join(" ");
if (question) {
  const chunks = await retrieveChunks(question, { topK: 1 });
  chunks.forEach((c, i) => {
    console.log(`\n[${i + 1}] ${c.source_title} | similarity: ${(c.similarity * 100).toFixed(1)}%`);
    console.log(c.content.slice(0, 300) + "...");
  });
  await pool.end();
}