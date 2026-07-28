import { pool } from "../db.js";
import { llm } from "../llm.js";
import { retrieveChunks } from "../query.js";
import { buildLectureSystemPrompt, buildLectureUserPrompt } from "./prompts.js";
import { SystemMessage, HumanMessage } from "@langchain/core/messages";

// ─── Step 1: Save the request ────────────────────────────────────────────────

async function insertLectureRequest(client, request) {
  const { rows } = await client.query(
    `INSERT INTO lecture_requests (topic, audience, duration, learning_objective)
     VALUES ($1, $2, $3, $4)
     RETURNING id`,
    [request.topic, request.audience, request.duration, request.learning_objective]
  );
  return rows[0].id;
}

// ─── Step 2: Call GLM and parse JSON safely ──────────────────────────────────

async function generateLectureContent(request, chunks) {
  const response = await llm.invoke([
    new SystemMessage(buildLectureSystemPrompt()),
    new HumanMessage(buildLectureUserPrompt(request, chunks)),
  ]);

  const raw = response.content.trim();

  // Strip markdown code fences if GLM wraps in ```json ... ```
  const cleaned = raw.replace(/^```(?:json)?\n?/, "").replace(/\n?```$/, "");

  try {
    return JSON.parse(cleaned);
  } catch (err) {
    console.error("❌ GLM returned invalid JSON:\n", raw);
    throw new Error("LLM did not return valid JSON. Raw output logged above.");
  }
}

// ─── Step 3: Compute verification score ─────────────────────────────────────
// Simple heuristic: avg similarity of chunks actually cited across sections

function computeVerificationScore(lectureJSON, chunks) {
  // Collect all cited chunk IDs from all sections
  const citedIds = new Set(
    lectureJSON.sections.flatMap((s) => s.cited_chunk_ids ?? [])
  );

  if (citedIds.size === 0) return 0;

  // Average similarity of cited chunks
  const citedChunks = chunks.filter((c) => citedIds.has(c.chunk_id));
  if (citedChunks.length === 0) return 0;

  const avgSimilarity =
    citedChunks.reduce((sum, c) => sum + parseFloat(c.similarity), 0) /
    citedChunks.length;

  // Scale to 0–100
  return parseFloat((avgSimilarity * 100).toFixed(2));
}

// ─── Step 4: Save lecture + citations atomically ─────────────────────────────

async function insertLectureAndCitations(client, requestId, lectureJSON, verificationScore, chunks) {
  // Save generated lecture
  const { rows } = await client.query(
    `INSERT INTO generated_lectures (request_id, lecture_content, verification_score)
     VALUES ($1, $2, $3)
     RETURNING id`,
    [requestId, JSON.stringify(lectureJSON), verificationScore]
  );
  const lectureId = rows[0].id;

  // Build a lookup map: chunk_id → chunk row (for citation_text)
  const chunkMap = Object.fromEntries(chunks.map((c) => [c.chunk_id, c]));

  // Insert one citation per cited chunk across all sections
  const allCitedIds = [
    ...new Set(lectureJSON.sections.flatMap((s) => s.cited_chunk_ids ?? [])),
  ];

  for (const chunkId of allCitedIds) {
    const chunk = chunkMap[chunkId];
    if (!chunk) continue; // LLM hallucinated an ID — skip

    await client.query(
      `INSERT INTO lecture_citations (lecture_id, chunk_id, citation_text)
       VALUES ($1, $2, $3)`,
      [
        lectureId,
        chunkId,
        // citation_text = first 200 chars of the chunk as a snippet
        chunk.content.slice(0, 200).trim(),
      ]
    );
  }

  return lectureId;
}

// ─── Main orchestrator ───────────────────────────────────────────────────────

export async function generateLecture(request) {
  console.log(`\n🎓 Generating lecture: "${request.topic}"`);

  // 1. Retrieve relevant chunks from knowledge base
  console.log("   Retrieving chunks...");
  const chunks = await retrieveChunks(
    `${request.topic} ${request.learning_objective}`,
    {
      topK: 8,
      topic: request.topic_filter ?? null,  // optional: filter by ingested topic tag
      threshold: 0.4,
      rerank: true,   // over-fetch candidates, then cross-encoder rerank down to topK
      fetchK: 24,
    }
  );

  if (chunks.length === 0) {
    throw new Error(
      "No relevant chunks found. Please ingest PDFs related to this topic first."
    );
  }
  console.log(`   Found ${chunks.length} relevant chunks`);

  // 2. Generate lecture via GLM
  console.log("   Calling Z.ai GLM...");
  const lectureJSON = await generateLectureContent(request, chunks);
  console.log(`   Lecture generated: ${lectureJSON.sections.length} sections`);

  // 3. Compute score
  const verificationScore = computeVerificationScore(lectureJSON, chunks);
  console.log(`   Verification score: ${verificationScore}/100`);

  // 4. Persist everything in one transaction
  const client = await pool.connect();
  try {
    await client.query("BEGIN");

    const requestId = await insertLectureRequest(client, request);
    const lectureId = await insertLectureAndCitations(
      client,
      requestId,
      lectureJSON,
      verificationScore,
      chunks
    );

    await client.query("COMMIT");
    console.log(`✅ Saved — lecture ID: ${lectureId}\n`);

    return {
      lectureId,
      requestId,
      verificationScore,
      lecture: lectureJSON,
      chunksUsed: chunks.length,
    };
  } catch (err) {
    await client.query("ROLLBACK");
    throw err;
  } finally {
    client.release();
  }
}