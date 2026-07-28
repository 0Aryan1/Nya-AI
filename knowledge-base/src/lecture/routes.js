import { Router } from "express";
import { z } from "zod";
import { generateLecture } from "./generator.js";
import { pool } from "../db.js";

export const lectureRouter = Router();

// ─── Input schema ─────────────────────────────────────────────────────────────

const GenerateSchema = z.object({
  topic: z.string().min(3),
  audience: z.string().min(2),
  duration: z.number().int().min(5).max(180),
  learning_objective: z.string().min(10),
  topic_filter: z.string().optional(),   // matches `topic` column in knowledge_chunks
  jurisdiction: z.string().optional(),   // e.g. "India", "US Federal" — informs authority framing in the prompt
});

// ─── POST /api/lectures/generate ─────────────────────────────────────────────

lectureRouter.post("/generate", async (req, res) => {
  const parsed = GenerateSchema.safeParse(req.body);
  if (!parsed.success) {
    return res.status(400).json({ error: parsed.error.flatten() });
  }

  try {
    const result = await generateLecture(parsed.data);
    return res.status(201).json(result);
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: err.message });
  }
});

// ─── GET /api/lectures/:id ────────────────────────────────────────────────────
// Returns lecture + its citations with source info

lectureRouter.get("/:id", async (req, res) => {
  const parsedId = z.string().uuid().safeParse(req.params.id);
  if (!parsedId.success) {
    return res.status(400).json({ error: "Lecture ID must be a valid UUID" });
  }

  const client = await pool.connect();
  try {
    // Fetch lecture
    const { rows: lectureRows } = await client.query(
      `SELECT gl.*, lr.topic, lr.audience, lr.duration, lr.learning_objective
       FROM generated_lectures gl
       JOIN lecture_requests lr ON lr.id = gl.request_id
       WHERE gl.id = $1`,
      [parsedId.data]
    );

    if (lectureRows.length === 0) {
      return res.status(404).json({ error: "Lecture not found" });
    }

    // Fetch citations with source info
    const { rows: citationRows } = await client.query(
      `SELECT lc.citation_text, lc.source_url,
              kc.content      AS chunk_content,
              kc.topic        AS chunk_topic,
              ks.title        AS source_title,
              ks.source_type
       FROM lecture_citations lc
       LEFT JOIN knowledge_chunks kc ON kc.id = lc.chunk_id
       LEFT JOIN knowledge_sources ks ON ks.id = kc.source_id
       WHERE lc.lecture_id = $1`,
      [parsedId.data]
    );

    return res.json({
      lecture: lectureRows[0],
      citations: citationRows,
    });
  } finally {
    client.release();
  }
});

// ─── GET /api/lectures ────────────────────────────────────────────────────────
// List all lectures (summary only)

lectureRouter.get("/", async (req, res) => {
  const client = await pool.connect();
  try {
    const { rows } = await client.query(
      `SELECT gl.id, gl.verification_score, gl.created_at,
              lr.topic, lr.audience, lr.duration
       FROM generated_lectures gl
       JOIN lecture_requests lr ON lr.id = gl.request_id
       ORDER BY gl.created_at DESC
       LIMIT 50`
    );
    return res.json(rows);
  } finally {
    client.release();
  }
});
