// Builds the system prompt for lecture generation
export function buildLectureSystemPrompt() {
  return `You are an expert academic lecture writer.
Your job is to generate a structured lecture using ONLY the provided knowledge base context.
Do not use outside knowledge. If the context is insufficient for any section, explicitly say so.

You MUST respond with valid JSON only — no markdown, no explanation outside the JSON.

The JSON structure must be exactly:
{
  "title": "string",
  "summary": "string (2-3 sentences overview)",
  "sections": [
    {
      "heading": "string",
      "content": "string (detailed explanation)",
      "key_points": ["string", "string"],
      "cited_chunk_ids": ["uuid", "uuid"]
    }
  ],
  "conclusion": "string",
  "suggested_questions": ["string", "string", "string"],
  "verification_notes": "string (self-assessment: how well context covered the topic, any gaps)"
}`;
}

// Builds the user prompt with retrieved context injected
export function buildLectureUserPrompt(request, chunks) {
  const contextBlock = chunks
    .map(
      (c, i) =>
        `[CHUNK ${i + 1}] id="${c.chunk_id}" source="${c.source_title}" similarity=${(c.similarity * 100).toFixed(1)}%\n${c.content}`
    )
    .join("\n\n---\n\n");

  return `Generate a lecture with the following requirements:

TOPIC: ${request.topic}
TARGET AUDIENCE: ${request.audience}
DURATION: ${request.duration} minutes
LEARNING OBJECTIVE: ${request.learning_objective}

Use ONLY the context below. Reference chunk IDs in "cited_chunk_ids" for each section.

<context>
${contextBlock}
</context>`;
}