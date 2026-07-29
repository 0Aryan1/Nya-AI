// Builds the system prompt for lecture generation, tuned for legal education.
export function buildLectureSystemPrompt() {
  return `You are an expert law school lecturer preparing a class-ready lecture for law students.
Your job is to generate a structured, legally rigorous lecture using ONLY the provided knowledge base context.
Do not use outside knowledge, and do not fill gaps with general legal knowledge you were trained on — if the context is insufficient for any section, explicitly say so rather than inventing rules, cases, or citations.

LEGAL-EDUCATION REQUIREMENTS:
1. AUTHORITY AWARENESS: When the context discusses cases, statutes, or regulations, distinguish binding authority from persuasive authority wherever the context makes this clear (e.g. jurisdiction, court level). If the context does not specify, say so rather than assuming.
2. PRECISE CITATION: When a section relies on a specific case, statute, or provision named in the context, name it explicitly in "content" (case name, statute/section number) — not just the internal chunk ID. Chunk IDs in "cited_chunk_ids" are for internal traceability only and are never a substitute for a real legal citation in the text.
3. IRAC WHERE APPLICABLE: For any section analyzing how a rule applies to facts or a hypothetical, structure that analysis as Issue → Rule → Application → Conclusion, either as explicit sub-headings within "content" or as a clearly signposted paragraph flow.
4. CURRENCY CAVEAT: Legal rules can be amended, overruled, or superseded after the source material was written. If the context gives no indication of how current the stated law is, note this uncertainty in "verification_notes" rather than presenting it as settled.
5. DEFINE TERMS OF ART: Any Latin maxim, doctrine name, or technical legal term central to the topic should be captured in "key_terms" with a plain-language definition grounded in the context.
6. EXAM/ISSUE-SPOTTING ORIENTATION: "suggested_questions" should be law-school style — issue-spotting prompts, "what if the facts were X instead" hypotheticals, or questions that test whether the rule was correctly understood — not generic comprehension questions.
7. NOT LEGAL ADVICE: This lecture is for academic instruction only. Do not phrase conclusions as advice to a specific real-world situation.

You MUST respond with valid JSON only — no markdown, no explanation outside the JSON.

The JSON structure must be exactly:
{
  "title": "string",
  "summary": "string (2-3 sentences overview)",
  "key_terms": [
    { "term": "string", "definition": "string" }
  ],
  "sections": [
    {
      "heading": "string",
      "content": "string (detailed explanation; use IRAC structure when analyzing rule application)",
      "key_points": ["string", "string"],
      "cited_chunk_ids": ["uuid", "uuid"]
    }
  ],
  "conclusion": "string",
  "suggested_questions": ["string", "string", "string"],
  "verification_notes": "string (self-assessment: how well context covered the topic, any gaps, any currency/jurisdiction uncertainty)"
}`;
}

// ─── Few-shot example ────────────────────────────────────────────────────────
// A single worked example demonstrating the exact behavior we want: IRAC
// structure, named legal citations in prose, chunk-ID-only internal citation,
// key_terms, a currency caveat, and issue-spotting style questions. The fake
// chunk IDs here are self-contained to this example and never collide with
// real UUIDs from the database.

const FEW_SHOT_USER_PROMPT = `Generate a law-school lecture with the following requirements:

TOPIC: Offer and Acceptance in Contract Formation
TARGET AUDIENCE: 1st year law students
DURATION: 30 minutes
LEARNING OBJECTIVE: Understand how a valid offer and acceptance form a binding contract, and how revocation and counter-offers affect this.
JURISDICTION: India

Use ONLY the context below. Name specific cases/statutes/provisions in your prose when the context provides them, and reference chunk IDs in "cited_chunk_ids" for each section for internal traceability.

<context>
[CHUNK 1] id="ex-chunk-001" source="Indian Contract Act, 1872 - Commentary" similarity=91.2%
Section 2(a) of the Indian Contract Act, 1872 defines a proposal (offer) as when one person signifies to another their willingness to do or abstain from doing something, with a view to obtaining the assent of that other person. Section 2(b) defines acceptance as the signification of assent by the person to whom the proposal is made.

---

[CHUNK 2] id="ex-chunk-002" source="Indian Contract Act, 1872 - Commentary" similarity=87.5%
In Lalman Shukla v. Gauri Datt (1913), the Allahabad High Court held that an offer must be communicated to the offeree before it can be accepted; a person cannot accept an offer they were unaware of. This case remains a leading authority on the communication requirement for offers in Indian contract law.

---

[CHUNK 3] id="ex-chunk-003" source="Indian Contract Act, 1872 - Commentary" similarity=82.0%
A counter-offer operates as a rejection of the original offer. Once rejected, the original offer cannot later be accepted unless the offeror renews it.
</context>`;

const FEW_SHOT_ASSISTANT_RESPONSE = JSON.stringify(
  {
    title: "Offer and Acceptance: Foundations of Contract Formation",
    summary:
      "This lecture covers how a valid offer and acceptance combine to form a binding contract under Indian law, focusing on the communication requirement and the effect of counter-offers.",
    key_terms: [
      {
        term: "Proposal (Offer)",
        definition:
          "Under Section 2(a) of the Indian Contract Act, 1872, a signification by one person to another of willingness to do or abstain from doing something, made with a view to obtaining that other person's assent.",
      },
      {
        term: "Acceptance",
        definition:
          "Under Section 2(b), the signification of assent by the person to whom a proposal is made.",
      },
      {
        term: "Counter-offer",
        definition:
          "A response to an offer that changes its terms; it rejects the original offer rather than accepting it.",
      },
    ],
    sections: [
      {
        heading: "What Makes an Offer Valid",
        content:
          "Section 2(a) and 2(b) of the Indian Contract Act, 1872 set out the basic definitions of offer and acceptance. An offer is not effective in a vacuum, however — it must reach the intended offeree.\n\nISSUE: Can a person accept an offer they did not know existed?\nRULE: In Lalman Shukla v. Gauri Datt (1913), the Allahabad High Court held that an offer must be communicated to the offeree before it can be accepted.\nAPPLICATION: If A offers a reward for finding a lost item but B finds and returns it without knowing of the reward, B cannot claim the reward, because B never accepted an offer they were unaware of.\nCONCLUSION: Communication of the offer to the offeree is a precondition for a valid acceptance.",
        key_points: [
          "Offer and acceptance are statutorily defined under Sections 2(a) and 2(b).",
          "An offer must be communicated before it can be validly accepted (Lalman Shukla v. Gauri Datt).",
        ],
        cited_chunk_ids: ["ex-chunk-001", "ex-chunk-002"],
      },
      {
        heading: "Counter-Offers and Rejection",
        content:
          "A counter-offer does not keep the original offer alive — it operates as a rejection of it. Once rejected, the original offer lapses and cannot later be accepted unless the offeror chooses to renew it.",
        key_points: [
          "A counter-offer is legally treated as a rejection, not a conditional acceptance.",
          "The original offer cannot be revived by the offeree after a counter-offer, without the offeror's renewal.",
        ],
        cited_chunk_ids: ["ex-chunk-003"],
      },
    ],
    conclusion:
      "A binding contract requires a communicated offer and a matching acceptance; introducing new terms via a counter-offer rejects the original offer rather than forming a contract on the new terms.",
    suggested_questions: [
      "If B performs an act matching A's offer terms but only learns of the offer afterward, has B accepted it? Why or why not?",
      "A offers to sell a car for ₹5,00,000. B replies 'I'll take it for ₹4,50,000.' A refuses to sell at either price. Has any contract formed? What is B's legal position if A later tries to accept the original ₹5,00,000 offer?",
      "How would the outcome in Lalman Shukla v. Gauri Datt change, if at all, if the offeree later discovered the offer but the underlying task was already complete?",
    ],
    verification_notes:
      "Context covered the core definitions and the communication and counter-offer rules well. The context does not indicate whether Lalman Shukla v. Gauri Datt has been distinguished or narrowed by later Indian case law, so its current precedential weight beyond the stated principle is not verified from this context alone.",
  },
  null,
  2
);

export function buildFewShotExample() {
  return {
    userPrompt: FEW_SHOT_USER_PROMPT,
    assistantResponse: FEW_SHOT_ASSISTANT_RESPONSE,
  };
}

// Builds the user prompt with retrieved context injected
export function buildLectureUserPrompt(request, chunks) {
  const contextBlock = chunks
    .map(
      (c, i) =>
        `[CHUNK ${i + 1}] id="${c.chunk_id}" source="${c.source_title}" similarity=${(c.similarity * 100).toFixed(1)}%\n${c.content}`
    )
    .join("\n\n---\n\n");

  const jurisdictionLine = request.jurisdiction
    ? `\nJURISDICTION: ${request.jurisdiction}`
    : "";

  return `Generate a law-school lecture with the following requirements:

TOPIC: ${request.topic}
TARGET AUDIENCE: ${request.audience}
DURATION: ${request.duration} minutes
LEARNING OBJECTIVE: ${request.learning_objective}${jurisdictionLine}

Use ONLY the context below. Name specific cases/statutes/provisions in your prose when the context provides them, and reference chunk IDs in "cited_chunk_ids" for each section for internal traceability.

<context>
${contextBlock}
</context>`;
}