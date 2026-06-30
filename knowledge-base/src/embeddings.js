import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import "dotenv/config";

export const CONCURRENCY = 3;
export const MAX_RETRIES = 5;
export const INITIAL_BACKOFF_MS = 1000;
export const REQUEST_DELAY_MS = 1200;

const EMBEDDING_DIMENSIONS = 768;
const RETRYABLE_NETWORK_CODES = new Set([
  "ECONNRESET",
  "ECONNREFUSED",
  "EAI_AGAIN",
  "ENETUNREACH",
  "ETIMEDOUT",
]);

class FixedDimensionGoogleEmbeddings extends GoogleGenerativeAIEmbeddings {
  _convertToContent(text) {
    return {
      ...super._convertToContent(text),
      outputDimensionality: EMBEDDING_DIMENSIONS,
    };
  }
}

class InvalidEmbeddingResponseError extends Error {
  constructor(message) {
    super(message);
    this.name = "InvalidEmbeddingResponseError";
    this.code = "INVALID_EMBEDDING_RESPONSE";
  }
}

export const embedder = new FixedDimensionGoogleEmbeddings({
  model: "gemini-embedding-001",
  apiKey: process.env.GOOGLE_API_KEY,
});

export function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function validateEmbedding(vector) {
  if (!Array.isArray(vector) || vector.length !== EMBEDDING_DIMENSIONS) {
    throw new InvalidEmbeddingResponseError(
      `Embedding is empty or is not ${EMBEDDING_DIMENSIONS}-dimensional`
    );
  }
}

export async function embedChunk(text) {
  const vector = await embedder.embedQuery(text);
  validateEmbedding(vector);
  return vector;
}

function getErrorStatus(error) {
  return Number(error?.status ?? error?.response?.status ?? error?.cause?.status);
}

function isRetryableError(error) {
  const status = getErrorStatus(error);
  const message = String(error?.message ?? "").toLowerCase();

  return (
    error?.code === "INVALID_EMBEDDING_RESPONSE" ||
    status === 429 ||
    (status >= 500 && status <= 599) ||
    RETRYABLE_NETWORK_CODES.has(error?.code) ||
    message.includes("rate limit") ||
    message.includes("resource exhausted") ||
    message.includes("quota exceeded") ||
    message.includes("temporarily unavailable") ||
    message.includes("fetch failed")
  );
}

function describeError(error) {
  const status = getErrorStatus(error);
  const message = String(error?.message ?? error);

  if (/quota|resource exhausted/i.test(message)) {
    return `Gemini quota exceeded: ${message}`;
  }
  if (status === 429 || /rate limit/i.test(message)) {
    return `Gemini rate limit reached: ${message}`;
  }
  if (status >= 500 && status <= 599) {
    return `Gemini service temporarily failed with HTTP ${status}: ${message}`;
  }
  if (error?.code === "INVALID_EMBEDDING_RESPONSE") {
    return `Gemini returned an invalid embedding response: ${message}`;
  }
  if (
    RETRYABLE_NETWORK_CODES.has(error?.code) ||
    /network|fetch failed/i.test(message)
  ) {
    return `Network failure while contacting Gemini: ${message}`;
  }
  return `Gemini embedding request failed: ${message}`;
}

export async function embedChunkWithRetry(
  text,
  chunkIndex,
  {
    maxRetries = MAX_RETRIES,
    initialBackoffMs = INITIAL_BACKOFF_MS,
  } = {}
) {
  let retryCount = 0;

  while (true) {
    try {
      return await embedChunk(text);
    } catch (error) {
      if (!isRetryableError(error) || retryCount >= maxRetries) {
        throw new Error(
          `${describeError(error)}. Chunk ${chunkIndex + 1} failed after ${
            retryCount + 1
          } attempt(s).`,
          { cause: error }
        );
      }

      const backoffMs = initialBackoffMs * 2 ** retryCount;
      retryCount += 1;
      console.warn(
        `Retry chunk ${chunkIndex + 1} (Attempt ${
          retryCount + 1
        }/${maxRetries + 1}) in ${backoffMs}ms: ${describeError(error)}`
      );
      await sleep(backoffMs);
    }
  }
}
