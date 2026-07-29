import { ChatOpenAI } from "@langchain/openai";
import "dotenv/config";

export const LLM_MAX_RETRIES = 3;
export const LLM_INITIAL_BACKOFF_MS = 1000;

const RETRYABLE_NETWORK_CODES = new Set([
  "ECONNRESET",
  "ECONNREFUSED",
  "EAI_AGAIN",
  "ENETUNREACH",
  "ETIMEDOUT",
]);

export const llm = new ChatOpenAI({
  model: "glm-4.7-flash",
  apiKey: process.env.ZAI_API_KEY,
  configuration: {
    baseURL: "https://api.z.ai/api/paas/v4",
  },
  temperature: 0.7,
  maxRetries: 0, // we handle retries ourselves below, so LangChain shouldn't also retry silently
});

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function getErrorStatus(error) {
  return Number(error?.status ?? error?.response?.status ?? error?.cause?.status);
}

function isRetryableError(error) {
  const status = getErrorStatus(error);
  const message = String(error?.message ?? "").toLowerCase();

  return (
    status === 429 ||
    (status >= 500 && status <= 599) ||
    RETRYABLE_NETWORK_CODES.has(error?.code) ||
    message.includes("rate limit") ||
    message.includes("resource exhausted") ||
    message.includes("quota exceeded") ||
    message.includes("timeout") ||
    message.includes("fetch failed")
  );
}

function describeError(error) {
  const status = getErrorStatus(error);
  const message = String(error?.message ?? error);

  if (status === 429 || /rate limit/i.test(message)) {
    return `GLM rate limit reached: ${message}`;
  }
  if (status >= 500 && status <= 599) {
    return `GLM service temporarily failed with HTTP ${status}: ${message}`;
  }
  if (RETRYABLE_NETWORK_CODES.has(error?.code) || /network|fetch failed/i.test(message)) {
    return `Network failure while contacting GLM: ${message}`;
  }
  return `GLM request failed: ${message}`;
}

/**
 * Invokes the LLM with the same retry/backoff pattern used for embeddings
 * and reranking elsewhere in this project. Unlike reranking, there is no
 * soft fallback here — lecture generation has no meaningful "default" output,
 * so exhausted retries still throw, just with a clearer error message.
 *
 * @param {Array} messages - LangChain message array (SystemMessage, HumanMessage, ...)
 * @param {object} opts
 * @param {number} opts.maxRetries
 * @param {number} opts.initialBackoffMs
 */
export async function invokeLLMWithRetry(
  messages,
  { maxRetries = LLM_MAX_RETRIES, initialBackoffMs = LLM_INITIAL_BACKOFF_MS } = {}
) {
  let retryCount = 0;

  while (true) {
    try {
      return await llm.invoke(messages);
    } catch (error) {
      if (!isRetryableError(error) || retryCount >= maxRetries) {
        throw new Error(
          `${describeError(error)}. Failed after ${retryCount + 1} attempt(s).`,
          { cause: error }
        );
      }

      const backoffMs = initialBackoffMs * 2 ** retryCount;
      retryCount += 1;
      console.warn(
        `Retrying GLM call (attempt ${retryCount + 1}/${maxRetries + 1}) in ${backoffMs}ms: ${describeError(error)}`
      );
      await sleep(backoffMs);
    }
  }
}
