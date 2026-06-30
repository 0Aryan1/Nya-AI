import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { PDFParse } from "pdf-parse";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Document } from "@langchain/core/documents";
import { pool } from "./db.js";
import {
  CONCURRENCY,
  REQUEST_DELAY_MS,
  embedChunkWithRetry,
  sleep,
} from "./embeddings.js";
import "dotenv/config";

const CHUNK_SIZE = 700;
const CHUNK_OVERLAP = 100;
const PREVIEW_LENGTH = 300;

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: CHUNK_SIZE,
  chunkOverlap: CHUNK_OVERLAP,
});

async function createSource(client, title, sourceType = "pdf") {
  const { rows } = await client.query(
    `INSERT INTO knowledge_sources (title, source_type)
     VALUES ($1, $2)
     RETURNING id`,
    [title, sourceType]
  );
  return rows[0].id;
}

async function insertChunk(client, sourceId, chunk, vector) {
  await client.query(
    `INSERT INTO knowledge_chunks (source_id, content, embedding, topic)
     VALUES ($1, $2, $3::vector, $4)`,
    [
      sourceId,
      chunk.pageContent,
      `[${vector.join(",")}]`,
      chunk.metadata.topic ?? null,
    ]
  );
}

async function runTransaction(operation, context) {
  const client = await pool.connect();
  try {
    await client.query("BEGIN");
    const result = await operation(client);
    await client.query("COMMIT");
    return result;
  } catch (error) {
    try {
      await client.query("ROLLBACK");
    } catch (rollbackError) {
      console.error(`Rollback failed: ${rollbackError.message}`);
    }
    throw new Error(`Database failure ${context}: ${error.message}`, {
      cause: error,
    });
  } finally {
    client.release();
  }
}

async function parsePDF(filePath) {
  let buffer;
  try {
    buffer = fs.readFileSync(filePath);
  } catch (error) {
    throw new Error(`Could not read PDF "${filePath}": ${error.message}`, {
      cause: error,
    });
  }

  const parser = new PDFParse({ data: buffer });
  try {
    return await parser.getText();
  } catch (error) {
    throw new Error(`Could not parse PDF "${filePath}": ${error.message}`, {
      cause: error,
    });
  } finally {
    await parser.destroy();
  }
}

function assertUsableText(parsed) {
  const meaningfulText = parsed.text
    .replace(/^-- \d+ of \d+ --$/gm, "")
    .replace(/\s/g, "");

  if (meaningfulText.length < 100) {
    throw new Error(
      "PDF has no usable text layer. Run OCR on the PDF before ingesting it."
    );
  }
}

async function storeSource(fileName) {
  return runTransaction(
    (client) => createSource(client, fileName),
    `while creating source "${fileName}"`
  );
}

async function storeChunk(sourceId, chunk, chunkNumber, totalChunks, vector) {
  return runTransaction(
    (client) => insertChunk(client, sourceId, chunk, vector),
    `while storing chunk ${chunkNumber}/${totalChunks}`
  );
}

function buildChunkPreview(text) {
  return text.replace(/\s+/g, " ").trim().slice(0, PREVIEW_LENGTH);
}

function logSkippedChunk(chunkIndex, chunk, error) {
  console.error(`Chunk ${chunkIndex + 1} failed`);
  console.error(`Characters: ${chunk.pageContent.length}`);
  console.error("");
  console.error("Preview:");
  console.error(`"${buildChunkPreview(chunk.pageContent)}"`);
  console.error(`Reason: ${error.message}`);
}

async function processChunk(sourceId, chunk, chunkIndex, totalChunks) {
  const chunkNumber = chunkIndex + 1;
  console.log(`Processing chunk ${chunkNumber}/${totalChunks}...`);

  const vector = await embedChunkWithRetry(
    chunk.pageContent,
    chunkIndex
  );
  await storeChunk(sourceId, chunk, chunkNumber, totalChunks, vector);
  console.log(`Stored chunk ${chunkNumber}`);
}

function createRequestThrottler(delayMs) {
  let nextStartAt = Date.now();

  return async function throttleRequestStart() {
    const now = Date.now();
    const waitMs = Math.max(0, nextStartAt - now);
    nextStartAt = Math.max(now, nextStartAt) + delayMs;

    if (waitMs > 0) {
      await sleep(waitMs);
    }
  };
}

async function processChunkQueue(sourceId, chunks) {
  const totalChunks = chunks.length;
  const throttleRequestStart = createRequestThrottler(REQUEST_DELAY_MS);
  const stats = {
    completed: 0,
    skipped: 0,
  };
  let nextChunkIndex = 0;

  async function worker() {
    while (nextChunkIndex < totalChunks) {
      const chunkIndex = nextChunkIndex;
      nextChunkIndex += 1;
      const chunk = chunks[chunkIndex];

      try {
        await throttleRequestStart();
        await processChunk(sourceId, chunk, chunkIndex, totalChunks);
        stats.completed += 1;
      } catch (error) {
        stats.skipped += 1;
        logSkippedChunk(chunkIndex, chunk, error);
      }
    }
  }

  const workerCount = Math.min(CONCURRENCY, totalChunks);
  await Promise.all(Array.from({ length: workerCount }, () => worker()));

  console.log(`Completed ${stats.completed}/${totalChunks} chunks`);
  console.log(`Skipped: ${stats.skipped} chunks`);
  return stats;
}

export async function ingestPDF(filePath, topic = null) {
  const fileName = path.basename(filePath);
  console.log(`\nIngesting: ${fileName}`);

  const parsed = await parsePDF(filePath);
  console.log(`Pages: ${parsed.total} | Characters: ${parsed.text.length}`);
  assertUsableText(parsed);

  const doc = new Document({
    pageContent: parsed.text,
    metadata: { source: fileName, topic },
  });
  const chunks = await splitter.splitDocuments([doc]);
  console.log(`Chunks: ${chunks.length}`);

  const sourceId = await storeSource(fileName);
  console.log(`Source ID: ${sourceId}`);

  const stats = await processChunkQueue(sourceId, chunks);
  console.log(
    `Done: ${stats.completed} chunks stored for "${fileName}" (${stats.skipped} skipped)\n`
  );

  return sourceId;
}

const isCli = process.argv[1]
  ? path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)
  : false;

if (isCli) {
  const [filePath, topic] = process.argv.slice(2);
  if (!filePath) {
    console.error("Usage: node src/ingest.js <path-to-pdf> [topic]");
    process.exit(1);
  }

  ingestPDF(filePath, topic ?? null)
    .catch((error) => {
      console.error(`Ingestion failed: ${error.message}`);
      process.exitCode = 1;
    })
    .finally(() => pool.end());
}
