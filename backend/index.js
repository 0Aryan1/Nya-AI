import * as dotenv from 'dotenv';
dotenv.config();

import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf"
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { PineconeStore } from "@langchain/pinecone";
import { Pinecone as PineconeClient } from "@pinecone-database/pinecone";



async function indexDocument(){

// Load PDF
const PDF_PATH = './dsa.pdf';
const pdfLoader = new PDFLoader(PDF_PATH);
const rawDocs = await pdfLoader.load();
console.log("PDF Loaded")

// Chunking
const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
const chunkedDocs = await textSplitter.splitDocuments(rawDocs);
console.log("PDF Chunked")

//Vector Embeding
const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GEMINI_API_KEY,
  modelName: "gemini-embedding-001",
});
console.log("Embedding Model Configured")

//Database Config
//Inittialize Pinecone client
const pinecone = new PineconeClient();
const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);
console.log("Pinecone Client Initialized")

//Langchain(chunking + embeddings + DB)
await PineconeStore.fromDocuments(chunkedDocs, embeddings, {
    pineconeIndex,
    maxConcurrency: 5,
  });
console.log("Documents Indexed in Pinecone")


}

indexDocument()