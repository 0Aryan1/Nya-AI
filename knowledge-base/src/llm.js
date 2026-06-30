import { ChatOpenAI } from "@langchain/openai";
import "dotenv/config";

export const llm = new ChatOpenAI({
  model: "glm-4.7-flash",
  apiKey: process.env.ZAI_API_KEY,
  configuration: {
    baseURL: "https://api.z.ai/api/paas/v4",
  },
  temperature: 0.7,
});
