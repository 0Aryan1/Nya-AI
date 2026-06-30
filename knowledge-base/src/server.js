import express from "express";
import { lectureRouter } from "./lecture/routes.js";
import "dotenv/config";

const app = express();

const defaultAllowedOrigins = [
  "https://nya-ai.vercel.app",
  "http://localhost:5173",
  "http://127.0.0.1:5173",
];

const allowedOrigins = new Set(
  (process.env.CORS_ORIGINS || defaultAllowedOrigins.join(","))
    .split(",")
    .map((origin) => origin.trim().replace(/\/$/, ""))
    .filter(Boolean),
);

app.use((req, res, next) => {
  const origin = req.get("Origin");

  // Requests without an Origin header are not browser cross-origin requests.
  if (!origin) return next();

  if (!allowedOrigins.has(origin)) {
    if (req.method === "OPTIONS") {
      return res.status(403).json({ error: "Origin is not allowed by CORS" });
    }
    return next();
  }

  res.setHeader("Access-Control-Allow-Origin", origin);
  res.setHeader("Vary", "Origin");
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
  res.setHeader("Access-Control-Max-Age", "86400");

  if (req.method === "OPTIONS") return res.sendStatus(204);
  return next();
});

app.use(express.json());

app.use("/api/lectures", lectureRouter);

// Health check
app.get("/health", (_, res) => res.json({ status: "ok" }));

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));
