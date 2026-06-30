import express from "express";
import { lectureRouter } from "./lecture/routes.js";
import "dotenv/config";

const app = express();
app.use(express.json());

app.use("/api/lectures", lectureRouter);

// Health check
app.get("/health", (_, res) => res.json({ status: "ok" }));

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));