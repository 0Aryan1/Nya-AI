import api from "./client.js";

const normalizeContent = (lecture = {}) => {
  const content = lecture.lecture_content ?? lecture.lecture ?? lecture;
  return {
    title: content.title ?? lecture.title ?? "Generated Lecture",
    summary: content.summary ?? "",
    sections: Array.isArray(content.sections) ? content.sections : [],
    conclusion: content.conclusion ?? "",
    suggested_questions: Array.isArray(content.suggested_questions)
      ? content.suggested_questions
      : [],
    verification_notes: content.verification_notes ?? "",
  };
};

export const normalizeLectureDetail = (payload) => {
  const lecture = payload.lecture ?? payload;
  return {
    id: lecture.id ?? payload.lectureId,
    requestId: lecture.request_id ?? payload.requestId,
    topic: lecture.topic,
    audience: lecture.audience,
    duration: lecture.duration,
    learningObjective: lecture.learning_objective,
    verificationScore: Number(lecture.verification_score ?? payload.verificationScore ?? 0),
    createdAt: lecture.created_at,
    content: normalizeContent(lecture),
    citations: payload.citations ?? [],
    chunksUsed: payload.chunksUsed,
  };
};

/** @param {{topic:string,audience:string,duration:number,learning_objective:string,topic_filter?:string}} payload */
export const generateLecture = (payload) =>
  api.post("/api/lectures/generate", payload).then((r) => r.data);

/** @returns {Promise<Array>} */
export const listLectures = () => api.get("/api/lectures").then((r) => r.data);

/** @param {string} id */
export const getLecture = (id) =>
  api.get(`/api/lectures/${id}`).then((r) => normalizeLectureDetail(r.data));
