import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { generateLecture } from "../api/lectures.js";
import LectureForm from "../components/lecture/LectureForm.jsx";
import PipelineProgress from "../components/pipeline/PipelineProgress.jsx";
import usePipelineProgress from "../components/pipeline/usePipelineProgress.js";
import Alert from "../components/ui/Alert.jsx";

const errorMessage = (error) =>
  error?.response?.data?.error ||
  error?.message ||
  "The lecture request failed. Check that the backend is running.";

export default function GeneratePage() {
  const navigate = useNavigate();
  const pipeline = usePipelineProgress();
  const [isSubmitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  const onSubmit = async (values) => {
    setSubmitting(true);
    setError("");
    pipeline.start();

    try {
      const result = await generateLecture({
        ...values,
        topic_filter: values.topic_filter?.trim() || undefined,
      });
      pipeline.complete();
      setTimeout(() => navigate(`/lectures/${result.lectureId}`), 450);
    } catch (err) {
      pipeline.fail(pipeline.currentStageIndex);
      setError(errorMessage(err));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="grid gap-6">
      {error ? (
        <Alert title="Generation failed" tone="danger">
          <p>{error}</p>
          {String(error).includes("No relevant chunks found") ? (
            <p className="mt-2">An admin needs to ingest more PDFs through the backend CLI. This frontend intentionally has no PDF upload flow.</p>
          ) : null}
        </Alert>
      ) : null}
      <LectureForm onSubmit={onSubmit} isSubmitting={isSubmitting} />
      {pipeline.status !== "idle" ? <PipelineProgress {...pipeline} /> : null}
    </div>
  );
}
