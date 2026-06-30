import { useCallback, useEffect, useRef, useState } from "react";

/*
SIMULATED PROGRESS - replace timer-based advancement with real SSE/WebSocket
stage events from backend if the backend adds a streaming endpoint in future.
Stage definitions and UI should not need to change, only the trigger mechanism.
*/
export const pipelineStages = [
  {
    label: "Saving request",
    description: "Preparing the lecture request for the backend.",
  },
  {
    label: "Retrieving knowledge base chunks",
    description: "Searching chunks via Gemini embedding and pgvector cosine similarity.",
  },
  {
    label: "Generating lecture with GLM",
    description: "Calling Z.ai GLM with the retrieved academic context.",
  },
  {
    label: "Computing verification score",
    description: "Checking cited chunks against the generated lecture sections.",
  },
  {
    label: "Saving lecture & citations",
    description: "Persisting the lecture and citation snippets.",
  },
];

export default function usePipelineProgress() {
  const timerRef = useRef(null);
  const [currentStageIndex, setCurrentStageIndex] = useState(-1);
  const [status, setStatus] = useState("idle");
  const [failedStageIndex, setFailedStageIndex] = useState(null);

  const clearTimer = useCallback(() => {
    if (timerRef.current) window.clearTimeout(timerRef.current);
    timerRef.current = null;
  }, []);

  const scheduleNext = useCallback(() => {
    clearTimer();
    timerRef.current = window.setTimeout(() => {
      setCurrentStageIndex((index) => {
        if (index >= pipelineStages.length - 2) return index;
        scheduleNext();
        return index + 1;
      });
    }, 1500 + Math.round(Math.random() * 1500));
  }, [clearTimer]);

  const start = useCallback(() => {
    setStatus("running");
    setFailedStageIndex(null);
    setCurrentStageIndex(0);
    scheduleNext();
  }, [scheduleNext]);

  const complete = useCallback(() => {
    clearTimer();
    setCurrentStageIndex(pipelineStages.length - 1);
    setStatus("complete");
  }, [clearTimer]);

  const fail = useCallback(
    (stageIndex) => {
      clearTimer();
      setFailedStageIndex(Math.max(stageIndex, 0));
      setStatus("failed");
    },
    [clearTimer]
  );

  useEffect(() => clearTimer, [clearTimer]);

  return {
    stages: pipelineStages,
    currentStageIndex,
    failedStageIndex,
    status,
    start,
    complete,
    fail,
  };
}
