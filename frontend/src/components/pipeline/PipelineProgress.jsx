import { Check, Circle, X } from "lucide-react";
import Loader from "../ui/Loader.jsx";

export default function PipelineProgress({
  stages,
  currentStageIndex,
  failedStageIndex,
  status,
}) {
  const percent =
    status === "complete"
      ? 100
      : Math.max(0, Math.round(((currentStageIndex + 1) / stages.length) * 100));

  return (
    <section className="brutal-panel mt-6 p-5">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <h2 className="text-xl font-black">Pipeline Progress</h2>
        <span className="brutal-badge">{percent}%</span>
      </div>
      <div className="mt-4 h-5 border-2 border-black bg-white dark:border-[#f8f4df] dark:bg-[#111111]">
        <div className="h-full bg-[#ff5f5f] transition-all dark:bg-[#7cf7c7]" style={{ width: `${percent}%` }} />
      </div>
      <div className="mt-5 grid gap-3">
        {stages.map((stage, index) => {
          const failed = failedStageIndex === index;
          const complete = status === "complete" || index < currentStageIndex;
          const active = status === "running" && index === currentStageIndex;
          const pending = !failed && !complete && !active;

          return (
            <div
              key={stage.label}
              className={`border-2 border-black p-4 dark:border-[#f8f4df] ${
                failed ? "bg-[#ff7a7a]" : active ? "bg-[#ffdf38]" : complete ? "bg-[#7cf7c7]" : "bg-white dark:bg-[#202020]"
              }`}
            >
              <div className="flex items-start gap-3">
                <span className="mt-1 text-black dark:text-[#f8f4df]">
                  {active ? <Loader label="" /> : complete ? <Check size={18} /> : failed ? <X size={18} /> : pending ? <Circle size={18} /> : null}
                </span>
                <div>
                  <p className="font-mono text-sm font-black uppercase text-black dark:text-[#f8f4df]">{stage.label}</p>
                  <p className="mt-1 text-sm font-semibold text-black dark:text-[#f8f4df]">{stage.description}</p>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
