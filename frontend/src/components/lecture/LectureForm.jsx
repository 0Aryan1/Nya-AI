import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { z } from "zod";

export const GenerateSchema = z.object({
  topic: z.string().min(3, "Topic must be at least 3 characters"),
  audience: z.string().min(2, "Audience must be at least 2 characters"),
  duration: z.coerce.number().int().min(5, "Minimum 5 minutes").max(180, "Maximum 180 minutes"),
  learning_objective: z.string().min(10, "Learning objective must be at least 10 characters"),
  topic_filter: z.string().optional(),
});

function FieldError({ message }) {
  return message ? <p className="mt-2 font-mono text-xs font-black uppercase text-[#b00020] dark:text-[#ff9c9c]">{message}</p> : null;
}

export default function LectureForm({ onSubmit, isSubmitting }) {
  const {
    register,
    handleSubmit,
    formState: { errors, isValid },
  } = useForm({
    resolver: zodResolver(GenerateSchema),
    mode: "onChange",
    defaultValues: {
      topic: "",
      audience: "",
      duration: 45,
      learning_objective: "",
      topic_filter: "",
    },
  });

  return (
    <form className="brutal-panel p-5" onSubmit={handleSubmit(onSubmit)}>
      <div className="mb-5">
        <p className="font-mono text-xs font-black uppercase">Generate Lecture</p>
        <h2 className="text-3xl font-black leading-tight">Build from the knowledge base</h2>
      </div>
      <div className="grid gap-5 md:grid-cols-2">
        <label className="block">
          <span className="font-mono text-sm font-black uppercase">Topic</span>
          <input className="brutal-input mt-2" {...register("topic")} aria-invalid={Boolean(errors.topic)} />
          <FieldError message={errors.topic?.message} />
        </label>
        <label className="block">
          <span className="font-mono text-sm font-black uppercase">Audience</span>
          <input className="brutal-input mt-2" {...register("audience")} aria-invalid={Boolean(errors.audience)} />
          <FieldError message={errors.audience?.message} />
        </label>
        <label className="block">
          <span className="font-mono text-sm font-black uppercase">Duration</span>
          <input type="number" className="brutal-input mt-2" {...register("duration")} aria-invalid={Boolean(errors.duration)} />
          <FieldError message={errors.duration?.message} />
        </label>
        <label className="block">
          <span className="font-mono text-sm font-black uppercase">Topic Filter</span>
          <input className="brutal-input mt-2" {...register("topic_filter")} />
          <p className="mt-2 text-xs font-semibold">Optional - filters knowledge base by topic tag. Leave blank to search all ingested sources.</p>
        </label>
        <label className="block md:col-span-2">
          <span className="font-mono text-sm font-black uppercase">Learning Objective</span>
          <textarea rows={5} className="brutal-input mt-2" {...register("learning_objective")} aria-invalid={Boolean(errors.learning_objective)} />
          <FieldError message={errors.learning_objective?.message} />
        </label>
      </div>
      <button type="submit" className="brutal-button mt-6" disabled={!isValid || isSubmitting}>
        {isSubmitting ? "Generating..." : "Generate Lecture"}
      </button>
    </form>
  );
}
