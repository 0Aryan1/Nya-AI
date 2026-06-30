import CitationsPanel from "./CitationsPanel.jsx";
import VerificationBadge from "./VerificationBadge.jsx";
import Alert from "../ui/Alert.jsx";

export default function LectureView({ lecture }) {
  const { content } = lecture;

  return (
    <div className="grid gap-6">
      <section className="brutal-panel p-5">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <h2 className="text-4xl font-black leading-tight">{content.title}</h2>
            <div className="mt-4 flex flex-wrap gap-2">
              {lecture.topic ? <span className="brutal-badge">{lecture.topic}</span> : null}
              {lecture.audience ? <span className="brutal-badge">{lecture.audience}</span> : null}
              {lecture.duration ? <span className="brutal-badge">{lecture.duration} mins</span> : null}
              {lecture.createdAt ? <span className="brutal-badge">{new Date(lecture.createdAt).toLocaleString()}</span> : null}
            </div>
          </div>
          <VerificationBadge score={lecture.verificationScore} />
        </div>
      </section>

      <section className="brutal-panel p-5">
        <h3 className="text-2xl font-black">Summary</h3>
        <p className="mt-3 whitespace-pre-wrap font-semibold leading-7">{content.summary}</p>
      </section>

      <section className="grid gap-4">
        {content.sections.map((section, index) => (
          <article key={`${section.heading}-${index}`} className="brutal-panel p-5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <h3 className="text-2xl font-black">{section.heading || `Section ${index + 1}`}</h3>
              <span className="brutal-badge">Sources {(section.cited_chunk_ids ?? []).length}</span>
            </div>
            <p className="mt-4 whitespace-pre-wrap font-semibold leading-7">{section.content}</p>
            {section.key_points?.length ? (
              <ul className="mt-4 list-disc space-y-2 pl-6 font-semibold">
                {section.key_points.map((point) => (
                  <li key={point}>{point}</li>
                ))}
              </ul>
            ) : null}
          </article>
        ))}
      </section>

      <section className="brutal-panel p-5">
        <h3 className="text-2xl font-black">Conclusion</h3>
        <p className="mt-3 whitespace-pre-wrap font-semibold leading-7">{content.conclusion}</p>
      </section>

      <section className="brutal-panel p-5">
        <h3 className="text-2xl font-black">Suggested Questions</h3>
        <div className="mt-3 flex flex-wrap gap-2">
          {content.suggested_questions.map((question) => (
            <span key={question} className="brutal-badge normal-case">
              {question}
            </span>
          ))}
        </div>
      </section>

      {content.verification_notes ? (
        <Alert title="Verification Notes" tone="warning">
          <p className="whitespace-pre-wrap">{content.verification_notes}</p>
        </Alert>
      ) : null}

      <section className="brutal-panel p-5">
        <h3 className="mb-4 text-2xl font-black">Citations</h3>
        <CitationsPanel citations={lecture.citations} />
      </section>
    </div>
  );
}
