import { useState } from "react";

export default function CitationsPanel({ citations = [] }) {
  const [openIndex, setOpenIndex] = useState(null);

  if (!citations.length) {
    return <p className="text-sm font-semibold">No citations were returned for this lecture.</p>;
  }

  return (
    <div className="grid gap-3">
      {citations.map((citation, index) => (
        <article key={`${citation.source_title}-${index}`} className="border-2 border-black bg-white p-4 dark:border-[#f8f4df] dark:bg-[#202020]">
          <div className="flex flex-wrap items-center gap-2">
            <h4 className="font-black">{citation.source_title || "Untitled source"}</h4>
            {citation.source_type ? <span className="brutal-badge">{citation.source_type}</span> : null}
            {citation.chunk_topic ? <span className="brutal-badge">{citation.chunk_topic}</span> : null}
          </div>
          <p className="mt-3 text-sm font-semibold">{citation.citation_text}</p>
          <button className="brutal-button-secondary mt-3" type="button" onClick={() => setOpenIndex(openIndex === index ? null : index)}>
            {openIndex === index ? "Hide full chunk" : "View full chunk"}
          </button>
          {openIndex === index ? <pre className="mt-3 whitespace-pre-wrap border-2 border-black bg-[#f7f1dc] p-3 text-sm dark:border-[#f8f4df] dark:bg-[#111111]">{citation.chunk_content}</pre> : null}
        </article>
      ))}
    </div>
  );
}
