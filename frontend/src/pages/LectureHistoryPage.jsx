import { Search } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { listLectures } from "../api/lectures.js";
import VerificationBadge from "../components/lecture/VerificationBadge.jsx";
import Alert from "../components/ui/Alert.jsx";
import Loader from "../components/ui/Loader.jsx";

export default function LectureHistoryPage() {
  const navigate = useNavigate();
  const [lectures, setLectures] = useState([]);
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState("date");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    listLectures()
      .then(setLectures)
      .catch((err) => setError(err?.response?.data?.error || err?.message || "Lecture history could not be loaded."))
      .finally(() => setLoading(false));
  }, []);

  const filtered = useMemo(() => {
    const items = lectures.filter((lecture) => lecture.topic?.toLowerCase().includes(query.toLowerCase()));
    return items.sort((a, b) =>
      sort === "score"
        ? Number(b.verification_score) - Number(a.verification_score)
        : new Date(b.created_at) - new Date(a.created_at)
    );
  }, [lectures, query, sort]);

  if (loading) {
    return (
      <div className="brutal-panel p-6">
        <Loader label="Loading history" />
      </div>
    );
  }

  if (error) {
    return (
      <Alert title="History unavailable" tone="danger">
        {error}
      </Alert>
    );
  }

  return (
    <section className="brutal-panel p-5">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <p className="font-mono text-xs font-black uppercase">Lecture History</p>
          <h2 className="text-3xl font-black">Generated lectures</h2>
        </div>
        <div className="flex flex-col gap-3 sm:flex-row">
          <label className="relative block">
            <Search className="absolute left-3 top-3" size={16} />
            <input className="brutal-input pl-10" placeholder="Filter topic" value={query} onChange={(event) => setQuery(event.target.value)} />
          </label>
          <select className="brutal-input" value={sort} onChange={(event) => setSort(event.target.value)}>
            <option value="date">Sort by date</option>
            <option value="score">Sort by score</option>
          </select>
        </div>
      </div>

      {!filtered.length ? (
        <div className="mt-6 border-2 border-black bg-[#ffdf38] p-8 text-center font-mono font-black uppercase text-black">
          No lectures generated yet.
        </div>
      ) : (
        <div className="mt-6 overflow-x-auto">
          <table className="w-full min-w-[760px] border-collapse border-2 border-black bg-white dark:border-[#f8f4df] dark:bg-[#202020]">
            <thead>
              <tr className="bg-[#ffdf38] text-left text-black">
                <th className="border-2 border-black p-3 font-mono text-xs uppercase">Topic</th>
                <th className="border-2 border-black p-3 font-mono text-xs uppercase">Audience</th>
                <th className="border-2 border-black p-3 font-mono text-xs uppercase">Duration</th>
                <th className="border-2 border-black p-3 font-mono text-xs uppercase">Verification</th>
                <th className="border-2 border-black p-3 font-mono text-xs uppercase">Created</th>
                <th className="border-2 border-black p-3 font-mono text-xs uppercase">Open</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((lecture) => (
                <tr key={lecture.id}>
                  <td className="border-2 border-black p-3 font-semibold dark:border-[#f8f4df]">{lecture.topic}</td>
                  <td className="border-2 border-black p-3 font-semibold dark:border-[#f8f4df]">{lecture.audience}</td>
                  <td className="border-2 border-black p-3 font-semibold dark:border-[#f8f4df]">{lecture.duration} mins</td>
                  <td className="border-2 border-black p-3 dark:border-[#f8f4df]"><VerificationBadge score={lecture.verification_score} /></td>
                  <td className="border-2 border-black p-3 font-semibold dark:border-[#f8f4df]">{new Date(lecture.created_at).toLocaleString()}</td>
                  <td className="border-2 border-black p-3 dark:border-[#f8f4df]">
                    <button className="brutal-button" type="button" onClick={() => navigate(`/lectures/${lecture.id}`)}>
                      Open
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
