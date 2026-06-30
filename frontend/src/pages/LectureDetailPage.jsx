import { RefreshCw } from "lucide-react";
import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { getLecture } from "../api/lectures.js";
import LectureView from "../components/lecture/LectureView.jsx";
import Alert from "../components/ui/Alert.jsx";
import Loader from "../components/ui/Loader.jsx";

export default function LectureDetailPage() {
  const { id } = useParams();
  const [lecture, setLecture] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const loadLecture = async () => {
    setLoading(true);
    setError("");
    try {
      setLecture(await getLecture(id));
    } catch (err) {
      setError(err?.response?.data?.error || err?.message || "Lecture could not be loaded.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadLecture();
  }, [id]);

  if (loading) {
    return (
      <div className="brutal-panel p-6">
        <Loader label="Loading lecture" />
      </div>
    );
  }

  if (error) {
    return (
      <Alert
        title="Lecture not available"
        tone="danger"
        action={
          <button className="brutal-button-secondary" onClick={loadLecture} type="button">
            <RefreshCw size={16} /> Retry
          </button>
        }
      >
        {error}
      </Alert>
    );
  }

  return <LectureView lecture={lecture} />;
}
