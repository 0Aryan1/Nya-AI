import { Navigate, Route, Routes } from "react-router-dom";
import Navbar from "./components/layout/Navbar.jsx";
import GeneratePage from "./pages/GeneratePage.jsx";
import LectureDetailPage from "./pages/LectureDetailPage.jsx";
import LectureHistoryPage from "./pages/LectureHistoryPage.jsx";

export default function App() {
  return (
    <div className="min-h-screen bg-[#f7f1dc] text-black dark:bg-[#161616] dark:text-[#f8f4df]">
      <Navbar />
      <main className="mx-auto w-full max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        <Routes>
          <Route path="/" element={<GeneratePage />} />
          <Route path="/generate" element={<GeneratePage />} />
          <Route path="/lectures" element={<LectureHistoryPage />} />
          <Route path="/lectures/:id" element={<LectureDetailPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
