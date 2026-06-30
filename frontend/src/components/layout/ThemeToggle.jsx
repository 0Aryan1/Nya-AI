import { Moon, Sun } from "lucide-react";
import { useEffect, useState } from "react";

const getInitialTheme = () =>
  localStorage.getItem("nyaai-theme") ||
  (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");

export default function ThemeToggle() {
  const [theme, setTheme] = useState(getInitialTheme);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    localStorage.setItem("nyaai-theme", theme);
  }, [theme]);

  const isDark = theme === "dark";

  return (
    <button
      type="button"
      aria-label="Toggle theme"
      className="brutal-button-secondary h-11 w-11 px-0"
      onClick={() => setTheme(isDark ? "light" : "dark")}
    >
      {isDark ? <Sun size={18} /> : <Moon size={18} />}
    </button>
  );
}
