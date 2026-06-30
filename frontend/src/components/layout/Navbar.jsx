import { NavLink } from "react-router-dom";
import ThemeToggle from "./ThemeToggle.jsx";

const navClass = ({ isActive }) =>
  `border-2 border-black px-3 py-2 font-mono text-xs font-black uppercase shadow-brutal dark:border-[#f8f4df] dark:shadow-brutal-dark ${
    isActive
      ? "bg-[#ff5f5f] text-black dark:bg-[#7cf7c7]"
      : "bg-white text-black dark:bg-[#202020] dark:text-[#f8f4df]"
  }`;

export default function Navbar() {
  return (
    <header className="border-b-2 border-black bg-[#7cf7c7] dark:border-[#f8f4df] dark:bg-[#111111]">
      <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-4 sm:flex-row sm:items-center sm:justify-between sm:px-6 lg:px-8">
        <div>
          <p className="font-mono text-xs font-black uppercase">NYAAI</p>
          <h1 className="text-2xl font-black leading-tight">AI Lecture Prep</h1>
        </div>
        <nav className="flex flex-wrap items-center gap-3">
          <NavLink to="/generate" className={navClass}>
            Generate Lecture
          </NavLink>
          <NavLink to="/lectures" className={navClass}>
            Lecture History
          </NavLink>
          <ThemeToggle />
        </nav>
      </div>
    </header>
  );
}
