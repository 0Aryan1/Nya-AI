/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["Space Mono", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      boxShadow: {
        brutal: "6px 6px 0 #111111",
        "brutal-dark": "6px 6px 0 #f8f4df",
      },
    },
  },
  plugins: [],
};
