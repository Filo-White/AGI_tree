/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      animation: {
        "pulse-slow": "pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "glow": "glow 1.5s ease-in-out infinite alternate",
      },
      keyframes: {
        glow: {
          "0%": { boxShadow: "0 0 5px var(--glow-color, #10b981)" },
          "100%": { boxShadow: "0 0 20px var(--glow-color, #10b981)" },
        },
      },
    },
  },
  plugins: [],
};
