import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        eu: {
          blue: "#003399",
          yellow: "#FFCC00",
          light: "#F5F7FA",
        },
      },
    },
  },
  plugins: [],
};

export default config;
