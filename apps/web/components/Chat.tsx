"use client";

import { useState } from "react";

export default function Chat() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!question.trim() || loading) return;
    setLoading(true);
    setAnswer(null);

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      if (!res.ok) throw new Error(`Request failed with status ${res.status}`);
      const data: { answer: string } = await res.json();
      setAnswer(data.answer);
    } catch {
      setAnswer("An error occurred while fetching the answer.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-3xl mx-auto w-full p-6 space-y-6">
      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          disabled={loading}
          placeholder="Ask a question about the EU Taxonomy and press Enter"
          aria-label="Question"
          className="flex-1 border border-gray-300 rounded-lg px-4 py-3 bg-white focus:outline-none focus:ring-2 focus:ring-eu-blue focus:border-transparent disabled:opacity-60"
        />
        <button
          type="submit"
          disabled={loading || !question.trim()}
          className="px-5 py-3 rounded-lg bg-eu-blue text-white font-medium disabled:opacity-60"
        >
          Search
        </button>
      </form>

      {(loading || answer) && (
        <div className="bg-white rounded-lg border border-gray-200 px-4 py-3 text-gray-800 whitespace-pre-wrap">
          {loading ? <span className="text-gray-500 animate-pulse">Thinking…</span> : answer}
        </div>
      )}
    </div>
  );
}
