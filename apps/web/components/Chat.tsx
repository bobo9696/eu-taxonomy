"use client";

import { useState } from "react";

export interface Source {
  section: string;
  question: string;
  chunk_id: number;
}

interface AskResponse {
  answer: string;
  sources: Source[];
  contexts: string[];
  retrieval_ms: number;
  generation_ms: number;
  total_ms: number;
}

interface ScoreResponse {
  faithfulness: number | null;
  answer_relevancy: number | null;
}

function ScoreBadge({ label, value }: { label: string; value: number | null }) {
  if (value == null) return null;
  let color = "bg-red-100 text-red-800";
  if (value >= 0.8) color = "bg-green-100 text-green-800";
  else if (value >= 0.5) color = "bg-yellow-100 text-yellow-800";
  const percent = (value * 100).toFixed(1); // 91.2%
  const decimal = value.toFixed(3);         // 0.912
  return (
    <span
      className={`text-xs font-mono font-medium px-2.5 py-1 rounded-full ${color}`}
      title={`${label}: ${percent}% (raw = ${decimal})`}
    >
      {label}: {percent}% ({decimal})
    </span>
  );
}

function MetricRow({ label, value, unit = "" }: { label: string; value: number | null | string; unit?: string }) {
  const display = value == null ? "—" : typeof value === "number" ? `${value.toFixed(1)}${unit}` : `${value}${unit}`;
  return (
    <tr className="border-b last:border-0">
      <td className="py-1.5 pr-4 text-sm text-gray-600">{label}</td>
      <td className="py-1.5 text-right text-sm font-mono text-gray-800 tabular-nums">{display}</td>
    </tr>
  );
}

export default function Chat() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<string | null>(null);
  const [sources, setSources] = useState<Source[]>([]);
  const [contexts, setContexts] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [promptType, setPromptType] = useState<"basic" | "intermediate" | "expert">("expert");
  const [ragType, setRagType] = useState<"classic" | "vectorless" | "sector_aware">("classic");

  const [retrievalMs, setRetrievalMs] = useState<number | null>(null);
  const [generationMs, setGenerationMs] = useState<number | null>(null);
  const [totalMs, setTotalMs] = useState<number | null>(null);

  const [scores, setScores] = useState<ScoreResponse | null>(null);
  const [scoring, setScoring] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!question.trim()) return;
    setLoading(true);
    setAnswer(null);
    setSources([]);
    setContexts([]);
    setScores(null);
    setRetrievalMs(null);
    setGenerationMs(null);
    setTotalMs(null);

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, prompt_type: promptType, rag_type: ragType }),
      });
      const data: AskResponse = await res.json();
      setAnswer(data.answer);
      setSources(data.sources);
      setContexts(data.contexts || []);
      setRetrievalMs(data.retrieval_ms ?? null);
      setGenerationMs(data.generation_ms ?? null);
      setTotalMs(data.total_ms ?? null);

      // Run live scoring asynchronously so the UI doesn't block
      if (data.contexts && data.contexts.length > 0) {
        setScoring(true);
        fetch("/api/score", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question, answer: data.answer, contexts: data.contexts }),
        })
          .then((r) => r.json())
          .then((scoreData: ScoreResponse) => setScores(scoreData))
          .catch(() => setScores(null))
          .finally(() => setScoring(false));
      }
    } catch (err) {
      setAnswer("An error occurred while fetching the answer.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-3xl mx-auto w-full p-6">
      <div className="bg-white rounded-xl shadow p-6 mb-6">
        <h2 className="text-xl font-semibold mb-4 text-eu-blue">Ask a question</h2>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Question
            </label>
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="e.g., What does DNSH mean in the EU Taxonomy?"
              className="w-full border border-gray-300 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-eu-blue focus:border-transparent"
            />
          </div>

          <div className="flex flex-wrap items-center gap-4">
            <label className="text-sm font-medium text-gray-700">Retrieval:</label>
            <select
              value={ragType}
              onChange={(e) => setRagType(e.target.value as "classic" | "vectorless" | "sector_aware")}
              className="border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-eu-blue"
            >
              <option value="classic">Classic (vector search)</option>
              <option value="vectorless">Vectorless (LLM navigation)</option>
              <option value="sector_aware">Sector-aware (sector filter)</option>
            </select>
            <label className="text-sm font-medium text-gray-700">Prompt variant:</label>
            <select
              value={promptType}
              onChange={(e) => setPromptType(e.target.value as "basic" | "intermediate" | "expert")}
              className="border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-eu-blue"
            >
              <option value="expert">Expert (CoT, 6-element)</option>
              <option value="intermediate">Intermediate (rules)</option>
              <option value="basic">Basic (minimal)</option>
            </select>
            <button
              type="submit"
              disabled={loading}
              className="ml-auto bg-eu-blue text-white font-medium px-6 py-2 rounded-lg hover:bg-blue-800 disabled:opacity-50 transition-colors"
            >
              {loading ? "Thinking..." : "Ask"}
            </button>
          </div>
        </form>
      </div>

      {answer && (
        <div className="bg-white rounded-xl shadow p-6 space-y-5">
          <div>
            <h3 className="text-lg font-semibold text-eu-blue mb-3">Answer</h3>
            <div className="prose max-w-none text-gray-800 whitespace-pre-wrap">
              {answer}
            </div>
          </div>

          {/* Timings + Scores Table */}
          <div className="border-t pt-4">
            <h4 className="text-sm font-semibold text-gray-600 mb-2">
              Evaluation ({ragType.replace("_", " ")} · {promptType})
            </h4>
            <div className="overflow-x-auto">
              <table className="w-full text-sm border-collapse">
                <tbody>
                  <MetricRow label="Retrieval time" value={retrievalMs} unit=" ms" />
                  <MetricRow label="Generation time" value={generationMs} unit=" ms" />
                  <MetricRow label="Total time" value={totalMs} unit=" ms" />
                  {scoring ? (
                    <tr>
                      <td colSpan={2} className="py-2 text-xs text-gray-500 animate-pulse">
                        Scoring with ragas…
                      </td>
                    </tr>
                  ) : (
                    <>
                      <MetricRow label="Faithfulness" value={scores?.faithfulness ?? null} />
                      <MetricRow label="Answer relevancy" value={scores?.answer_relevancy ?? null} />
                    </>
                  )}
                </tbody>
              </table>
            </div>
            <div className="flex flex-wrap items-center gap-2 mt-3">
              <ScoreBadge label="Faithfulness" value={scores?.faithfulness ?? null} />
              <ScoreBadge label="Answer relevancy" value={scores?.answer_relevancy ?? null} />
            </div>
            <p className="text-xs text-gray-400 mt-2">
              Faithfulness = is the answer grounded in the retrieved context? <br />
              Answer relevancy = does the answer address the question?
            </p>
          </div>

          {sources.length > 0 && (
            <div className="border-t pt-4">
              <h4 className="text-sm font-semibold text-gray-600 mb-2">
                Sources ({sources.length})
              </h4>
              <ul className="space-y-3">
                {sources.map((s, i) => (
                  <li key={i} className="bg-gray-50 rounded-lg p-3">
                    <p className="text-xs font-bold text-gray-500 uppercase tracking-wide">
                      {s.section}
                    </p>
                    <p className="text-sm font-medium text-gray-800 mt-1">
                      Q: {s.question}
                    </p>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
