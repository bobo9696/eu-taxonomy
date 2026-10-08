"use client";

import { useState } from "react";
import type { Source } from "@/components/Chat";

type RagType = "classic" | "vectorless" | "sector_aware";
type PromptType = "basic" | "intermediate" | "expert";
type Scores = Record<string, number | null>;

interface Sample {
  question: string;
  answerable: boolean;
  answer: string;
  sources: Source[];
  scores: Scores;
}

interface EvaluateResponse {
  rag_type: RagType;
  prompt_type: PromptType;
  samples: Sample[];
  metrics: Scores;
}

const RAG_TYPES: RagType[] = ["classic", "vectorless", "sector_aware"];

const METRICS = [
  { key: "retrieval_hit", label: "Retrieval hit", hint: "Target FAQ entry was retrieved (deterministic)" },
  { key: "context_precision", label: "Context precision", hint: "Retrieved entries are relevant (ragas)" },
  { key: "context_recall", label: "Context recall", hint: "Reference answer is covered by the context (ragas)" },
  { key: "faithfulness", label: "Faithfulness", hint: "Answer is grounded in the context (ragas)" },
  { key: "answer_relevancy", label: "Answer relevancy", hint: "Answer addresses the question (ragas)" },
  { key: "answer_rate", label: "Answers answerable", hint: "Answered questions the FAQ covers instead of declining" },
  { key: "refusal_rate", label: "Detects unanswerable", hint: "Declined questions the FAQ does not cover" },
];

function Badge({ ok, children }: { ok: boolean; children: React.ReactNode }) {
  return (
    <span className={`text-xs px-2 py-1 rounded ${ok ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"}`}>
      {children}
    </span>
  );
}

function formatScore(value: number | null | undefined) {
  return value == null ? "—" : value.toFixed(3);
}

async function runEvaluation(ragType: RagType, promptType: PromptType): Promise<EvaluateResponse> {
  const res = await fetch(`/api/evaluate?rag_type=${ragType}&prompt_type=${promptType}`, { method: "POST" });
  if (!res.ok) throw new Error(`${ragType} evaluation failed (HTTP ${res.status})`);
  return res.json();
}

export default function Evaluation() {
  const [promptType, setPromptType] = useState<PromptType>("expert");
  const [results, setResults] = useState<EvaluateResponse[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleRun() {
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      setResults(await Promise.all(RAG_TYPES.map((ragType) => runEvaluation(ragType, promptType))));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Evaluation failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-5xl mx-auto w-full px-6 space-y-6">
      <div className="bg-white rounded-xl shadow p-6">
        <h2 className="text-xl font-semibold mb-2 text-eu-blue">Compare retrieval strategies</h2>
        <p className="text-sm text-gray-600 mb-4">
          Runs both pipelines on the golden set and scores them with ragas. It mixes questions the FAQ answers with questions it does not, which the pipeline must decline. Takes about 30 seconds.
        </p>
        <div className="flex flex-wrap items-center gap-4">
          <label className="text-sm font-medium text-gray-700">Prompt variant:</label>
          <select
            value={promptType}
            onChange={(e) => setPromptType(e.target.value as PromptType)}
            className="border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-eu-blue"
          >
            <option value="expert">Expert (default)</option>
            <option value="intermediate">Intermediate</option>
            <option value="basic">Basic</option>
          </select>
          <button
            onClick={handleRun}
            disabled={loading}
            className="ml-auto bg-eu-blue text-white font-medium px-6 py-2 rounded-lg hover:bg-blue-800 disabled:opacity-50 transition-colors"
          >
            {loading ? "Evaluating..." : "Run evaluation"}
          </button>
        </div>
        {error && <p className="mt-4 text-sm text-red-700">{error}</p>}
      </div>

      {results && (
        <>
          <div className="bg-white rounded-xl shadow p-6 overflow-x-auto">
            <h3 className="text-lg font-semibold text-eu-blue mb-3">Average scores</h3>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-gray-500 border-b">
                  <th className="py-2 pr-4 font-medium">Metric</th>
                  {results.map((r) => (
                    <th key={r.rag_type} className="py-2 px-4 font-medium capitalize text-right">{r.rag_type}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {METRICS.map(({ key, label, hint }) => {
                  const values = results.map((r) => r.metrics[key]);
                  const best = Math.max(...values.filter((v): v is number => v != null));
                  return (
                    <tr key={key} className="border-b last:border-0">
                      <td className="py-2 pr-4">
                        <p className="font-medium text-gray-800">{label}</p>
                        <p className="text-xs text-gray-500">{hint}</p>
                      </td>
                      {values.map((v, i) => (
                        <td
                          key={results[i].rag_type}
                          className={`py-2 px-4 text-right tabular-nums ${v === best && values.some((o) => o !== v) ? "font-bold text-eu-blue" : "text-gray-700"}`}
                        >
                          {formatScore(v)}
                        </td>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="bg-white rounded-xl shadow p-6">
            <h3 className="text-lg font-semibold text-eu-blue mb-3">Per question</h3>
            <div className="space-y-3">
              {results[0].samples.map((sample, i) => (
                <details key={sample.question} className="border rounded-lg">
                  <summary className="cursor-pointer p-3 flex flex-wrap items-center gap-3">
                    <span className="flex-1 font-medium text-gray-800">
                      {sample.question}
                      {!sample.answerable && (
                        <span className="ml-2 text-xs font-normal text-gray-500">(not in the FAQ)</span>
                      )}
                    </span>
                    {results.map(({ rag_type, samples }) => {
                      const { scores } = samples[i];
                      return sample.answerable ? (
                        <Badge key={rag_type} ok={!!scores.retrieval_hit && !!scores.answer_rate}>
                          {rag_type}: {!scores.answer_rate ? "declined" : scores.retrieval_hit ? "hit" : "miss"}
                        </Badge>
                      ) : (
                        <Badge key={rag_type} ok={!!scores.refusal_rate}>
                          {rag_type}: {scores.refusal_rate ? "declined" : "answered"}
                        </Badge>
                      );
                    })}
                  </summary>
                  <div className="grid md:grid-cols-2 gap-4 p-3 border-t">
                    {results.map((r) => {
                      const s = r.samples[i];
                      return (
                        <div key={r.rag_type} className="space-y-3">
                          <h4 className="text-sm font-semibold capitalize text-eu-blue">{r.rag_type}</h4>
                          <p className="text-sm text-gray-800 whitespace-pre-wrap">{s.answer}</p>
                          <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                            {METRICS.filter(({ key }) => s.scores[key] != null).map(({ key, label }) => (
                              <div key={key} className="contents">
                                <dt className="text-gray-500">{label}</dt>
                                <dd className="text-right tabular-nums">{formatScore(s.scores[key])}</dd>
                              </div>
                            ))}
                          </dl>
                          <ul className="space-y-2">
                            {s.sources.map((src) => (
                              <li key={src.chunk_id} className="bg-gray-50 rounded p-2">
                                <p className="text-xs font-bold text-gray-500 uppercase tracking-wide">{src.section}</p>
                                <p className="text-xs text-gray-800 mt-1">Q: {src.question}</p>
                              </li>
                            ))}
                          </ul>
                        </div>
                      );
                    })}
                  </div>
                </details>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
