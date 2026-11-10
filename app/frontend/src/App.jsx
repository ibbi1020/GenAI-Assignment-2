import { useEffect, useState } from "react";
import { api } from "./api.js";

const DISCLAIMER =
  "This is an educational screening prototype. It is not a clinical enrollment decision and must not be used to enroll or exclude a patient from a trial.";

let nextId = 1;

function emptyCriterion() {
  nextId += 1;
  return { id: nextId, criterion_type: "inclusion", criterion_text: "" };
}

export default function App() {
  const [page, setPage] = useState("screen");
  const [health, setHealth] = useState(null);
  useEffect(() => {
    let stop = false;
    const tick = () => {
      api("/api/health")
        .then((payload) => {
          if (!stop) setHealth(payload);
        })
        .catch(() => {
          if (!stop) setHealth(null);
        });
    };
    tick();
    const timer = setInterval(tick, 4000);
    return () => {
      stop = true;
      clearInterval(timer);
    };
  }, []);

  return (
    <div className="page">
      <header className="top">
        <div>
          <p className="eyebrow">Generative AI assignment 2</p>
          <h1>Criterion screening</h1>
        </div>
        <nav>
          <button className={page === "screen" ? "on" : ""} onClick={() => setPage("screen")} type="button">
            Screening
          </button>
          <button className={page === "lab" ? "on" : ""} onClick={() => setPage("lab")} type="button">
            Inference lab
          </button>
        </nav>
      </header>
      <p className="notice">{DISCLAIMER}</p>
      <p className="status">
        {health == null
          ? "The API is not running."
          : health.model_loaded
            ? `Model loaded${health.adapter_ready ? " with the fine-tuned adapter" : " (base model; adapter not found)"}.`
            : health.loading
              ? "The model is loading."
              : health.error || "The model is not loaded yet."}
      </p>
      {page === "screen" ? <Screening /> : <Lab />}
    </div>
  );
}

function Screening() {
  const [note, setNote] = useState("");
  const [criteria, setCriteria] = useState([emptyCriterion()]);
  const [examples, setExamples] = useState([]);
  const [cards, setCards] = useState([]);
  const [overall, setOverall] = useState(null);
  const [numbered, setNumbered] = useState("");
  const [formError, setFormError] = useState("");
  const [running, setRunning] = useState(false);

  useEffect(() => {
    api("/api/examples")
      .then((payload) => setExamples(payload.examples || []))
      .catch(() => setExamples([]));
  }, []);

  function updateCriterion(id, patch) {
    setCriteria((rows) => rows.map((row) => (row.id === id ? { ...row, ...patch } : row)));
  }

  async function runAll() {
    setFormError("");
    if (!note.trim()) {
      setFormError("The patient note is empty. Paste a note, or load an example.");
      return;
    }
    if (criteria.some((row) => !row.criterion_text.trim())) {
      setFormError("Every criterion needs text. Remove the empty row or fill it in.");
      return;
    }
    setRunning(true);
    const previousCards = cards;
    const previousOverall = overall;
    const previousNote = numbered;
    const nextCards = [];
    let shownNote = numbered;
    let anyOk = false;
    for (const row of criteria) {
      try {
        const result = await api("/api/screen/criterion", {
          note,
          criterion_type: row.criterion_type,
          criterion_text: row.criterion_text,
        });
        shownNote = result.numbered_note;
        anyOk = true;
        setFormError("");
        nextCards.push({ ...result, localId: row.id });
        setNumbered(result.numbered_note);
        setCards([...nextCards]);
      } catch (error) {
        nextCards.push({
          ok: false,
          localId: row.id,
          criterion_type: row.criterion_type,
          criterion_text: row.criterion_text,
          error: error.message,
        });
        if (anyOk) {
          setCards([...nextCards]);
        } else {
          setFormError(error.message);
          setCards(previousCards);
          setOverall(previousOverall);
          setNumbered(previousNote);
        }
      }
    }
    if (!anyOk) {
      setRunning(false);
      return;
    }
    try {
      const summary = await api("/api/aggregate", { criteria: nextCards });
      setOverall(summary);
    } catch (error) {
      setOverall(previousOverall || { outcome: "not scored", rule: error.message, is_ground_truth: false });
      setFormError(error.message);
    }
    setNumbered(shownNote);
    setRunning(false);
  }

  async function downloadReport() {
    const response = await fetch("/api/report", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ numbered_note: numbered || note, criteria: cards, aggregate: overall || {} }),
    });
    if (!response.ok) {
      setFormError("The report could not be built.");
      return;
    }
    const html = await response.text();
    const blob = new Blob([html], { type: "text/html" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "eligibility-screening-report.html";
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <main className="grid">
      <section className="panel">
        <h2>Patient note</h2>
        <label>
          Train-split example
          <select
            defaultValue=""
            onChange={(event) => {
              const example = examples[Number(event.target.value)];
              if (!example) return;
              setNote(example.note);
              setCriteria(
                example.criteria.map((row) => ({ ...emptyCriterion(), ...row })),
              );
              setCards([]);
              setOverall(null);
            }}
          >
            <option value="">Paste your own note</option>
            {examples.map((example, index) => (
              <option key={example.trial_id + example.patient_id} value={index}>
                {example.patient_id} · {example.trial_id}
              </option>
            ))}
          </select>
        </label>
        <textarea
          value={note}
          onChange={(event) => setNote(event.target.value)}
          rows={12}
          placeholder={"0. A 58-year-old woman presents with chest pain.\n1. The pain started while she was walking."}
        />
        <h2>Criteria</h2>
        {criteria.map((row) => (
          <div className="criterion" key={row.id}>
            <select
              value={row.criterion_type}
              onChange={(event) => updateCriterion(row.id, { criterion_type: event.target.value })}
            >
              <option value="inclusion">inclusion</option>
              <option value="exclusion">exclusion</option>
            </select>
            <input
              value={row.criterion_text}
              onChange={(event) => updateCriterion(row.id, { criterion_text: event.target.value })}
              placeholder="Criterion text"
            />
            <button type="button" onClick={() => setCriteria((rows) => rows.filter((item) => item.id !== row.id))}>
              Remove
            </button>
          </div>
        ))}
        <div className="row">
          <button type="button" onClick={() => setCriteria((rows) => [...rows, emptyCriterion()])}>
            Add criterion
          </button>
          <button type="button" className="primary" disabled={running} onClick={runAll}>
            {running ? "Running…" : "Run screening"}
          </button>
        </div>
        {formError ? <p className="error">{formError}</p> : null}
      </section>
      <section className="panel">
        <h2>Results</h2>
        {numbered ? (
          <>
            <h3>Numbered note</h3>
            <pre>{numbered}</pre>
          </>
        ) : null}
        {cards.map((card) => (
          <article className="card" key={card.localId}>
            <p className="eyebrow">
              {card.criterion_type}
            </p>
            <h3>{card.criterion_text}</h3>
            {card.ok === false ? <p className="error">{card.error}</p> : null}
            {card.ok !== false ? (
              <>
                <p>
                  <strong>Label:</strong> {card.eligibility}
                </p>
                <p>
                  <strong>Missing information:</strong> {card.missing_information ? "yes" : "no"}
                </p>
                <ul>
                  {(card.evidence || []).length === 0 ? <li>No sentence was cited.</li> : null}
                  {(card.evidence || []).map((item) => (
                    <li key={item.id}>
                      <strong>[{item.id}]</strong> {item.text}
                    </li>
                  ))}
                </ul>
                <p>{card.explanation}</p>
              </>
            ) : null}
          </article>
        ))}
        {overall ? (
          <article className="card overall">
            <h3>Overall line: {overall.outcome}</h3>
            <p>{overall.rule}</p>
            {overall.unscored ? <p>{overall.unscored} criteria were not included because they did not score.</p> : null}
            <button type="button" onClick={downloadReport}>
              Download report
            </button>
          </article>
        ) : null}
      </section>
    </main>
  );
}

function Lab() {
  const [prompt, setPrompt] = useState("Patient note: 0. A 58-year-old woman has chest pain that started while walking.");
  const [tokens, setTokens] = useState(16);
  const [gamma, setGamma] = useState(4);
  const [seq, setSeq] = useState(32);
  const [block, setBlock] = useState(8);
  const [causal, setCausal] = useState(true);
  const [output, setOutput] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function run(path, body) {
    setBusy(true);
    setError("");
    try {
      const payload = await api(path, body);
      setOutput(JSON.stringify(payload, null, 2));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="panel lab">
      <h2>Live comparison</h2>
      <p>These buttons run the implementations in this repository. They are not stored charts.</p>
      <label>
        Prompt
        <textarea rows={4} value={prompt} onChange={(event) => setPrompt(event.target.value)} />
      </label>
      <div className="row">
        <label>
          New tokens
          <input type="number" min="1" max="64" value={tokens} onChange={(event) => setTokens(Number(event.target.value))} />
        </label>
        <label>
          Draft lookahead
          <input type="number" min="1" max="8" value={gamma} onChange={(event) => setGamma(Number(event.target.value))} />
        </label>
      </div>
      <div className="row">
        <button type="button" disabled={busy} onClick={() => run("/api/bench/kv", { prompt, max_new_tokens: tokens })}>
          Cached vs uncached greedy
        </button>
        <button
          type="button"
          disabled={busy}
          onClick={() => run("/api/bench/speculative", { prompt, max_new_tokens: tokens, gamma })}
        >
          Speculative vs greedy
        </button>
      </div>
      <h2>Tiled attention</h2>
      <div className="row">
        <label>
          Sequence
          <input type="number" min="1" max="256" value={seq} onChange={(event) => setSeq(Number(event.target.value))} />
        </label>
        <label>
          Block
          <input type="number" min="1" max="128" value={block} onChange={(event) => setBlock(Number(event.target.value))} />
        </label>
        <label className="check">
          <input type="checkbox" checked={causal} onChange={(event) => setCausal(event.target.checked)} />
          Causal mask
        </label>
      </div>
      <button
        type="button"
        disabled={busy}
        onClick={() => run("/api/bench/attention", { batch: 1, heads: 2, seq, dim: 32, block_size: block, causal })}
      >
        Reference vs tiled attention
      </button>
      {error ? <p className="error">{error}</p> : null}
      {busy ? <p>Running. A long uncached comparison can take a while.</p> : null}
      {output ? <pre>{output}</pre> : null}
    </main>
  );
}
