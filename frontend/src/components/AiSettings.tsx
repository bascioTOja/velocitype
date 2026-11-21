import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { coachApi } from "../api/endpoints";
import type { PromptCustom, PromptKey, PromptSet, TokenKey } from "../api/types";
import { Button, Card, Spinner } from "./ui";
import "./ai-settings.css";

// Must match CoachPromptsUpdate bounds in backend/app/schemas/coach.py.
const MIN_TOKENS = 128;
const MAX_TOKENS = 32768;

const FIELDS: {
  key: PromptKey;
  label: string;
  hint?: string;
  rows: number;
  tokens?: { key: TokenKey; label: string };
}[] = [
  { key: "analysis_system", label: "Analysis · system prompt", rows: 3 },
  {
    key: "analysis_user",
    label: "Analysis · instruction",
    hint: "Use {{data}} where the trainee stats (JSON) should be inserted.",
    rows: 7,
    tokens: { key: "analysis_max_tokens", label: "Analysis · max response tokens" },
  },
  { key: "drill_system", label: "Drill · system prompt", rows: 3 },
  {
    key: "drill_user",
    label: "Drill · instruction",
    hint: "Use {{focus}} where the weak keys should be inserted.",
    rows: 7,
    tokens: { key: "drill_max_tokens", label: "Drill · max response tokens" },
  },
];

type Texts = Record<PromptKey, string>;
// Kept as strings while editing so the input can be cleared mid-typing.
type Tokens = Record<TokenKey, string>;

const TOKEN_KEYS: TokenKey[] = ["analysis_max_tokens", "drill_max_tokens"];

function parseTokens(v: string): number | null {
  const n = Number(v);
  return Number.isInteger(n) && n >= MIN_TOKENS && n <= MAX_TOKENS ? n : null;
}

export function AiSettings() {
  const qc = useQueryClient();
  const { data, isLoading } = useQuery({
    queryKey: ["coach", "prompts"],
    queryFn: coachApi.getPrompts,
  });
  const [fields, setFields] = useState<Texts | null>(null);
  const [tokens, setTokens] = useState<Tokens | null>(null);

  useEffect(() => {
    if (!data) return;
    const c = data.custom;
    const d = data.defaults;
    setFields({
      analysis_system: c.analysis_system ?? d.analysis_system,
      analysis_user: c.analysis_user ?? d.analysis_user,
      drill_system: c.drill_system ?? d.drill_system,
      drill_user: c.drill_user ?? d.drill_user,
    });
    setTokens({
      analysis_max_tokens: String(c.analysis_max_tokens ?? d.analysis_max_tokens),
      drill_max_tokens: String(c.drill_max_tokens ?? d.drill_max_tokens),
    });
  }, [data]);

  const save = useMutation({
    mutationFn: (custom: PromptCustom) => coachApi.savePrompts(custom),
    onSuccess: (res) => qc.setQueryData(["coach", "prompts"], res),
  });

  if (isLoading || !data || !fields || !tokens) {
    return (
      <Card>
        <h3 className="tf-card-title">AI Settings</h3>
        <Spinner />
      </Card>
    );
  }

  const d: PromptSet = data.defaults;
  const tokenOverride = (k: TokenKey): number | null => {
    const n = parseTokens(tokens[k]);
    return n === null || n === d[k] ? null : n;
  };
  // A field equal to the default is sent as null so it tracks future default changes.
  const toCustom = (f: Texts): PromptCustom => ({
    analysis_system: f.analysis_system === d.analysis_system ? null : f.analysis_system,
    analysis_user: f.analysis_user === d.analysis_user ? null : f.analysis_user,
    analysis_max_tokens: tokenOverride("analysis_max_tokens"),
    drill_system: f.drill_system === d.drill_system ? null : f.drill_system,
    drill_user: f.drill_user === d.drill_user ? null : f.drill_user,
    drill_max_tokens: tokenOverride("drill_max_tokens"),
  });

  const isCustom = (k: PromptKey) => fields[k] !== d[k];
  const tokenInvalid = (k: TokenKey) => parseTokens(tokens[k]) === null;
  const anyInvalid = TOKEN_KEYS.some(tokenInvalid);

  return (
    <Card>
      <h3 className="tf-card-title">AI Settings</h3>
      <p className="tf-settings-note">
        The prompts sent to the AI coach. Edit to fine-tune tone and focus;
        the app automatically injects your stats. Leave a field at its default (or
        use “Reset to defaults”) to follow the built-in prompt.
      </p>

      {FIELDS.map((f) => (
        <div key={f.key}>
          <div className="tf-field">
            <label>
              {f.label}
              {isCustom(f.key) && <span className="tf-prompt-custom"> · customized</span>}
            </label>
            <textarea
              className="tf-prompt-area mono"
              rows={f.rows}
              value={fields[f.key]}
              onChange={(e) => setFields({ ...fields, [f.key]: e.target.value })}
            />
            {f.hint && <p className="tf-prompt-hint">{f.hint}</p>}
          </div>

          {f.tokens && (
            <div className="tf-field">
              <label htmlFor={`tf-${f.tokens.key}`}>
                {f.tokens.label}
                {parseTokens(tokens[f.tokens.key]) !== d[f.tokens.key] && (
                  <span className="tf-prompt-custom"> · customized</span>
                )}
              </label>
              <input
                id={`tf-${f.tokens.key}`}
                type="number"
                inputMode="numeric"
                className="tf-prompt-area tf-prompt-tokens mono"
                min={MIN_TOKENS}
                max={MAX_TOKENS}
                step={1}
                value={tokens[f.tokens.key]}
                aria-invalid={tokenInvalid(f.tokens.key)}
                onChange={(e) =>
                  setTokens({ ...tokens, [f.tokens!.key]: e.target.value })
                }
              />
              <p className="tf-prompt-hint">
                {tokenInvalid(f.tokens.key)
                  ? `Enter a whole number between ${MIN_TOKENS} and ${MAX_TOKENS}.`
                  : `Default ${d[f.tokens.key]}. Reasoning ("thinking") models spend tokens before answering - give them ~1500+, or the reply comes back empty`}
              </p>
            </div>
          )}
        </div>
      ))}

      <div className="tf-ai-actions">
        <Button
          variant="primary"
          disabled={save.isPending || anyInvalid}
          onClick={() => save.mutate(toCustom(fields))}
        >
          {save.isPending ? "Saving…" : "Save prompts"}
        </Button>
        <Button
          variant="ghost"
          disabled={save.isPending}
          onClick={() => {
            setFields({
              analysis_system: d.analysis_system,
              analysis_user: d.analysis_user,
              drill_system: d.drill_system,
              drill_user: d.drill_user,
            });
            setTokens({
              analysis_max_tokens: String(d.analysis_max_tokens),
              drill_max_tokens: String(d.drill_max_tokens),
            });
            save.mutate({
              analysis_system: null,
              analysis_user: null,
              analysis_max_tokens: null,
              drill_system: null,
              drill_user: null,
              drill_max_tokens: null,
            });
          }}
        >
          Reset to defaults
        </Button>
        {save.isSuccess && <span className="tf-ai-saved">Saved ✓</span>}
      </div>
    </Card>
  );
}
