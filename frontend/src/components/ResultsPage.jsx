import React from 'react'

const BIOMARKER_DEFINITIONS = {
  type_token_ratio: 'The proportion of distinct words in the automatic transcript.',
  recording_word_rate_wpm: 'Recognized words divided by the complete recording duration.',
  articulation_rate_wpm: 'Recognized words divided by detected voiced duration.',
  pause_total_seconds: 'Total duration of detected internal pauses in the CTD recording.',
  pause_mean_seconds: 'Average duration of detected internal pauses in the CTD recording.',
  silence_ratio: 'The measured proportion of CTD recording time classified as silence.',
  f0_std_hz: 'Variation in estimated pitch during the CTD recording.',
  rms_mean: 'Average measured signal energy in the CTD recording.',
  p_initial_word_count: 'Number of recognized P-initial words in the PFT transcript.',
}

function numberValue(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'Unavailable'
  const numeric = Number(value)
  return Math.abs(numeric) >= 100 ? numeric.toFixed(0) : numeric.toFixed(2)
}

function definitionFor(featureKey) {
  const suffix = Object.keys(BIOMARKER_DEFINITIONS).find((key) => featureKey.endsWith(key))
  return BIOMARKER_DEFINITIONS[suffix] || 'A speech characteristic measured by the frozen feature pipeline.'
}

function ScoreGauge({ score, threshold }) {
  const safeScore = Math.min(100, Math.max(0, score))
  return (
    <div className="score-gauge" role="img" aria-label={`Screening score ${safeScore.toFixed(0)} out of 100; research operating point ${threshold.toFixed(0)}`}>
      <div className="score-gauge__labels" aria-hidden="true"><span>0</span><span>100</span></div>
      <div className="score-gauge__track">
        <span className="score-gauge__fill" style={{ width: `${safeScore}%` }} />
        <span className="score-gauge__threshold" style={{ left: `${threshold}%` }} />
        <span className="score-gauge__marker" style={{ left: `${safeScore}%` }} />
      </div>
      <p><span className="threshold-key" aria-hidden="true" /> Research operating point: {threshold.toFixed(0)} / 100</p>
    </div>
  )
}

function ContributionList({ title, items, tone }) {
  const maximum = Math.max(...items.map((item) => item.contribution_magnitude), 0.0001)
  return (
    <section className={`contribution-group contribution-group--${tone}`}>
      <h3>{title}</h3>
      {items.length === 0 ? <p className="muted">No signed contributors were returned in this direction.</p> : (
        <ul>
          {items.map((item) => (
            <li key={`${tone}-${item.feature_key}`}>
              <div className="contribution-heading"><strong>{item.label}</strong><span>{numberValue(item.observed_value)}</span></div>
              <div className="contribution-bar" aria-hidden="true"><span style={{ width: `${Math.max(6, item.contribution_magnitude / maximum * 100)}%` }} /></div>
              <p>{item.explanation}</p>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}

export default function ResultsPage({ response, onRestart }) {
  const { result, tasks, observed_biomarkers: biomarkers, explanation, model } = response
  const score = result.cognitive_speech_screening_score
  const thresholdDisplay = result.research_threshold * 100
  return (
    <div className="results-page">
      <section className="result-hero" aria-labelledby="result-heading">
        <p className="eyebrow">Assessment complete</p>
        <h1 id="result-heading">Cognitive Speech Screening Score</h1>
        <div className="score-number"><strong>{Math.round(score)}</strong><span>/ 100</span></div>
        <p className="score-caption">Display transformation of the frozen classifier score—not a disease probability.</p>
        <ScoreGauge score={score} threshold={thresholdDisplay} />
        <div className="classification-card"><span>Screening pattern</span><strong>{result.classification}</strong></div>
      </section>

      <section className="notice-card notice-card--important" aria-labelledby="result-disclaimer-title">
        <h2 id="result-disclaimer-title">How to interpret this result</h2>
        <p>This result reflects speech patterns identified by a research machine learning model. It is not a medical diagnosis or a validated estimate of the probability of dementia or another disease.</p>
        <p>If you have concerns about memory, thinking, or cognitive health, consider speaking with a qualified healthcare professional.</p>
      </section>

      <section className="results-section" aria-labelledby="task-summary-heading">
        <div className="section-heading"><p className="eyebrow">Recorded tasks</p><h2 id="task-summary-heading">Task summary</h2></div>
        <div className="task-summary-grid">
          {Object.entries(tasks).map(([task, taskResult]) => (
            <article className="summary-card" key={task}>
              <span className="task-chip">{task.toUpperCase()}</span>
              <dl><div><dt>Duration</dt><dd>{taskResult.duration_seconds.toFixed(1)} s</dd></div><div><dt>Recognized words</dt><dd>{taskResult.word_count}</dd></div></dl>
              <details><summary>Automatic transcript</summary><p>{taskResult.transcript || 'No transcript text returned.'}</p><small>Generated by faster-whisper and may contain recognition errors.</small></details>
            </article>
          ))}
        </div>
      </section>

      <section className="results-section" aria-labelledby="observed-heading">
        <div className="section-heading"><p className="eyebrow">Descriptive measurements</p><h2 id="observed-heading">Observed speech characteristics</h2><p>Neutral measurements from this assessment. They are not labelled normal or abnormal.</p></div>
        <div className="biomarker-grid">
          {biomarkers.map((item) => (
            <article className="biomarker-card" key={item.feature_key}><h3>{item.label}</h3><p className="biomarker-value">{numberValue(item.observed_value)} <span>{item.unit}</span></p><p>{definitionFor(item.feature_key)}</p></article>
          ))}
        </div>
      </section>

      <section className="results-section" aria-labelledby="explanation-heading">
        <div className="section-heading"><p className="eyebrow">Local model explanation · log-odds scale</p><h2 id="explanation-heading">Factors influencing this model output</h2><p>These values explain the behavior of this machine-learning model; they do not establish medical causes or represent probability changes.</p></div>
        <div className="contribution-grid">
          <ContributionList title="Toward Possible impairment-like output" items={explanation.toward_impairment_like} tone="impairment" />
          <ContributionList title="Toward Healthy-like output" items={explanation.toward_healthy_like} tone="healthy" />
        </div>
      </section>

      <details className="model-details">
        <summary>Model details</summary>
        <dl><div><dt>Classifier</dt><dd>{model.model_type}</dd></div><div><dt>ASR</dt><dd>{model.asr}</dd></div><div><dt>Predictors</dt><dd>{model.feature_count}</dd></div><div><dt>Modalities</dt><dd>Linguistic + speech-rate + CTD acoustic</dd></div><div><dt>Research threshold</dt><dd>{result.research_threshold.toFixed(2)}</dd></div></dl>
        <p>Official held-out evaluation is available in the project documentation and is not mixed into this individual result.</p>
      </details>
      <div className="results-actions"><button className="button button--primary" type="button" onClick={onRestart}>Start new assessment</button></div>
    </div>
  )
}
