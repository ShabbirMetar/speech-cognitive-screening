export function screeningResponse(classification = 'Healthy-like speech pattern', score = 42) {
  const task = (transcript, wordCount) => ({
    transcript,
    duration_seconds: 12.4,
    word_count: wordCount,
    transcription_runtime_seconds: 0.7,
  })
  const contribution = (feature, label, value, magnitude, direction) => ({
    feature_key: feature,
    label,
    observed_value: value,
    train_reference_mean: value - 0.1,
    was_median_imputed: false,
    log_odds_contribution: direction === 'healthy' ? -magnitude : magnitude,
    contribution_magnitude: magnitude,
    direction: direction === 'healthy' ? 'toward_healthy_like' : 'toward_possible_impairment_like',
    explanation: `${label} contributed toward the model's ${direction === 'healthy' ? 'healthy-like' : 'possible-impairment-like'} screening output.`,
  })
  return {
    request_id: 'demo-request',
    result: {
      classification,
      cognitive_speech_screening_score: score,
      raw_model_score: score / 100,
      research_threshold: 0.53,
    },
    tasks: {
      sft: task('cat dog bird', 3),
      pft: task('paper place person', 3),
      ctd: task('a person is describing a picture', 6),
    },
    observed_biomarkers: [
      { feature_key: 'sft_type_token_ratio', label: 'SFT Lexical Diversity (TTR)', observed_value: 0.75, unit: 'ratio' },
      { feature_key: 'ctd_pause_total_seconds', label: 'CTD Total Pause Duration', observed_value: 1.4, unit: 'seconds' },
    ],
    explanation: {
      method: 'exact frozen linear-model contributions',
      scale: 'log-odds',
      toward_impairment_like: [contribution('ctd_pause_total_seconds', 'CTD Total Pause Duration', 1.4, 0.3, 'impairment')],
      toward_healthy_like: [contribution('sft_type_token_ratio', 'SFT Lexical Diversity (TTR)', 0.75, 0.2, 'healthy')],
    },
    model: {
      model_type: 'Logistic Regression',
      model_version: '1.0.0',
      asr: 'faster-whisper base.en',
      feature_representation: 'ASR_RATE_PLUS_CTD',
      feature_count: 51,
      research_threshold: 0.53,
      disclaimer: 'Research screening prototype; this result is not a medical diagnosis.',
    },
    disclaimer: 'Research screening prototype; this result is not a medical diagnosis.',
  }
}
