export function featureProfileFixture() {
  return {
    available: true,
    source_description: 'Comparison with descriptive reference profiles from the 320-participant PROCESS-2 TRAIN partition. These references are research aggregates, not clinical normal ranges.',
    source_participants: 320,
    healthy_train_participants: 160,
    impaired_train_participants: 160,
    official_test_participants_used: 0,
    standardized_value_method: 'Frozen TRAIN-fitted median imputer followed by frozen StandardScaler; no scaler is fitted here.',
    features: Array.from({ length: 51 }, (_, index) => {
      const group = index < 32 ? 'ASR_LINGUISTIC' : index < 38 ? 'ASR_RATE' : 'CTD_ACOUSTIC'
      const task = group === 'CTD_ACOUSTIC' ? 'CTD' : index % 3 === 0 ? 'SFT' : index % 3 === 1 ? 'PFT' : 'CTD'
      const rawValue = index === 4 ? null : index + 0.5
      return {
        feature_key: `${task.toLowerCase()}_feature_${index + 1}`,
        label: index === 0 ? 'SFT Word Count' : `Feature ${index + 1}`,
        task,
        group,
        unit: group === 'ASR_RATE' ? 'words per minute' : 'ratio',
        raw_value: rawValue,
        model_input_value: rawValue === null ? 2.25 : rawValue,
        was_median_imputed: rawValue === null,
        standardized_value: index === 0 ? 0.5 : (index % 7) - 3,
        healthy_reference: { valid_n: 160, mean: 1.1, median: 1.0, q25: 0.8, q75: 1.3 },
        impaired_reference: { valid_n: 160, mean: 1.4, median: 1.2, q25: 0.9, q75: 1.7 },
        healthy_reference_standardized: -0.25,
        impaired_reference_standardized: 0.35,
        coefficient: index % 2 === 0 ? 0.2 : -0.2,
        model_direction: index % 2 === 0 ? 'higher_values_toward_possible_impairment_like' : 'higher_values_toward_healthy_like',
        local_log_odds_contribution: index % 2 === 0 ? 0.1 : -0.1,
      }
    }),
  }
}

export function screeningResponse(classification = 'Healthy-like speech pattern', score = 42, includeProfile = false) {
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
    feature_profile: includeProfile ? featureProfileFixture() : null,
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
