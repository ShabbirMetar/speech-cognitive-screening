import React, { useMemo, useState } from 'react'

const PROFILE_LIMIT = 3

const FILTERS = [
  { key: 'all', label: 'All (51)' },
  { key: 'ASR_LINGUISTIC', label: 'Linguistic (32)' },
  { key: 'ASR_RATE', label: 'Speech rate (6)' },
  { key: 'CTD_ACOUSTIC', label: 'CTD acoustic (13)' },
]

const GROUP_LABELS = {
  ASR_LINGUISTIC: 'Linguistic',
  ASR_RATE: 'Speech rate',
  CTD_ACOUSTIC: 'CTD acoustic',
}

function displayNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'Unavailable'
  const numeric = Number(value)
  if (Math.abs(numeric) >= 100) return numeric.toFixed(0)
  if (Math.abs(numeric) < 0.01 && numeric !== 0) return numeric.toExponential(2)
  return numeric.toFixed(digits)
}

function standardizedNumber(value) {
  const numeric = Number(value)
  return `${numeric >= 0 ? '+' : ''}${numeric.toFixed(2)} SD`
}

function standardizedPosition(value) {
  const numeric = Number(value)
  return Math.min(100, Math.max(0, (numeric + PROFILE_LIMIT) / (PROFILE_LIMIT * 2) * 100))
}

function cappedSuffix(value) {
  return Math.abs(Number(value)) > PROFILE_LIMIT ? ' (display capped)' : ''
}

function directionText(direction) {
  if (direction === 'higher_values_toward_possible_impairment_like') {
    return 'Higher values push this fitted model toward the Possible impairment-like output.'
  }
  if (direction === 'higher_values_toward_healthy_like') {
    return 'Higher values push this fitted model toward the Healthy-like output.'
  }
  return 'This fitted model assigns a weak or negligible linear effect.'
}

function referenceSummary(reference, unit) {
  return `Mean ${displayNumber(reference.mean)} ${unit}; median ${displayNumber(reference.median)} ${unit}`
}

function ProfileTrack({ item }) {
  const healthyPosition = standardizedPosition(item.healthy_reference_standardized)
  const currentPosition = standardizedPosition(item.standardized_value)
  const impairedPosition = standardizedPosition(item.impaired_reference_standardized)
  const label = `${item.label}. Healthy TRAIN reference ${standardizedNumber(item.healthy_reference_standardized)}; current assessment ${standardizedNumber(item.standardized_value)}; Impaired TRAIN reference ${standardizedNumber(item.impaired_reference_standardized)}. Standardized display spans minus 3 to plus 3 standard deviations; positions outside that range are capped visually.`
  return (
    <div className="feature-profile-track" role="img" aria-label={label}>
      <span className="feature-profile-track__baseline" aria-hidden="true" />
      <span className="feature-profile-marker feature-profile-marker--healthy" style={{ left: `${healthyPosition}%` }} aria-hidden="true" title={`Healthy TRAIN: ${standardizedNumber(item.healthy_reference_standardized)}`} />
      <span className="feature-profile-marker feature-profile-marker--current" style={{ left: `${currentPosition}%` }} aria-hidden="true" title={`Current: ${standardizedNumber(item.standardized_value)}`} />
      <span className="feature-profile-marker feature-profile-marker--impaired" style={{ left: `${impairedPosition}%` }} aria-hidden="true" title={`Impaired TRAIN: ${standardizedNumber(item.impaired_reference_standardized)}`} />
    </div>
  )
}

function RawValueCell({ item }) {
  if (item.raw_value !== null && item.raw_value !== undefined) {
    return <>{displayNumber(item.raw_value)} <span className="feature-unit">{item.unit}</span></>
  }
  return (
    <span className="feature-imputed-note">
      Undefined for this transcript; the frozen model used its TRAIN-derived median imputation ({displayNumber(item.model_input_value)} {item.unit}).
    </span>
  )
}

export default function FeatureProfile({ profile }) {
  const [filter, setFilter] = useState('all')
  const filteredFeatures = useMemo(() => {
    if (!profile?.features) return []
    return filter === 'all' ? profile.features : profile.features.filter((item) => item.group === filter)
  }, [filter, profile])

  if (!profile?.available) {
    return (
      <section className="results-section" aria-labelledby="feature-profile-heading">
        <div className="section-heading">
          <p className="eyebrow">All frozen predictors</p>
          <h2 id="feature-profile-heading">Complete speech-feature profile</h2>
          <p>The optional aggregate TRAIN reference visualization is unavailable. The frozen screening result remains available and unchanged.</p>
        </div>
      </section>
    )
  }

  return (
    <section className="results-section" aria-labelledby="feature-profile-heading">
      <div className="section-heading">
        <p className="eyebrow">All frozen predictors</p>
        <h2 id="feature-profile-heading">Complete speech-feature profile</h2>
        <p>{profile.source_description}</p>
        <p className="profile-method-note">A standardized value indicates how far a feature is from the TRAIN reference mean in standard-deviation units. It does not mean left is healthy or right is impaired; the two cohort markers show the observed group averages, while every feature has its own fitted-model direction.</p>
      </div>

      <div className="profile-legend" aria-label="Feature profile legend">
        <span><i className="legend-marker legend-marker--healthy" aria-hidden="true" />Healthy TRAIN reference</span>
        <span><i className="legend-marker legend-marker--current" aria-hidden="true" />Current assessment</span>
        <span><i className="legend-marker legend-marker--impaired" aria-hidden="true" />Impaired TRAIN reference</span>
      </div>

      <div className="profile-filter" aria-label="Filter features by modality">
        {FILTERS.map((item) => (
          <button key={item.key} className={`profile-filter__button ${filter === item.key ? 'is-active' : ''}`} type="button" aria-pressed={filter === item.key} onClick={() => setFilter(item.key)}>{item.label}</button>
        ))}
      </div>

      <div className="feature-profile" aria-label={`${filteredFeatures.length} standardized feature comparisons`}>
        <div className="feature-profile__axis" aria-hidden="true">
          <div className="feature-profile__ticks"><span>-3</span><span>-2</span><span>-1</span><span>0</span><span>+1</span><span>+2</span><span>+3</span></div>
        </div>
        {filteredFeatures.map((item) => (
          <article className="feature-profile-row" key={item.feature_key}>
            <div className="feature-profile-row__title"><strong>{item.label}</strong><small>{item.task} · {GROUP_LABELS[item.group]}</small></div>
            <ProfileTrack item={item} />
            <div className="feature-profile-row__values" aria-label={`${item.label} standardized values`}>
              <span>Healthy {standardizedNumber(item.healthy_reference_standardized)}{cappedSuffix(item.healthy_reference_standardized)}</span>
              <span>Current {standardizedNumber(item.standardized_value)}{cappedSuffix(item.standardized_value)}</span>
              <span>Impaired {standardizedNumber(item.impaired_reference_standardized)}{cappedSuffix(item.impaired_reference_standardized)}</span>
            </div>
          </article>
        ))}
      </div>

      <details className="all-features-details">
        <summary>View all 51 calculated speech features</summary>
        <p>Every row below is passed to the frozen model in its fixed schema order. Raw values are shown without global imputation; only an undefined value is labelled when the frozen pipeline used its TRAIN-derived median imputation.</p>
        <div className="all-features-table-wrap" tabIndex="0" aria-label="All calculated frozen speech features table">
          <table className="all-features-table">
            <thead><tr><th>Feature</th><th>Task / group</th><th>Current raw value</th><th>Healthy TRAIN reference</th><th>Impaired TRAIN reference</th><th>Current standardized value</th><th>Model direction</th><th>Local contribution</th></tr></thead>
            <tbody>
              {profile.features.map((item) => (
                <tr key={`table-${item.feature_key}`}>
                  <th scope="row">{item.label}<small>{item.feature_key}</small></th>
                  <td>{item.task}<small>{GROUP_LABELS[item.group]}</small></td>
                  <td><RawValueCell item={item} /></td>
                  <td>{referenceSummary(item.healthy_reference, item.unit)}<small>n = {item.healthy_reference.valid_n}</small></td>
                  <td>{referenceSummary(item.impaired_reference, item.unit)}<small>n = {item.impaired_reference.valid_n}</small></td>
                  <td>{standardizedNumber(item.standardized_value)}</td>
                  <td>{directionText(item.model_direction)}</td>
                  <td>{item.local_log_odds_contribution >= 0 ? '+' : ''}{item.local_log_odds_contribution.toFixed(3)} log-odds</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="all-features-details__note">Reference profiles are descriptive aggregates from the 320-participant TRAIN cohort, not clinical normal ranges or disease ranges. Contributions describe the fitted model and do not establish medical causes.</p>
      </details>
    </section>
  )
}
