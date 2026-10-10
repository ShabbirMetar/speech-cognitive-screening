import React, { useCallback, useEffect, useRef, useState } from 'react'

import AssessmentStepper from './components/AssessmentStepper.jsx'
import ResultsPage from './components/ResultsPage.jsx'
import TaskScreen from './components/TaskScreen.jsx'
import { analyzeScreening, checkHealth } from './services/api.js'

const TASK_ORDER = ['sft', 'pft', 'ctd']

function serviceCopy(state) {
  if (state === 'ready') return ['Service ready', 'ready']
  if (state === 'initializing') return ['Screening model is initializing', 'waiting']
  if (state === 'unavailable') return ['Screening service unavailable', 'error']
  return ['Checking local service', 'waiting']
}

function AppHeader({ serviceState }) {
  const [label, tone] = serviceCopy(serviceState)
  return (
    <header className="app-header">
      <div className="app-header__brand">
        <span className="brand-mark" aria-hidden="true">S</span>
        <div><strong>SpeechScreen</strong><small>Research prototype</small></div>
      </div>
      <div className={`service-pill service-pill--${tone}`} role="status"><span aria-hidden="true" />{label}</div>
    </header>
  )
}

function Landing({ serviceState, onStart, onRefresh }) {
  const ready = serviceState === 'ready'
  return (
    <main className="landing">
      <section className="landing-hero">
        <div className="landing-copy">
          <p className="eyebrow">Explainable multimodal speech analysis</p>
          <h1>Speech-Based<br /><span>Cognitive Decline Screening</span></h1>
          <p className="landing-lead">This research prototype analyzes characteristics of speech and language to produce a preliminary speech-based cognitive screening result.</p>
          <div className="landing-actions">
            <button className="button button--primary button--large" type="button" onClick={onStart} disabled={!ready}>Start assessment</button>
            {!ready && <button className="button button--secondary" type="button" onClick={onRefresh}>Check service again</button>}
          </div>
          <div className="landing-disclaimer" role="note">
            <strong>Research use only</strong>
            <p>This application is a research and educational prototype. It does not provide a medical diagnosis and should not replace assessment by a qualified healthcare professional.</p>
          </div>
        </div>
        <div className="landing-visual" aria-label="Three-part assessment overview">
          <div className="visual-orbit" aria-hidden="true"><span /><span /><span /></div>
          <div className="visual-card visual-card--main"><span className="visual-icon" aria-hidden="true">◉</span><p>Three guided speech tasks</p><strong>Language · timing · voice</strong></div>
          <div className="visual-card visual-card--task"><span>01</span><strong>Semantic fluency</strong></div>
          <div className="visual-card visual-card--task"><span>02</span><strong>Phonemic fluency</strong></div>
          <div className="visual-card visual-card--task"><span>03</span><strong>Picture description</strong></div>
        </div>
      </section>
      <section className="privacy-strip" aria-label="Assessment privacy summary">
        <div><strong>Local processing</strong><span>Audio stays with the local FastAPI service</span></div>
        <div><strong>Temporary recordings</strong><span>Speech is not permanently stored by default</span></div>
        <div><strong>Explainable output</strong><span>Readable factors accompany the screening signal</span></div>
      </section>
    </main>
  )
}

function Introduction({ permissionState, permissionError, onBegin, onBack }) {
  return (
    <main className="content-page intro-page">
      <button className="text-button" type="button" onClick={onBack}>← Back to home</button>
      <div className="section-heading section-heading--center"><p className="eyebrow">Before you begin</p><h1>Three short speech tasks</h1><p>Set aside a few quiet minutes and complete each prompt independently.</p></div>
      <div className="intro-grid">
        <article><span>01</span><h2>Semantic Fluency</h2><p>Name animals for approximately one minute.</p></article>
        <article><span>02</span><h2>Phonemic Fluency</h2><p>Produce words beginning with the letter P.</p></article>
        <article><span>03</span><h2>Picture Description</h2><p>Describe everything happening in the presented picture.</p></article>
      </div>
      <section className="preparation-card">
        <h2>For a clear recording</h2>
        <ul><li>Use a quiet environment and speak naturally.</li><li>Allow microphone access when prompted.</li><li>Complete each task without combining the prompts.</li><li>Recordings are processed for this assessment and are not permanently stored by default.</li></ul>
      </section>
      {permissionError && <div className="inline-alert inline-alert--error" role="alert">{permissionError}</div>}
      <div className="screen-actions screen-actions--center">
        <button className="button button--primary button--large" type="button" onClick={onBegin} disabled={permissionState === 'requesting'}>{permissionState === 'requesting' ? 'Requesting microphone…' : permissionState === 'denied' ? 'Retry microphone access' : 'Begin'}</button>
        <p className="action-hint">Your browser will ask for microphone permission once.</p>
      </div>
    </main>
  )
}

function Review({ recordings, serviceState, submitError, onReRecord, onAnalyze, onBack, onRefresh }) {
  const complete = TASK_ORDER.every((task) => recordings[task])
  const ready = serviceState === 'ready'
  return (
    <main className="content-page review-page">
      <button className="text-button" type="button" onClick={onBack}>← Back</button>
      <div className="section-heading"><p className="eyebrow">Final check</p><h1>Review your recordings</h1><p>Listen back if you wish. Nothing is uploaded until you select Analyze speech.</p></div>
      <div className="review-list">
        {TASK_ORDER.map((task) => {
          const recording = recordings[task]
          return (
            <article className="review-row" key={task}>
              <div className="review-row__status"><span aria-hidden="true">✓</span><div><strong>{task.toUpperCase()}</strong><small>{recording ? `${recording.durationSeconds.toFixed(1)} seconds · WAV ready` : 'Missing recording'}</small></div></div>
              {recording && <audio controls src={recording.url} aria-label={`${task.toUpperCase()} recording playback`} />}
              <button className="button button--secondary" type="button" onClick={() => onReRecord(task)}>Re-record</button>
            </article>
          )
        })}
      </div>
      {!complete && <div className="inline-alert inline-alert--error" role="alert">All three recordings are required before submission.</div>}
      {serviceState !== 'ready' && <div className="inline-alert" role="status">{serviceState === 'initializing' ? 'The screening model is still initializing.' : 'The local screening service is unavailable.'}<button className="text-button" type="button" onClick={onRefresh}>Check again</button></div>}
      {submitError && <div className="inline-alert inline-alert--error" role="alert">{submitError}</div>}
      <div className="screen-actions"><button className="button button--primary button--large" type="button" disabled={!complete || !ready} onClick={onAnalyze}>Analyze speech</button><p className="action-hint">CPU transcription may take a short while.</p></div>
    </main>
  )
}

function Processing() {
  return (
    <main className="processing-page" aria-live="polite">
      <div className="processing-spinner" aria-hidden="true"><span /><span /><span /></div>
      <p className="eyebrow">Local processing</p><h1>Analyzing your recordings</h1><p>This may take a short while on the local CPU. Please keep this page open.</p>
      <ul><li>Preparing three WAV recordings</li><li>Transcribing speech with faster-whisper</li><li>Analyzing language, timing, pauses, pitch and energy</li><li>Generating the screening result and model explanation</li></ul>
      <small>These are processing activities, not precise progress percentages.</small>
    </main>
  )
}

export default function App() {
  const [step, setStep] = useState('home')
  const [serviceState, setServiceState] = useState('checking')
  const [permissionState, setPermissionState] = useState('unknown')
  const [permissionError, setPermissionError] = useState('')
  const [recordings, setRecordings] = useState({ sft: null, pft: null, ctd: null })
  const [returnToReview, setReturnToReview] = useState(false)
  const [result, setResult] = useState(null)
  const [submitError, setSubmitError] = useState('')
  const streamRef = useRef(null)
  const recordingsRef = useRef(recordings)
  const requestControllerRef = useRef(null)

  useEffect(() => { recordingsRef.current = recordings }, [recordings])
  const refreshHealth = useCallback(async () => {
    setServiceState('checking')
    try {
      const health = await checkHealth()
      setServiceState(health.screening_model_ready ? 'ready' : 'initializing')
    } catch {
      setServiceState('unavailable')
    }
  }, [])
  useEffect(() => { refreshHealth() }, [refreshHealth])
  useEffect(() => () => {
    requestControllerRef.current?.abort()
    streamRef.current?.getTracks().forEach((track) => track.stop())
    Object.values(recordingsRef.current).forEach((recording) => recording?.url && URL.revokeObjectURL(recording.url))
  }, [])

  const getMicrophoneStream = useCallback(async () => {
    const active = streamRef.current
    if (active?.getTracks().some((track) => track.readyState === 'live')) return active
    if (!navigator.mediaDevices?.getUserMedia) {
      setPermissionState('unsupported')
      throw new Error('Microphone recording is not supported by this browser.')
    }
    setPermissionState('requesting')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      setPermissionState('granted')
      setPermissionError('')
      return stream
    } catch (error) {
      const denied = error?.name === 'NotAllowedError' || error?.name === 'SecurityError'
      setPermissionState(denied ? 'denied' : 'error')
      const message = denied ? 'Microphone permission was denied. Allow access in your browser settings, then retry.' : 'No usable microphone was found. Check the device connection and browser audio settings.'
      setPermissionError(message)
      throw new Error(message)
    }
  }, [])

  function stopMicrophone() {
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
  }

  async function beginAssessment() {
    try { await getMicrophoneStream(); setStep('sft') } catch { /* The introduction shows the safe permission error. */ }
  }

  function storeRecording(task, recording) {
    setRecordings((current) => {
      if (current[task]?.url && current[task].url !== recording.url) URL.revokeObjectURL(current[task].url)
      return { ...current, [task]: recording }
    })
  }

  function clearRecording(task) {
    setRecordings((current) => {
      if (current[task]?.url) URL.revokeObjectURL(current[task].url)
      return { ...current, [task]: null }
    })
  }

  function continueTask(task) {
    if (returnToReview) { setReturnToReview(false); setStep('review'); refreshHealth(); return }
    const next = TASK_ORDER[TASK_ORDER.indexOf(task) + 1]
    if (next) setStep(next)
    else { setStep('review'); refreshHealth() }
  }

  function goBackFromTask(task) {
    if (returnToReview) { setReturnToReview(false); setStep('review'); return }
    const index = TASK_ORDER.indexOf(task)
    setStep(index === 0 ? 'intro' : TASK_ORDER[index - 1])
  }

  async function submitAssessment() {
    if (!TASK_ORDER.every((task) => recordings[task]) || serviceState !== 'ready') return
    setSubmitError('')
    setStep('processing')
    stopMicrophone()
    const controller = new AbortController()
    requestControllerRef.current = controller
    try {
      setResult(await analyzeScreening(recordings, { signal: controller.signal }))
      setStep('results')
    } catch (error) {
      setSubmitError(error?.message || 'The assessment could not be analyzed. Your recordings are still available to retry.')
      setStep('review')
      refreshHealth()
    } finally { requestControllerRef.current = null }
  }

  function resetAssessment() {
    requestControllerRef.current?.abort()
    stopMicrophone()
    Object.values(recordings).forEach((recording) => recording?.url && URL.revokeObjectURL(recording.url))
    setRecordings({ sft: null, pft: null, ctd: null })
    setPermissionState('unknown')
    setPermissionError('')
    setSubmitError('')
    setResult(null)
    setReturnToReview(false)
    setStep('home')
    refreshHealth()
  }

  return (
    <div className="app-shell">
      <AppHeader serviceState={serviceState} />
      {step !== 'home' && <AssessmentStepper currentStep={step} />}
      {step === 'home' && <Landing serviceState={serviceState} onStart={() => setStep('intro')} onRefresh={refreshHealth} />}
      {step === 'intro' && <Introduction permissionState={permissionState} permissionError={permissionError} onBegin={beginAssessment} onBack={() => setStep('home')} />}
      {TASK_ORDER.includes(step) && (
        <main className="content-page">
          <TaskScreen task={step} recording={recordings[step]} getMicrophoneStream={getMicrophoneStream} onRecorded={(recording) => storeRecording(step, recording)} onClear={() => clearRecording(step)} onContinue={() => continueTask(step)} onBack={() => goBackFromTask(step)} />
        </main>
      )}
      {step === 'review' && <Review recordings={recordings} serviceState={serviceState} submitError={submitError} onReRecord={(task) => { setReturnToReview(true); setStep(task) }} onAnalyze={submitAssessment} onBack={() => setStep('ctd')} onRefresh={refreshHealth} />}
      {step === 'processing' && <Processing />}
      {step === 'results' && result && <main className="content-page"><ResultsPage response={result} onRestart={resetAssessment} /></main>}
      <footer className="app-footer"><span>Research and educational prototype</span><span>Not a medical diagnostic system</span></footer>
    </div>
  )
}
