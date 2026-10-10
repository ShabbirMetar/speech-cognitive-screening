import React, { useEffect, useRef, useState } from 'react'

import { convertRecordedMediaToWav } from '../utils/wavEncoder.js'

const MIN_RECORDING_SECONDS = 1

function preferredMimeType() {
  const candidates = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/ogg']
  return candidates.find((type) => window.MediaRecorder?.isTypeSupported?.(type)) || ''
}

function formatTimer(seconds) {
  const value = Math.max(0, Math.floor(seconds))
  return `${String(Math.floor(value / 60)).padStart(2, '0')}:${String(value % 60).padStart(2, '0')}`
}

export default function RecordingPanel({ taskName, recording, getMicrophoneStream, onRecorded, onClear }) {
  const [isRecording, setIsRecording] = useState(false)
  const [isConverting, setIsConverting] = useState(false)
  const [elapsedSeconds, setElapsedSeconds] = useState(0)
  const [error, setError] = useState('')
  const recorderRef = useRef(null)
  const timerRef = useRef(null)
  const startedAtRef = useRef(0)

  useEffect(() => () => {
    window.clearInterval(timerRef.current)
    const recorder = recorderRef.current
    if (recorder && recorder.state !== 'inactive') {
      recorder.onstop = null
      recorder.stop()
    }
  }, [])

  async function startRecording() {
    setError('')
    if (!window.MediaRecorder) {
      setError('This browser does not support microphone recording. Try a current version of Chrome, Edge, or Firefox.')
      return
    }
    try {
      const stream = await getMicrophoneStream()
      const mimeType = preferredMimeType()
      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream)
      const chunks = []
      recorderRef.current = recorder
      recorder.ondataavailable = (event) => { if (event.data?.size) chunks.push(event.data) }
      recorder.onerror = () => {
        setError('The browser could not record this task. Check the microphone and retry.')
        setIsRecording(false)
      }
      recorder.onstop = async () => {
        window.clearInterval(timerRef.current)
        setIsRecording(false)
        setIsConverting(true)
        try {
          const sourceBlob = new Blob(chunks, { type: recorder.mimeType || mimeType })
          const converted = await convertRecordedMediaToWav(sourceBlob)
          if (converted.durationSeconds < MIN_RECORDING_SECONDS) throw new Error('The recording was too short. Record at least one second of speech.')
          onRecorded({ ...converted, url: URL.createObjectURL(converted.blob) })
        } catch (conversionError) {
          setError(conversionError?.message || 'The recording could not be prepared as WAV audio.')
        } finally {
          setIsConverting(false)
        }
      }
      recorder.start(250)
      startedAtRef.current = performance.now()
      setElapsedSeconds(0)
      setIsRecording(true)
      timerRef.current = window.setInterval(() => setElapsedSeconds((performance.now() - startedAtRef.current) / 1000), 250)
    } catch (recordingError) {
      setError(recordingError?.message || 'Microphone access was not available.')
    }
  }

  function stopRecording() {
    const recorder = recorderRef.current
    if (recorder?.state === 'recording') recorder.stop()
  }

  return (
    <section className="recording-card" aria-labelledby={`${taskName}-recording-title`}>
      <div className="recording-card__status">
        <div>
          <p className="eyebrow" id={`${taskName}-recording-title`}>Microphone recording</p>
          <p className="recording-state" aria-live="polite">{isRecording ? 'Recording in progress' : isConverting ? 'Preparing WAV recording' : recording ? 'Recording ready' : 'Ready to record'}</p>
        </div>
        <div className={`recording-timer ${isRecording ? 'recording-timer--active' : ''}`} aria-label={`Recording time ${formatTimer(elapsedSeconds)}`}>
          <span className="recording-dot" aria-hidden="true" />{formatTimer(isRecording ? elapsedSeconds : recording?.durationSeconds || 0)}
        </div>
      </div>
      {error && <div className="inline-alert inline-alert--error" role="alert">{error}</div>}
      {!recording && !isRecording && !isConverting && <button className="button button--primary" type="button" onClick={startRecording}>Start recording</button>}
      {isRecording && <button className="button button--stop" type="button" onClick={stopRecording}>Stop recording</button>}
      {isConverting && <div className="mini-loader" role="status">Converting locally to WAV…</div>}
      {recording && !isRecording && (
        <div className="recording-playback">
          <audio controls src={recording.url} aria-label={`${taskName} recording playback`} />
          <div className="recording-meta"><span>WAV upload</span><span>{recording.durationSeconds.toFixed(1)} seconds</span></div>
          <button className="button button--secondary" type="button" onClick={() => { setError(''); setElapsedSeconds(0); onClear() }}>Retry recording</button>
        </div>
      )}
    </section>
  )
}
