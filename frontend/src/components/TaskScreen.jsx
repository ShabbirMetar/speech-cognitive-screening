import React, { useEffect, useState } from 'react'

import RecordingPanel from './RecordingPanel.jsx'

const TASKS = {
  sft: {
    eyebrow: 'Task 1 of 3',
    title: 'Semantic Fluency Task',
    prompt: 'Name as many animals as you can.',
    detail: 'Speak naturally for approximately one minute. Stop when you are finished; the interface will not cut you off automatically.',
    accent: 'Animals',
  },
  pft: {
    eyebrow: 'Task 2 of 3',
    title: 'Phonemic Fluency Task',
    prompt: 'Say as many words as you can that begin with the letter P.',
    detail: 'Speak naturally for approximately one minute. Do not use the same word repeatedly.',
    accent: 'P',
  },
  ctd: {
    eyebrow: 'Task 3 of 3',
    title: 'Picture Description Task',
    prompt: 'Describe everything you see happening in the picture.',
    detail: 'Speak naturally until you feel your description is finished.',
    accent: 'Describe',
  },
}

const BUNDLED_STIMULUS_URL = '/assets/cookie-theft-picture.png'

function Stimulus({ source }) {
  const [currentSource, setCurrentSource] = useState(source)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    setCurrentSource(source)
    setFailed(false)
  }, [source])

  function handleImageError() {
    if (currentSource !== BUNDLED_STIMULUS_URL) {
      setCurrentSource(BUNDLED_STIMULUS_URL)
      return
    }
    setFailed(true)
  }

  if (failed) {
    return (
      <p className="inline-alert inline-alert--error" role="alert">
        The picture stimulus could not be displayed. Please reload the page before recording this task.
      </p>
    )
  }

  return (
    <figure className="stimulus-figure">
      <img
        src={currentSource}
        alt="Picture used for the Cookie Theft description speech task"
        onError={handleImageError}
      />
      <figcaption>Keep the picture visible while you describe everything you notice.</figcaption>
    </figure>
  )
}

export default function TaskScreen({ task, recording, getMicrophoneStream, onRecorded, onClear, onContinue, onBack }) {
  const content = TASKS[task]
  return (
    <section className={`task-layout task-layout--${task}`}>
      <div className="task-copy">
        <button className="text-button" type="button" onClick={onBack}>← Back</button>
        <p className="eyebrow">{content.eyebrow}</p>
        <h1>{content.title}</h1>
        <div className="task-prompt">
          <span className="task-prompt__accent" aria-hidden="true">{content.accent}</span>
          <div><p>{content.prompt}</p><small>{content.detail}</small></div>
        </div>
        {task === 'ctd' && (
          <Stimulus source={import.meta.env.VITE_CTD_STIMULUS_URL || BUNDLED_STIMULUS_URL} />
        )}
      </div>
      <div>
        <RecordingPanel taskName={task.toUpperCase()} recording={recording} getMicrophoneStream={getMicrophoneStream} onRecorded={onRecorded} onClear={onClear} />
        <div className="screen-actions">
          <button className="button button--primary" type="button" disabled={!recording} onClick={onContinue}>Continue</button>
          {!recording && <p className="action-hint">Record this task before continuing.</p>}
        </div>
      </div>
    </section>
  )
}
