import React from 'react'

const STEPS = [
  ['intro', 'Introduction'],
  ['sft', 'Semantic Fluency'],
  ['pft', 'Phonemic Fluency'],
  ['ctd', 'Picture Description'],
  ['review', 'Review'],
  ['results', 'Results'],
]

export default function AssessmentStepper({ currentStep }) {
  const normalizedStep = currentStep === 'processing' ? 'review' : currentStep
  const activeIndex = Math.max(0, STEPS.findIndex(([key]) => key === normalizedStep))
  return (
    <nav className="stepper" aria-label="Assessment progress">
      <ol>
        {STEPS.map(([key, label], index) => {
          const state = index < activeIndex ? 'complete' : index === activeIndex ? 'current' : 'upcoming'
          return (
            <li key={key} className={`stepper__item stepper__item--${state}`} aria-current={state === 'current' ? 'step' : undefined}>
              <span className="stepper__number" aria-hidden="true">{index < activeIndex ? '✓' : index + 1}</span>
              <span className="stepper__label">{label}</span>
            </li>
          )
        })}
      </ol>
    </nav>
  )
}
