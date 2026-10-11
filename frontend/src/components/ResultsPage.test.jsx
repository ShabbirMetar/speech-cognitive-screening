import React from 'react'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'

import ResultsPage from './ResultsPage.jsx'
import { screeningResponse } from '../test/fixtures.js'

function renderResults(classification = 'Healthy-like speech pattern', score = 31) {
  return render(<ResultsPage response={screeningResponse(classification, score, true)} onRestart={() => {}} />)
}

describe('complete feature profile and score display', () => {
  it('renders distinct Healthy TRAIN, current, and Impaired TRAIN references for all 51 features', () => {
    renderResults()
    expect(screen.getByRole('heading', { name: 'Complete speech-feature profile' })).toBeInTheDocument()
    expect(screen.getAllByText('Healthy TRAIN reference').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Current assessment').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Impaired TRAIN reference').length).toBeGreaterThan(0)
    expect(screen.getByLabelText('51 standardized feature comparisons')).toBeInTheDocument()
    expect(screen.getByText(/research aggregates, not clinical normal ranges/i)).toBeInTheDocument()
    expect(document.querySelectorAll('.feature-profile__ticks > span')).toHaveLength(7)
  })

  it('filters the profile by frozen feature group while retaining the all-feature table', async () => {
    const user = userEvent.setup()
    renderResults()
    await user.click(screen.getByRole('button', { name: 'Speech rate (6)' }))
    expect(screen.getByLabelText('6 standardized feature comparisons')).toBeInTheDocument()
    await user.click(screen.getByText('View all 51 calculated speech features'))
    expect(screen.getAllByText(/frozen model used its TRAIN-derived median imputation/i)).toHaveLength(1)
    expect(screen.getAllByText('Higher values push this fitted model toward the Possible impairment-like output.').length).toBeGreaterThan(0)
  })

  it.each([
    [0, '0%'],
    [53, '53%'],
    [100, '100%'],
  ])('fills the score bar to the returned score of %s', (score, expectedWidth) => {
    renderResults('Healthy-like speech pattern', score)
    expect(screen.getByTestId('score-fill')).toHaveStyle({ width: expectedWidth })
    expect(screen.getByTestId('score-marker')).toHaveStyle({ left: expectedWidth })
    expect(screen.getByTestId('threshold-marker')).toHaveStyle({ left: '53%' })
    expect(screen.getByRole('meter', { name: 'Cognitive Speech Screening Score' })).toHaveAttribute(
      'aria-valuetext',
      `Cognitive Speech Screening Score: ${score} out of 100. Research operating point: 53.`,
    )
  })

  it('keeps the backend classification rather than deriving it from the display score', () => {
    renderResults('Healthy-like speech pattern', 53)
    expect(screen.getByText('Healthy-like speech pattern')).toBeInTheDocument()
  })
})
