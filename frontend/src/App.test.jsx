import React from 'react'
import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import App from './App.jsx'
import { analyzeScreening, checkHealth } from './services/api.js'
import { screeningResponse } from './test/fixtures.js'

vi.mock('./services/api.js', () => ({
  checkHealth: vi.fn(),
  analyzeScreening: vi.fn(),
}))

vi.mock('./components/RecordingPanel.jsx', () => ({
  default: ({ taskName, recording, onRecorded, onClear }) => (
    <section aria-label={`${taskName} mock recorder`}>
      {!recording ? (
        <button type="button" onClick={() => onRecorded({
          blob: new Blob(['RIFF test'], { type: 'audio/wav' }),
          url: `blob:${taskName.toLowerCase()}`,
          durationSeconds: 8.5,
          uploadMimeType: 'audio/wav',
        })}>Record {taskName}</button>
      ) : (
        <><span>{taskName} recording ready</span><button type="button" onClick={onClear}>Clear {taskName}</button></>
      )}
    </section>
  ),
}))

const track = { readyState: 'live', stop: vi.fn() }

async function begin(user) {
  const start = await screen.findByRole('button', { name: 'Start assessment' })
  await waitFor(() => expect(start).toBeEnabled())
  await user.click(start)
  await user.click(screen.getByRole('button', { name: 'Begin' }))
  await screen.findByRole('heading', { name: 'Semantic Fluency Task' })
}

async function recordAll(user) {
  await user.click(screen.getByRole('button', { name: 'Record SFT' }))
  await user.click(screen.getByRole('button', { name: 'Continue' }))
  await user.click(screen.getByRole('button', { name: 'Record PFT' }))
  await user.click(screen.getByRole('button', { name: 'Continue' }))
  await user.click(screen.getByRole('button', { name: 'Record CTD' }))
  await user.click(screen.getByRole('button', { name: 'Continue' }))
  await screen.findByRole('heading', { name: 'Review your recordings' })
}

beforeEach(() => {
  checkHealth.mockResolvedValue({ status: 'ok', screening_model_ready: true })
  analyzeScreening.mockResolvedValue(screeningResponse())
  Object.defineProperty(navigator, 'mediaDevices', {
    configurable: true,
    value: { getUserMedia: vi.fn().mockResolvedValue({ getTracks: () => [track] }) },
  })
  URL.createObjectURL = vi.fn(() => 'blob:mock')
  URL.revokeObjectURL = vi.fn()
  track.stop.mockClear()
})

describe('assessment workflow', () => {
  it('shows the landing disclaimer and prevents advancing without a recording', async () => {
    const user = userEvent.setup()
    render(<App />)
    expect(screen.getByText(/does not provide a medical diagnosis/i)).toBeInTheDocument()
    await begin(user)
    expect(screen.getByRole('button', { name: 'Continue' })).toBeDisabled()
    expect(screen.getByText('Record this task before continuing.')).toBeInTheDocument()
  })

  it('shows a clear microphone-denied state', async () => {
    navigator.mediaDevices.getUserMedia.mockRejectedValueOnce(Object.assign(new Error('denied'), { name: 'NotAllowedError' }))
    const user = userEvent.setup()
    render(<App />)
    const start = await screen.findByRole('button', { name: 'Start assessment' })
    await waitFor(() => expect(start).toBeEnabled())
    await user.click(start)
    await user.click(screen.getByRole('button', { name: 'Begin' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Microphone permission was denied')
  })

  it('navigates all tasks and requires all three recordings before submission', async () => {
    const user = userEvent.setup()
    render(<App />)
    await begin(user)
    await recordAll(user)
    const analyze = screen.getByRole('button', { name: 'Analyze speech' })
    expect(analyze).toBeEnabled()
    await user.click(screen.getAllByRole('button', { name: 'Re-record' })[0])
    await user.click(screen.getByRole('button', { name: 'Clear SFT' }))
    await user.click(screen.getByRole('button', { name: '← Back' }))
    expect(screen.getByRole('button', { name: 'Analyze speech' })).toBeDisabled()
    expect(screen.getByText(/All three recordings are required/)).toBeInTheDocument()
  })

  it.each([
    ['Healthy-like speech pattern', 42],
    ['Possible impairment-like speech pattern', 67],
  ])('renders the returned %s result and research disclaimer', async (classification, score) => {
    analyzeScreening.mockResolvedValueOnce(screeningResponse(classification, score))
    const user = userEvent.setup()
    render(<App />)
    await begin(user)
    await recordAll(user)
    await user.click(screen.getByRole('button', { name: 'Analyze speech' }))

    expect(await screen.findByRole('heading', { name: 'Cognitive Speech Screening Score' })).toBeInTheDocument()
    expect(screen.getByText(classification)).toBeInTheDocument()
    expect(screen.getByText(/not a medical diagnosis or a validated estimate/i)).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Factors influencing this model output' })).toBeInTheDocument()
    expect(screen.getByText(/do not establish medical causes or represent probability changes/i)).toBeInTheDocument()
  })

  it('shows an indeterminate processing state while the request is pending', async () => {
    let resolveRequest
    analyzeScreening.mockReturnValueOnce(new Promise((resolve) => { resolveRequest = resolve }))
    const user = userEvent.setup()
    render(<App />)
    await begin(user)
    await recordAll(user)
    await user.click(screen.getByRole('button', { name: 'Analyze speech' }))
    expect(screen.getByRole('heading', { name: 'Analyzing your recordings' })).toBeInTheDocument()
    expect(screen.getByText(/not precise progress percentages/i)).toBeInTheDocument()
    await act(async () => resolveRequest(screeningResponse()))
    expect(await screen.findByRole('heading', { name: 'Cognitive Speech Screening Score' })).toBeInTheDocument()
  })

  it('returns safely to review when backend analysis fails', async () => {
    analyzeScreening.mockRejectedValueOnce(new Error('Speech transcription could not be completed.'))
    const user = userEvent.setup()
    render(<App />)
    await begin(user)
    await recordAll(user)
    await user.click(screen.getByRole('button', { name: 'Analyze speech' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Speech transcription could not be completed')
    expect(screen.getByRole('heading', { name: 'Review your recordings' })).toBeInTheDocument()
  })

  it('clears object URLs and returns home for a new assessment', async () => {
    const user = userEvent.setup()
    render(<App />)
    await begin(user)
    await recordAll(user)
    await user.click(screen.getByRole('button', { name: 'Analyze speech' }))
    await user.click(await screen.findByRole('button', { name: 'Start new assessment' }))
    expect(screen.getByRole('heading', { name: /Speech-Based/i })).toBeInTheDocument()
    expect(URL.revokeObjectURL).toHaveBeenCalledTimes(3)
  })
})
