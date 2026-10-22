import React, { useState } from 'react'
import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import TaskScreen from './TaskScreen.jsx'

vi.mock('./RecordingPanel.jsx', () => ({
  default: () => {
    const [isRecording, setIsRecording] = useState(false)
    return isRecording ? (
      <button type="button" onClick={() => setIsRecording(false)}>Stop recording</button>
    ) : (
      <button type="button" onClick={() => setIsRecording(true)}>Start recording</button>
    )
  },
}))

const defaultProps = {
  task: 'ctd',
  recording: null,
  getMicrophoneStream: vi.fn(),
  onRecorded: vi.fn(),
  onClear: vi.fn(),
  onContinue: vi.fn(),
  onBack: vi.fn(),
}

afterEach(() => {
  cleanup()
  vi.unstubAllEnvs()
})

describe('Picture Description task stimulus', () => {
  it('uses the bundled image with non-leading alt text and no old placeholder', () => {
    render(<TaskScreen {...defaultProps} />)

    const image = screen.getByRole('img', {
      name: 'Picture used for the Cookie Theft description speech task',
    })
    expect(image).toHaveAttribute('src', '/assets/cookie-theft-picture.png')
    expect(screen.queryByText(/stimulus is not bundled/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/facilitator must display/i)).not.toBeInTheDocument()
  })

  it('keeps the picture visible while the recording controls change state', async () => {
    const user = userEvent.setup()
    render(<TaskScreen {...defaultProps} />)

    await user.click(screen.getByRole('button', { name: 'Start recording' }))

    expect(screen.getByRole('button', { name: 'Stop recording' })).toBeInTheDocument()
    expect(screen.getByRole('img', {
      name: 'Picture used for the Cookie Theft description speech task',
    })).toBeInTheDocument()
  })

  it('retains the optional VITE_CTD_STIMULUS_URL override', () => {
    vi.stubEnv('VITE_CTD_STIMULUS_URL', '/assets/authorized-override.png')
    render(<TaskScreen {...defaultProps} />)

    expect(screen.getByRole('img', {
      name: 'Picture used for the Cookie Theft description speech task',
    })).toHaveAttribute('src', '/assets/authorized-override.png')
  })
})
