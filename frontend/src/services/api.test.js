import { describe, expect, it, vi } from 'vitest'

import { analyzeScreening, checkHealth } from './api.js'

function response(payload, ok = true, status = 200) {
  return { ok, status, json: vi.fn().mockResolvedValue(payload) }
}

describe('screening API client', () => {
  it('sends exactly the three WAV fields and lets the browser set the multipart boundary', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(response({ result: 'ok' }))
    const blob = new Blob(['RIFF synthetic'], { type: 'audio/wav' })
    await analyzeScreening(
      { sft: { blob }, pft: { blob }, ctd: { blob } },
      { fetchImpl, timeoutMs: 1_000 },
    )

    const [, request] = fetchImpl.mock.calls[0]
    expect(request.method).toBe('POST')
    expect([...request.body.keys()]).toEqual(['sft_audio', 'pft_audio', 'ctd_audio'])
    expect([...request.body.values()].every((file) => file.type === 'audio/wav')).toBe(true)
    expect(request.headers).toBeUndefined()
  })

  it('refuses submission without all three recordings', async () => {
    const blob = new Blob(['RIFF'], { type: 'audio/wav' })
    await expect(analyzeScreening({ sft: { blob }, pft: { blob } })).rejects.toThrow('All three task recordings')
  })

  it('maps an unavailable health endpoint to a safe user-facing error', async () => {
    await expect(checkHealth({ fetchImpl: vi.fn().mockRejectedValue(new TypeError('network')) })).rejects.toThrow('Screening service unavailable')
  })
})
