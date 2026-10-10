import { describe, expect, it } from 'vitest'

import { encodeAudioBufferAsWav } from './wavEncoder.js'

describe('WAV encoder', () => {
  it('creates an interleaved PCM WAV without resampling or changing channel count', async () => {
    const channels = [new Float32Array([0, 0.5, -0.5]), new Float32Array([0.25, -0.25, 1])]
    const wav = encodeAudioBufferAsWav({
      numberOfChannels: 2,
      sampleRate: 48_000,
      length: 3,
      getChannelData: (channel) => channels[channel],
    })
    const arrayBuffer = await new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(reader.result)
      reader.onerror = () => reject(reader.error)
      reader.readAsArrayBuffer(wav)
    })
    const bytes = new Uint8Array(arrayBuffer)
    const header = new TextDecoder().decode(bytes.slice(0, 12))
    const view = new DataView(bytes.buffer)

    expect(wav.type).toBe('audio/wav')
    expect(header.startsWith('RIFF')).toBe(true)
    expect(header.endsWith('WAVE')).toBe(true)
    expect(view.getUint16(22, true)).toBe(2)
    expect(view.getUint32(24, true)).toBe(48_000)
    expect(bytes.length).toBe(44 + 3 * 2 * 2)
  })
})
