function writeAscii(view, offset, value) {
  for (let index = 0; index < value.length; index += 1) view.setUint8(offset + index, value.charCodeAt(index))
}

export function encodeAudioBufferAsWav(audioBuffer) {
  const channels = audioBuffer.numberOfChannels
  const sampleRate = audioBuffer.sampleRate
  const frameCount = audioBuffer.length
  if (!channels || !sampleRate || !frameCount) throw new Error('The browser produced an empty decoded recording.')

  const bytesPerSample = 2
  const dataLength = frameCount * channels * bytesPerSample
  const buffer = new ArrayBuffer(44 + dataLength)
  const view = new DataView(buffer)
  writeAscii(view, 0, 'RIFF')
  view.setUint32(4, 36 + dataLength, true)
  writeAscii(view, 8, 'WAVE')
  writeAscii(view, 12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, channels, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * channels * bytesPerSample, true)
  view.setUint16(32, channels * bytesPerSample, true)
  view.setUint16(34, bytesPerSample * 8, true)
  writeAscii(view, 36, 'data')
  view.setUint32(40, dataLength, true)

  const channelData = Array.from({ length: channels }, (_, channel) => audioBuffer.getChannelData(channel))
  let offset = 44
  for (let frame = 0; frame < frameCount; frame += 1) {
    for (let channel = 0; channel < channels; channel += 1) {
      const sample = Math.max(-1, Math.min(1, channelData[channel][frame]))
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true)
      offset += bytesPerSample
    }
  }
  return new Blob([buffer], { type: 'audio/wav' })
}

export async function convertRecordedMediaToWav(sourceBlob) {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext
  if (!AudioContextClass) throw new Error('This browser cannot convert microphone audio to WAV.')
  const context = new AudioContextClass()
  try {
    const encodedBytes = await sourceBlob.arrayBuffer()
    const audioBuffer = await context.decodeAudioData(encodedBytes.slice(0))
    return {
      blob: encodeAudioBufferAsWav(audioBuffer),
      durationSeconds: audioBuffer.duration,
      sampleRate: audioBuffer.sampleRate,
      channelCount: audioBuffer.numberOfChannels,
      sourceMimeType: sourceBlob.type || 'browser-default',
      uploadMimeType: 'audio/wav',
    }
  } finally {
    await context.close?.()
  }
}
