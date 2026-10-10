const DEFAULT_API_BASE_URL = 'http://127.0.0.1:8000'

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE_URL).replace(/\/+$/, '')

export class ApiError extends Error {
  constructor(message, status = null, code = 'request_failed') {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
  }
}

async function responseError(response) {
  let message = 'The screening service could not complete the request.'
  try {
    const payload = await response.json()
    if (typeof payload?.detail === 'string' && payload.detail.trim()) message = payload.detail
  } catch {
    // Keep the safe generic message when the response is not JSON.
  }
  return new ApiError(message, response.status, response.status === 503 ? 'not_ready' : 'server_error')
}

export async function checkHealth({ fetchImpl = fetch, signal } = {}) {
  let response
  try {
    response = await fetchImpl(`${API_BASE_URL}/api/health`, { signal })
  } catch (error) {
    if (error?.name === 'AbortError') throw error
    throw new ApiError('Screening service unavailable. Start the local FastAPI backend and try again.', null, 'unavailable')
  }
  if (!response.ok) throw await responseError(response)
  return response.json()
}

export async function getModelInfo({ fetchImpl = fetch, signal } = {}) {
  const response = await fetchImpl(`${API_BASE_URL}/api/v1/screening/model-info`, { signal })
  if (!response.ok) throw await responseError(response)
  return response.json()
}

export async function analyzeScreening(recordings, { fetchImpl = fetch, signal, timeoutMs = 300_000 } = {}) {
  const required = ['sft', 'pft', 'ctd']
  if (!required.every((task) => recordings?.[task]?.blob instanceof Blob)) {
    throw new ApiError('All three task recordings are required before analysis.', null, 'missing_recording')
  }

  const formData = new FormData()
  for (const task of required) {
    formData.append(`${task}_audio`, recordings[task].blob, `${task}_recording.wav`)
  }

  const controller = new AbortController()
  let timedOut = false
  const timeout = window.setTimeout(() => {
    timedOut = true
    controller.abort()
  }, timeoutMs)
  const abortFromCaller = () => controller.abort()
  signal?.addEventListener('abort', abortFromCaller, { once: true })

  try {
    const response = await fetchImpl(`${API_BASE_URL}/api/v1/screening/analyze`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    })
    if (!response.ok) throw await responseError(response)
    return await response.json()
  } catch (error) {
    if (timedOut) throw new ApiError('The local analysis took too long. You can retry without re-recording.', null, 'timeout')
    if (signal?.aborted) throw new ApiError('The analysis was cancelled.', null, 'cancelled')
    if (error instanceof ApiError) throw error
    throw new ApiError('Could not reach the local screening service. Check that FastAPI is running.', null, 'network_error')
  } finally {
    window.clearTimeout(timeout)
    signal?.removeEventListener('abort', abortFromCaller)
  }
}
