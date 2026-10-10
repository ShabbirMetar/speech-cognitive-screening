import '@testing-library/jest-dom/vitest'

import { afterEach, vi } from 'vitest'
import { cleanup } from '@testing-library/react'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

if (!URL.createObjectURL) URL.createObjectURL = vi.fn(() => 'blob:recording')
if (!URL.revokeObjectURL) URL.revokeObjectURL = vi.fn()
