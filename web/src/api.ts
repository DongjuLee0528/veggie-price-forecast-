import { useEffect, useState } from 'react'

const cache = new Map<string, unknown>()

export function useApi<T>(path: string | null) {
  const [attempt, setAttempt] = useState(0)
  const [result, setResult] = useState<{ path: string; error: string } | null>(null)
  const [, refresh] = useState(0)
  const data = path ? cache.get(path) as T | undefined : undefined

  useEffect(() => {
    if (!path) return
    const controller = new AbortController()
    const load = () => fetch(path, { signal: controller.signal, cache: "no-store" }).then(async response => {
      if (!response.ok) {
        const body = await response.json().catch(() => null)
        throw new Error(body?.message ?? '서버에 연결할 수 없습니다.')
      }
      return response.json()
    }).then(value => {
      if (!controller.signal.aborted) {
        setResult(null)
        cache.set(path, value)
        refresh(value => value + 1)
      }
    }).catch((error: Error) => {
      if (!controller.signal.aborted) setResult({ path, error: error.message })
    })
    void load()
    const timer = window.setInterval(() => { if (!document.hidden) void load() }, 300000)
    return () => { window.clearInterval(timer); controller.abort() }
  }, [path, attempt])

  return {
    data,
    error: result?.path === path ? result?.error : undefined,
    retry: () => {
      setResult(null)
      setAttempt(value => value + 1)
    },
  }
}
