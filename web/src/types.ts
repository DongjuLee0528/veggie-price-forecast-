export type Item = { name: string; type: string; unit: string }
export type Point = { date: string; price: number }
export type ShortTerm = {
  horizon: number
  date: string | null
  p10: number | null
  p50: number
  p90: number | null
  base_price: number
  model: string
}
export type Forecast = {
  item: string
  type: string
  base_date: string
  unit: string
  data_pending?: boolean
  model_version: string
  signal?: string | null
  recent_actual: Point[]
  short_term: ShortTerm[]
  mid_term: { k: number; period_start: string; period_end: string; p50: number; actual: number | null }[]
}
export type History = { item: string; unit: string; actual: Point[]; last_year: Point[] }
export type LogPoint = {
  target_date: string | null
  made_on: string
  p50: number
  actual: number | null
  source: 'backtest' | 'live'
}
export type ForecastLog = {
  points: LogPoint[]
  nmae: number | null
  direction_acc: number | null
  period_start: string | null
  period_end: string | null
}

export type DirectionResult = {
  origin_price: number | null
  predicted_price: number
  actual_price: number | null
  base_date: string
  target_date: string | null
  predicted_direction: number | null
  observed_direction: number | null
  hit: boolean | null
  retrospective: boolean
}
