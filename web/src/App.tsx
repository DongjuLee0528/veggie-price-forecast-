import { useState } from 'react'
import {
  CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { useApi } from './api'
import type { Forecast, ForecastLog, History, Item, Point } from './types'

const money = new Intl.NumberFormat('ko-KR', { maximumFractionDigits: 0 })
const formatPrice = (value: number) => money.format(value)
const percent = (value: number | null) => value === null ? '—' : `${(value * 100).toFixed(2)}%`
const shortDate = (date: string) => date.slice(5).replace('-', '.')
const tabs = ['예보', '가격 기록', '예측 기록'] as const

function LoadState({ error, retry }: { error?: string; retry: () => void }) {
  return error ? (
    <div className="status-card" role="alert">
      <strong>데이터를 불러오지 못했습니다.</strong>
      <p>{error}</p>
      <button className="retry" onClick={retry}>다시 시도</button>
    </div>
  ) : <div className="status-card" role="status">가격 데이터를 불러오는 중입니다.</div>
}

function PriceChart({ rows, forecast = false, comparison = false }: {
  rows: { x: string | number; label?: string; actual?: number | null; estimate?: number | null; previous?: number }[]
  forecast?: boolean
  comparison?: boolean
}) {
  const labels = new Map(rows.map(row => [row.x, row.label ?? String(row.x)]))
  if (rows.length === 0) return <p className="empty">표시할 가격 기록이 없습니다.</p>
  return <div className="chart" aria-label="도매 가격 추이 그래프">
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={rows} margin={{ top: 12, right: 12, bottom: 2, left: 0 }}>
        <CartesianGrid stroke="#dfe9e3" vertical={false} strokeDasharray="3 5" />
        <XAxis dataKey="x" type={forecast ? 'number' : 'category'}
          domain={forecast ? [rows[0].x as number, 10] : undefined}
          ticks={forecast ? [-28, -21, -14, -7, 0, 5, 10] : undefined}
          tickFormatter={value => labels.get(value) ?? String(value)}
          tick={{ fontSize: 11, fill: '#58736a' }} tickLine={false} axisLine={false} minTickGap={18} />
        <YAxis width={65} tickFormatter={formatPrice} tick={{ fontSize: 11, fill: '#58736a' }}
          tickLine={false} axisLine={false} domain={['auto', 'auto']} />
        <Tooltip labelFormatter={value => labels.get(value) ?? String(value)}
          formatter={(value: number, name: string) => [`${formatPrice(value)} 원/kg`, name]}
          contentStyle={{ borderColor: '#dfe9e3', borderRadius: 10 }} />
        {forecast && <ReferenceLine x={0} stroke="#9ab0a7" strokeDasharray="4 4" />}
        <Line name="실제 가격" dataKey="actual" type="linear" stroke="#245a47" strokeWidth={2.5}
          dot={false} isAnimationActive={false} connectNulls />
        {(forecast || comparison) && <Line name="예측 가격" dataKey="estimate" type="linear"
          stroke="#237889" strokeWidth={2.5} strokeDasharray="6 4" isAnimationActive={false}
          dot={forecast ? { r: 3, fill: '#237889' } : false} connectNulls />}
        {!forecast && !comparison && <Line name="전년 동기" dataKey="previous" type="linear"
          stroke="#8a734a" strokeWidth={1.8} strokeDasharray="5 4" dot={false}
          isAnimationActive={false} connectNulls />}
      </LineChart>
    </ResponsiveContainer>
  </div>
}

function ForecastPanel({ forecast }: { forecast: Forecast }) {
  const recent = forecast.recent_actual
  const latest = recent.at(-1)
  const rows = [
    ...recent.map((point, index) => ({ x: index - recent.length + 1, label: shortDate(point.date),
      actual: point.price, estimate: index === recent.length - 1 ? point.price : undefined })),
    ...forecast.short_term.map(point => ({ x: point.horizon, label: `+${point.horizon}거래일`, estimate: point.p50 })),
  ]
  return <>
    <section className="forecast-panel" aria-labelledby="forecast-title">
      <div className="section-heading">
        <div><h2 id="forecast-title">가격 흐름</h2><p>최근 {recent.length}거래일 실측과 단기 예측</p></div>
        <span className="unit">{forecast.unit}</span>
      </div>
      <div className="legend"><span>실제 가격</span><span className="estimate">예측 가격</span></div>
      <PriceChart rows={rows} forecast />
      <p className="chart-note">예측 지점은 품목별 거래일 기준입니다. 점선은 표시된 예측 지점을 연결합니다.</p>
    </section>
    <section className="price-summary" aria-label="예측 가격 요약">
      <div className="price-tile latest">
        <span>마지막 실측가</span><strong>{latest ? formatPrice(latest.price) : '—'}</strong>
        <small>{latest?.date}</small>
      </div>
      {forecast.short_term.map(point => {
        const change = (point.p50 / point.base_price - 1) * 100
        return <div className="price-tile" key={point.horizon}>
          <span>{point.horizon}거래일 후</span><strong>{formatPrice(point.p50)}</strong>
          <small className={change > 0 ? 'up' : change < 0 ? 'down' : ''}>
            {change > 0 ? '+' : ''}{change.toFixed(1)}%<span className="comparison-label"> · 7거래일 평균 대비</span>
          </small>
        </div>
      })}
    </section>
  </>
}

function mergeHistory(actual: Point[], previous: Point[]) {
  const dates = new Map<string, { actual?: number; previous?: number }>()
  for (const point of actual) dates.set(point.date, { actual: point.price })
  for (const point of previous) dates.set(point.date, { ...dates.get(point.date), previous: point.price })
  return [...dates.entries()].sort(([left], [right]) => left.localeCompare(right))
    .map(([date, values]) => ({ x: date, label: shortDate(date), ...values }))
}

function HistoryPanel({ item }: { item: string }) {
  const ranges = [['1m', '1개월'], ['3m', '3개월'], ['1y', '1년'], ['all', '전체']]
  const [range, setRange] = useState('3m')
  const [previous, setPrevious] = useState(true)
  const request = useApi<History>(`/api/history?item=${encodeURIComponent(item)}&range=${range}`)
  return <section className="forecast-panel">
    <div className="section-heading"><h2>가격 기록</h2><span className="unit">원/kg</span></div>
    <div className="filter-row">
      <div className="pills" aria-label="조회 기간">{ranges.map(([value, label]) =>
        <button key={value} aria-pressed={range === value} onClick={() => setRange(value)}>{label}</button>)}</div>
      <label className="checkbox"><input type="checkbox" checked={previous} onChange={event => setPrevious(event.target.checked)} /> 전년 동기</label>
    </div>
    {request.data ? <PriceChart rows={mergeHistory(request.data.actual, previous ? request.data.last_year : [])} />
      : <LoadState error={request.error} retry={request.retry} />}
  </section>
}

function LogPanel({ item }: { item: string }) {
  const [horizon, setHorizon] = useState(10)
  const request = useApi<ForecastLog>(`/api/forecast-log?item=${encodeURIComponent(item)}&h=${horizon}`)
  const rows = request.data?.points.filter(point => point.source === 'backtest' && point.target_date !== null)
    .map(point => ({ x: point.target_date!, label: shortDate(point.target_date!), actual: point.actual, estimate: point.p50 })) ?? []
  return <section className="forecast-panel">
    <div className="section-heading"><h2>예측 기록</h2><span className="unit">원/kg</span></div>
    <div className="filter-row"><div className="pills" aria-label="예측 기간">{[1, 5, 10].map(value =>
      <button key={value} aria-pressed={horizon === value} onClick={() => setHorizon(value)}>{value}거래일 후</button>)}</div></div>
    {request.data ? <>
      <div className="metric-row">
        <div><span>평균 가격 오차</span><strong>{percent(request.data.nmae)}</strong></div>
        <div><span>방향 적중률</span><strong>{percent(request.data.direction_acc)}</strong></div>
      </div>
      {request.data.period_start && <p className="chart-note">사후 평가 {request.data.period_start} ~ {request.data.period_end} · 최근 기록을 그래프에 표시</p>}
      <div className="legend"><span>실제 가격</span><span className="estimate">예측 가격</span></div>
      <PriceChart rows={rows} comparison />
      <p className="chart-note">방향은 예측 기준일의 최근 7거래일 평균 가격과 비교합니다.</p>
    </> : <LoadState error={request.error} retry={request.retry} />}
  </section>
}

export default function App() {
  const [selected, setSelected] = useState('배추')
  const [tab, setTab] = useState<typeof tabs[number]>('예보')
  const items = useApi<Item[]>('/api/items')
  const item = items.data?.find(item => item.name === selected) ?? items.data?.[0]
  const forecast = useApi<Forecast>(item ? `/api/forecast?item=${encodeURIComponent(item.name)}` : null)
  return <>
    <header className="masthead">
      <div className="shell header-inner">
        <div className="brand"><span className="brand-mark" aria-hidden="true"><i /><i /><i /></span>농산물 가격예보</div>
        {forecast.data && <div className="data-date"><span>데이터 기준일</span><time dateTime={forecast.data.base_date}>{forecast.data.base_date}</time></div>}
      </div>
    </header>
    <main className="shell">
      {!items.data ? <LoadState error={items.error} retry={items.retry} /> : <>
        <nav className="item-tabs" aria-label="품목 선택">{items.data.map(option =>
          <button key={option.name} aria-pressed={item?.name === option.name} onClick={() => setSelected(option.name)}>{option.name}</button>)}</nav>
        {item && <>
          <div className="title-row"><h1>{item.name}</h1><span className="item-type">{item.type} · 도매 경락가</span></div>
          <nav className="page-tabs" aria-label="가격예보 메뉴">{tabs.map(name =>
            <button key={name} aria-current={tab === name ? 'page' : undefined} onClick={() => setTab(name)}>{name}</button>)}</nav>
          {tab === '예보' && (forecast.data ? <ForecastPanel forecast={forecast.data} /> : <LoadState error={forecast.error} retry={forecast.retry} />)}
          {tab === '가격 기록' && <HistoryPanel item={item.name} />}
          {tab === '예측 기록' && <LogPanel item={item.name} />}
        </>}
      </>}
    </main>
    <footer className="shell">도매가 기준 예측이며, 소매가는 며칠 늦게 따라 움직일 수 있습니다. 구매 판단 참고용입니다.</footer>
  </>
}
