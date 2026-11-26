import { useState } from 'react'
import {
  CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { useApi } from './api'
import type { DirectionResult, Forecast, ForecastLog, History, Item, Point } from './types'

const money = new Intl.NumberFormat('ko-KR', { maximumFractionDigits: 0 })
const formatPrice = (value: number) => money.format(value)
const percent = (value: number | null) => value === null ? '—' : `${(value * 100).toFixed(2)}%`
const shortDate = (date: string) => date.slice(5).replace('-', '.')
const tabs = ['예보', '가격 기록', '예측 기록'] as const
const directionSignals: Record<string, { label: string; trend: string }> = {
  NEXT_DAY_UP: { label: '↑ 상승 예상', trend: 'up' },
  NEXT_DAY_DOWN: { label: '↓ 하락 예상', trend: 'down' },
  NEXT_DAY_FLAT: { label: '— 보합 예상', trend: '' },
  NEXT_DAY_PENDING: { label: '— 방향 판단 보류', trend: '' },
}

function radishScore(probability?: number | null) {
  if (probability == null) return ''
  return probability >= 0.65 || probability <= 0.35 ? ' · 로지스틱 점수 높음' : ' · 로지스틱'
}

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
          dot={comparison && rows.length < 3} isAnimationActive={false} connectNulls />
        {(forecast || comparison) && <Line name="예측 가격" dataKey="estimate" type="linear"
          stroke="#237889" strokeWidth={2.5} strokeDasharray="6 4" isAnimationActive={false}
          dot={forecast || (comparison && rows.length < 3) ? { r: 3, fill: '#237889' } : false} connectNulls />}
        {!forecast && !comparison && <Line name="전년 동기" dataKey="previous" type="linear"
          stroke="#8a734a" strokeWidth={1.8} strokeDasharray="5 4" dot={false}
          isAnimationActive={false} connectNulls />}
      </LineChart>
    </ResponsiveContainer>
  </div>
}

export function DirectionSummary({ forecast }: { forecast: Forecast }) {
  const reference = forecast.model_version.startsWith("api-provisional-v2-") && forecast.short_term.length
    ? { date: forecast.base_date, price: forecast.short_term[0].base_price }
    : forecast.recent_actual.find(point => point.date === forecast.base_date && point.price > 0)
  return <section className="forecast-panel direction-summary" aria-labelledby="direction-title">
    <div className="section-heading">
      <div>
        <h2 id="direction-title">오를까, 내릴까?</h2>
        <p>{reference ? '예측 기준일 관측가 대비' : '예측 기준일 관측가가 없어 방향을 표시할 수 없습니다.'}</p>
      </div>
      {reference && <div className="direction-reference">
        <span>{shortDate(reference.date)} 관측가</span>
        <strong>{formatPrice(reference.price)} <small>{forecast.unit}</small></strong>
        {forecast.model_version.startsWith('api-provisional-') && <span>잠정 정산가격</span>}
      </div>}
    </div>
    <div className="price-summary">
      {forecast.short_term.map(point => {
        const signal = reference && point.horizon === 1 && forecast.signal
          ? directionSignals[forecast.signal]
          : undefined
        const change = !signal && reference && point.model !== 'last_value'
          ? (point.p50 / reference.price - 1) * 100
          : null
        const trend = change === null || change === 0 ? '' : change > 0 ? 'up' : 'down'
        const direction = change === null ? '— 방향 판단 보류'
          : trend === 'up' ? '↑ 상승 예상' : trend === 'down' ? '↓ 하락 예상' : '— 보합 예상'
        const magnitude = change === null || change === 0 ? null
          : Math.abs(change) < 0.05 ? '0.1% 미만' : `${change > 0 ? '+' : '−'}${Math.abs(change).toFixed(1)}%`
        return <div className="price-tile" key={point.horizon}>
          <span>{point.horizon}거래일 후</span>
          <strong className={signal?.trend ?? trend}>{signal?.label ?? direction}</strong>
          <small>예측가 {formatPrice(point.p50)} {forecast.unit}{magnitude && ` · ${magnitude}`}{signal && (radishScore(forecast.radish_probability_up) || ' · 별도 방향 모델')}</small>
        </div>
      })}
    </div>
  </section>
}

function PeriodPrices({ item, period }: { item: string; period: string }) {
  const request = useApi<History>(`/api/history?item=${encodeURIComponent(item)}&range=1y`)
  if (!request.data) return <LoadState error={request.error} retry={request.retry} />
  const groups = new Map<string, { sum: number; count: number; end: string }>()
  for (const point of request.data.actual) {
    const start = new Date(point.date + 'T00:00:00Z')
    if (period === '월별') start.setUTCDate(1)
    else start.setUTCDate(start.getUTCDate() - (start.getUTCDay() + 6) % 7)
    const key = start.toISOString().slice(0, 10)
    const end = new Date(start)
    if (period === '월별') end.setUTCMonth(end.getUTCMonth() + 1, 0)
    else end.setUTCDate(end.getUTCDate() + 6)
    const group = groups.get(key) ?? { sum: 0, count: 0, end: end.toISOString().slice(0, 10) }
    group.sum += point.price
    group.count++
    groups.set(key, group)
  }
  const today = new Intl.DateTimeFormat('sv-SE', { timeZone: 'Asia/Seoul' }).format(new Date())
  return <PriceChart rows={[...groups].slice(-12).map(([date, group]) => ({
    x: date, label: `${shortDate(date)}~${shortDate(group.end)}${group.end >= today ? ' (진행 중)' : ''}`,
    actual: group.sum / group.count,
  }))} />
}

function ForecastPanel({ forecast }: { forecast: Forecast }) {
  const [period, setPeriod] = useState("일별")
  const recent = forecast.recent_actual
  const rows = [
    ...recent.map((point, index) => ({ x: index - recent.length + 1, label: shortDate(point.date),
      actual: point.price, estimate: index === recent.length - 1 ? point.price : undefined })),
    ...forecast.short_term.map(point => ({ x: point.horizon, label: `+${point.horizon}거래일`, estimate: point.p50 })),
  ]
  return <>
    {forecast.data_pending && <section className="forecast-panel" role="status">
      <h2>자료 확인 중</h2>
      <p>최신 예측에 필요한 자료를 확인하고 있습니다. 마지막 정상 예측을 표시합니다.</p>
      <p>이 품목의 예측 기준일: {forecast.base_date}</p>
    </section>}
    <DirectionSummary forecast={forecast} />
    <section className="forecast-panel" aria-labelledby="forecast-title">
      <div className="section-heading">
        <div><h2 id="forecast-title">가격 흐름</h2><p>{period === '일별' ? `최근 ${recent.length}거래일 실측과 단기 예측` : '거래일별 관측 가격의 평균'}</p></div>
        <span className="unit">{forecast.unit}</span>
      </div>
      <div className="pills" aria-label="가격 표시 단위">{['일별', '주별', '월별'].map(unit =>
        <button key={unit} aria-pressed={period === unit} onClick={() => setPeriod(unit)}>{unit}</button>)}</div>
      {period === '일별' ? <><div className="legend"><span>실제 가격</span><span className="estimate">예측 가격</span></div>
      <PriceChart rows={rows} forecast />
      <p className="chart-note">예측 지점은 품목별 거래일 기준입니다. 점선은 표시된 예측 지점을 연결합니다.</p></> : <PeriodPrices item={forecast.item} period={period} />}
    </section>
    {forecast.mid_term.length > 0 && <section className="forecast-panel" aria-labelledby="mid-forecast-title">
      <div className="section-heading"><h2 id="mid-forecast-title">중기 예측</h2><span className="unit">{forecast.unit}</span></div>
      <div className="legend"><span>실제 평균가</span><span className="estimate">예측 평균가</span></div>
      <PriceChart rows={forecast.mid_term.map(point => ({
        x: point.period_start, label: `${shortDate(point.period_start)}~${shortDate(point.period_end)}`,
        actual: point.actual, estimate: point.p50,
      }))} comparison />
      <div className="price-summary">
        {forecast.mid_term.map(point => <div className="price-tile" key={point.k}>
          <span>{shortDate(point.period_start)}~{shortDate(point.period_end)}</span>
          <strong>{formatPrice(point.p50)}</strong>
          <small>{point.actual === null ? '실측 대기'
            : `실측 ${formatPrice(point.actual)}원/kg · 오차 ${percent(Math.abs(point.p50 / point.actual - 1))}`}</small>
        </div>)}
      </div>
    </section>}
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
    <div className="section-heading"><div><h2>가격 기록</h2>{request.data?.actual.at(-1) && <p>최신 가격 {request.data.actual.at(-1)?.date}</p>}</div><span className="unit">원/kg</span></div>
    <div className="filter-row">
      <div className="pills" aria-label="조회 기간">{ranges.map(([value, label]) =>
        <button key={value} aria-pressed={range === value} onClick={() => setRange(value)}>{label}</button>)}</div>
      <label className="checkbox"><input type="checkbox" checked={previous} onChange={event => setPrevious(event.target.checked)} /> 전년 동기</label>
    </div>
    {request.data ? <PriceChart rows={mergeHistory(request.data.actual, previous ? request.data.last_year : [])} />
      : <LoadState error={request.error} retry={request.retry} />}
  </section>
}

function DirectionResults({ item, horizon }: { item: string; horizon: number }) {
  const request = useApi<DirectionResult[]>(`/api/direction-results?item=${encodeURIComponent(item)}&h=${horizon}`)
  if (!request.data) return <LoadState error={request.error} retry={request.retry} />
  const direction = (value: number | null) => value === null ? '—' : value > 0 ? '상승' : value < 0 ? '하락' : '보합'
  const scored = request.data.filter(row => row.hit !== null)
  const matched = request.data.filter(row => row.actual_price !== null && !row.retrospective)
  const rows = matched.slice().reverse().map(row => ({
    x: row.target_date!, label: `${shortDate(row.target_date!)} (기준 ${shortDate(row.base_date)} · +${horizon}거래일)`,
    actual: row.actual_price, estimate: row.predicted_price,
  }))
  return <section className="live-direction-results" aria-label="최근 방향 결과">
    <h3>최근 방향 결과 · 잠정</h3>
    <p className="chart-note">기준일 관측가와 이후 {horizon}번째 관측 거래일 가격을 비교합니다.</p>
    {request.data.length === 0 ? <p className="empty">저장된 예보가 없습니다.</p> :
      <table className="direction-results-table">
        <thead><tr><th scope="col">기준일</th><th scope="col">실측일</th><th scope="col">예측</th><th scope="col">실측</th><th scope="col">결과</th></tr></thead>
        <tbody>{request.data.map(row => <tr key={row.base_date}>
          <td>{shortDate(row.base_date)}{row.origin_price !== null && ` · ${formatPrice(row.origin_price)}원/kg`}</td>
          <td>{row.target_date ? shortDate(row.target_date) : '—'}</td>
          <td>{direction(row.predicted_direction)}{radishScore(row.radish_probability_up)} · {formatPrice(row.predicted_price)}원/kg</td>
          <td>{row.actual_price === null ? '—' : `${direction(row.observed_direction)} · ${formatPrice(row.actual_price)}원/kg`}</td>
          <td>{row.retrospective ? '재구성' : row.hit === null ? row.predicted_direction === null ? '판단 보류' : '정산 대기'
            : row.hit ? '적중' : '빗나감'}</td>
        </tr>)}</tbody>
      </table>}
    <p>최근 사전 예측 방향 적중률 {percent(scored.length ? scored.filter(row => row.hit).length / scored.length : null)} ({scored.length}건)</p>
    <p>사전 예측 평균 가격 오차 {percent(matched.length ? matched.reduce((sum, row) =>
      sum + Math.abs(row.predicted_price / row.actual_price! - 1), 0) / matched.length : null)} ({matched.length}건)</p>
    <h3>운영 예측과 실제 가격</h3>
    <PriceChart rows={rows} comparison />
  </section>
}

function LogPanel({ item }: { item: string }) {
  const [horizon, setHorizon] = useState(10)
  const request = useApi<ForecastLog>(`/api/forecast-log?item=${encodeURIComponent(item)}&h=${horizon}&backtestOnly=true`)
  const rows = request.data?.points.filter(point => point.source === 'backtest' && point.target_date !== null)
    .map(point => ({ x: point.target_date!, label: shortDate(point.target_date!), actual: point.actual, estimate: point.p50 })) ?? []
  return <section className="forecast-panel">
    <div className="section-heading"><h2>예측 기록</h2><span className="unit">원/kg</span></div>
    <div className="filter-row"><div className="pills" aria-label="예측 기간">{[1, 5, 10].map(value =>
      <button key={value} aria-pressed={horizon === value} onClick={() => setHorizon(value)}>{value}거래일 후</button>)}</div></div>
    <DirectionResults item={item} horizon={horizon} />
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
        {forecast.data && <div className="data-date"><span>예측 기준일</span><time dateTime={forecast.data.base_date}>{forecast.data.base_date}</time></div>}
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
