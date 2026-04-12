import { useMemo, useState } from 'react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api, type PredictResponse } from './api'
import './style.css'

const FEATURES = ['co', 'humidity', 'light', 'lpg', 'smoke', 'temp'] as const
type Feature = (typeof FEATURES)[number]

type RowPreview = Record<string, unknown>

function formatPct(value: number) {
  return `${value.toFixed(2)}%`
}

function formatNum(value: number, digits = 6) {
  return value.toFixed(digits)
}

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [previewRows, setPreviewRows] = useState<RowPreview[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [feature, setFeature] = useState<Feature>('co')
  const [showPlots, setShowPlots] = useState(true)

  const featureIndex = useMemo(() => FEATURES.indexOf(feature), [feature])

  const originalVsReconstructed = useMemo(() => {
    if (!result) return []
    const maxPoints = Math.min(result.original.length, 300)
    const data = []
    for (let i = 0; i < maxPoints; i += 1) {
      data.push({
        i,
        original: result.original[i][featureIndex],
        reconstructed: result.reconstructed[i][featureIndex],
      })
    }
    return data
  }, [result, featureIndex])

  const errorSeries = useMemo(() => {
    if (!result) return []
    const maxPoints = Math.min(result.sample_error.length, 600)
    const data = []
    for (let i = 0; i < maxPoints; i += 1) {
      data.push({ i, error: result.sample_error[i] })
    }
    return data
  }, [result])

  async function handleUpload() {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const form = new FormData()
      form.append('file', file)

      const response = await api.post<PredictResponse>('/predict?max_rows=5000', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })

      setResult(response.data)
    } catch (e: any) {
      const msg =
        e?.response?.data?.detail ||
        e?.message ||
        'Failed to call backend. Is FastAPI running on http://localhost:8000?'
      setError(String(msg))
    } finally {
      setLoading(false)
    }
  }

  async function buildPreview(selected: File) {
    // Simple preview: read first ~200KB and parse as text if possible (CSV).
    // For XLSX, we skip preview parsing (backend will handle).
    const name = selected.name.toLowerCase()
    if (!name.endsWith('.csv')) {
      setPreviewRows([])
      return
    }

    const chunk = selected.slice(0, 200_000)
    const text = await chunk.text()
    const lines = text.split(/\r?\n/).filter(Boolean).slice(0, 12)
    if (lines.length < 2) {
      setPreviewRows([])
      return
    }

    const headers = lines[0].split(',').map((h) => h.trim())
    const rows = lines.slice(1).map((line) => line.split(','))
    const previews = rows.map((cols) => {
      const row: Record<string, unknown> = {}
      headers.forEach((h, idx) => {
        row[h] = cols[idx]
      })
      return row
    })
    setPreviewRows(previews)
  }

  return (
    <div className="page">
      <header className="header">
        <div>
          <h1 className="title">IoT Autoencoder</h1>
          <p className="subtitle">
            Upload CSV/XLSX to reconstruct sensor data and detect anomalies.
          </p>
        </div>
      </header>

      <section className="card">
        <h2 className="sectionTitle">Upload</h2>
        <div className="row">
          <input
            type="file"
            accept=".csv,.xlsx,.xls"
            onChange={(e) => {
              const f = e.target.files?.[0] ?? null
              setFile(f)
              setResult(null)
              setError(null)
              if (f) void buildPreview(f)
            }}
          />
          <button className="button" onClick={() => void handleUpload()} disabled={!file || loading}>
            {loading ? 'Processing…' : 'Run Prediction'}
          </button>
        </div>
        <p className="hint">
          Required columns: <code>{FEATURES.join(', ')}</code> (optional <code>ts</code> will be ignored).
        </p>

        {error && <div className="errorBox">{error}</div>}
      </section>

      {previewRows.length > 0 && (
        <section className="card">
          <h2 className="sectionTitle">Data Preview (CSV only)</h2>
          <div className="tableWrap">
            <table className="table">
              <thead>
                <tr>
                  {Object.keys(previewRows[0]).map((k) => (
                    <th key={k}>{k}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {previewRows.map((r, idx) => (
                  <tr key={idx}>
                    {Object.values(r).map((v, j) => (
                      <td key={j}>{String(v ?? '')}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {result && (
        <>
          <section className="card">
            <h2 className="sectionTitle">Summary</h2>
            <div className="metrics">
              <div className="metric">
                <div className="metricLabel">MSE</div>
                <div className="metricValue">{formatNum(result.mse)}</div>
              </div>
              <div className="metric">
                <div className="metricLabel">Reconstruction Accuracy</div>
                <div className="metricValue">{formatPct(result.accuracy * 100)}</div>
              </div>
              <div className="metric">
                <div className="metricLabel">Anomaly %</div>
                <div className="metricValue">{formatPct(result.anomaly_percent)}</div>
              </div>
            </div>

            <div className="summaryGrid">
              <div className="summaryItem">
                <span className="summaryKey">MAE</span>
                <span className="summaryVal">{formatNum(result.mae)}</span>
              </div>
              <div className="summaryItem">
                <span className="summaryKey">Threshold</span>
                <span className="summaryVal">{formatNum(result.threshold, 6)}</span>
              </div>
              <div className="summaryItem">
                <span className="summaryKey">Threshold Accuracy</span>
                <span className="summaryVal">{formatPct(result.threshold_accuracy)}</span>
              </div>
            </div>
          </section>

          <section className="card">
            <div className="row spaceBetween">
              <h2 className="sectionTitle">Visualizations</h2>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={showPlots}
                  onChange={(e) => setShowPlots(e.target.checked)}
                />
                Show plots
              </label>
            </div>

            {showPlots && (
              <>
                <div className="row">
                  <label className="selectLabel">
                    Feature
                    <select value={feature} onChange={(e) => setFeature(e.target.value as Feature)}>
                      {FEATURES.map((f) => (
                        <option key={f} value={f}>
                          {f}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>

                <div className="chartGrid">
                  <div className="chartCard">
                    <div className="chartTitle">Original vs Reconstructed (scaled)</div>
                    <div className="chart">
                      <ResponsiveContainer width="100%" height={260}>
                        <LineChart data={originalVsReconstructed}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="i" />
                          <YAxis />
                          <Tooltip />
                          <Line type="monotone" dataKey="original" stroke="#2563eb" dot={false} />
                          <Line
                            type="monotone"
                            dataKey="reconstructed"
                            stroke="#ef4444"
                            dot={false}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  <div className="chartCard">
                    <div className="chartTitle">Reconstruction Error per sample</div>
                    <div className="chart">
                      <ResponsiveContainer width="100%" height={260}>
                        <LineChart data={errorSeries}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="i" />
                          <YAxis />
                          <Tooltip />
                          <Line type="monotone" dataKey="error" stroke="#ef4444" dot={false} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>
              </>
            )}
          </section>

          <section className="card">
            <h2 className="sectionTitle">Anomalies</h2>
            {result.anomalies.length === 0 ? (
              <p className="hint">No anomalies detected for the selected threshold.</p>
            ) : (
              <div className="tableWrap">
                <table className="table">
                  <thead>
                    <tr>
                      <th>Index</th>
                      <th>Error</th>
                      {FEATURES.map((f) => (
                        <th key={f}>{f}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.anomalies.slice(0, 200).map((a) => (
                      <tr key={a.index}>
                        <td>{a.index}</td>
                        <td>{formatNum(a.error, 6)}</td>
                        {FEATURES.map((f) => (
                          <td key={f}>{formatNum(a.row[f], 6)}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {result.anomalies.length > 200 && (
              <p className="hint">Showing first 200 anomalies.</p>
            )}
          </section>
        </>
      )}
    </div>
  )
}

