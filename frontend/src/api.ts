import axios from 'axios'

export const api = axios.create({
  baseURL: 'http://localhost:8000',
})

export type PredictResponse = {
  mse: number
  mae: number
  accuracy: number
  threshold: number
  threshold_accuracy: number
  anomaly_percent: number
  anomalies: Array<{
    index: number
    error: number
    label: 'Anomaly'
    row: Record<string, number>
  }>
  sample_error: number[]
  original: number[][]
  reconstructed: number[][]
}

