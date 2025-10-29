import { fastApi } from '@/services/http';
import type { InferenceResult } from '@/types/results';

export async function runXGBoost(payload: { modelType: string; taskType: string; objective: string; }): Promise<{ results: InferenceResult[] }>
{
  const { data } = await fastApi.post('/run-xgboost', payload);
  return data;
}


