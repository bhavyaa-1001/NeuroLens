import { create } from 'zustand';
import type { InferenceResult } from '@/types/results';

type ResultsState = {
  results: InferenceResult[];
  setResults: (r: InferenceResult[]) => void;
  clear: () => void;
};

export const useResultsStore = create<ResultsState>((set) => ({
  results: [],
  setResults: (results) => set({ results }),
  clear: () => set({ results: [] }),
}));


