import { useState } from 'react';
import { runXGBoost } from '@/services/fastApi';
import { useResultsStore } from '@/state/resultsStore';
import { useWorkspaceStore } from '@/state/workspaceStore';

export default function RunModel() {
  const [loading, setLoading] = useState(false);
  const { setResults } = useResultsStore();
  const { modelType, taskType, objective } = useWorkspaceStore();

  const handleRun = async () => {
    setLoading(true);
    try {
      const resp = await runXGBoost({ modelType, taskType, objective });
      setResults(resp.results || []);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-primary-200">Run Model</h2>
        <button
          className="rounded-md bg-accent-600 hover:bg-accent-500 px-4 py-2 text-sm"
          onClick={handleRun}
          disabled={loading}
        >
          {loading ? 'Running…' : 'Run XGBoost Model'}
        </button>
      </div>
      <p className="text-xs mt-2 opacity-70">Sends request to FastAPI `/run-xgboost` and displays results.</p>
    </div>
  );
}


