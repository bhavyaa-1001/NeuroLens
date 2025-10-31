import { useState } from 'react';
import { useWorkspaceStore } from '@/state/workspaceStore';

type Props = { compact?: boolean };

const MODEL_TYPES = ['YOLO', 'ResNet', 'Mask R-CNN'] as const;
const TASK_TYPES = ['Detection', 'Segmentation', 'Classification'] as const;

export default function InputsPanel({ compact = false }: Props) {
  const { modelType, taskType, objective, setModelType, setTaskType, setObjective } = useWorkspaceStore();
  const [localObjective, setLocalObjective] = useState(objective);

  return (
    <div className="glass rounded-xl p-4 space-y-4">
      <h2 className="text-sm font-semibold text-primary-200">Inputs</h2>
      <div className="grid grid-cols-1 gap-3">
        <div>
          <label className="block text-xs mb-1">Model Type</label>
          <select
            value={modelType}
            onChange={(e) => setModelType(e.target.value as any)}
            className="w-full rounded-md bg-slate-900 border border-white/10 px-3 py-2 text-sm"
          >
            {MODEL_TYPES.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-xs mb-1">Task Type</label>
          <select
            value={taskType}
            onChange={(e) => setTaskType(e.target.value as any)}
            className="w-full rounded-md bg-slate-900 border border-white/10 px-3 py-2 text-sm"
          >
            {TASK_TYPES.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-xs mb-1">Project Objective</label>
          <textarea
            value={localObjective}
            onChange={(e) => setLocalObjective(e.target.value)}
            onBlur={() => setObjective(localObjective)}
            className="w-full min-h-[96px] rounded-md bg-slate-900 border border-white/10 px-3 py-2 text-sm"
            placeholder="Describe your objective..."
          />
        </div>
      </div>
    </div>
  );
}



