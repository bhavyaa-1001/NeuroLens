import { create } from 'zustand';

type ModelType = 'YOLO' | 'ResNet' | 'Mask R-CNN';
type TaskType = 'Detection' | 'Segmentation' | 'Classification';

type WorkspaceState = {
  modelType: ModelType;
  taskType: TaskType;
  objective: string;
  setModelType: (m: ModelType) => void;
  setTaskType: (t: TaskType) => void;
  setObjective: (o: string) => void;
};

export const useWorkspaceStore = create<WorkspaceState>((set) => ({
  modelType: 'YOLO',
  taskType: 'Detection',
  objective: '',
  setModelType: (modelType) => set({ modelType }),
  setTaskType: (taskType) => set({ taskType }),
  setObjective: (objective) => set({ objective }),
}));









