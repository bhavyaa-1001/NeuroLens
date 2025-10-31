import { create } from 'zustand';
import { listImages } from '@/services/nodeApi';
import type { DatasetImage } from '@/types/dataset';

type DatasetState = {
  images: DatasetImage[];
  loading: boolean;
  refreshImages: () => Promise<void>;
};

export const useDatasetStore = create<DatasetState>((set, get) => ({
  images: [],
  loading: false,
  refreshImages: async () => {
    set({ loading: true });
    try {
      const images = await listImages();
      set({ images });
    } finally {
      set({ loading: false });
    }
  },
}));









