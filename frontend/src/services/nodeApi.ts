import { nodeApi } from '@/services/http';
import type { DatasetImage } from '@/types/dataset';

export async function uploadDatasetZip(file: File): Promise<{ ok: boolean }>
{
  const form = new FormData();
  form.append('dataset', file);
  const { data } = await nodeApi.post('/api/datasets/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
}

export async function listImages(): Promise<DatasetImage[]>
{
  const { data } = await nodeApi.get('/api/images');
  return data?.images ?? [];
}


