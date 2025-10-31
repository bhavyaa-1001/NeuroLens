import { useRef, useState } from 'react';
import { uploadDatasetZip } from '@/services/nodeApi';
import { useDatasetStore } from '@/state/datasetStore';

export default function UploadArea() {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const { refreshImages } = useDatasetStore();

  const onSelect = async (file?: File | null) => {
    if (!file) return;
    setIsUploading(true);
    try {
      await uploadDatasetZip(file);
      await refreshImages();
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="flex items-center gap-3">
      <button
        className="rounded-md bg-primary-600 hover:bg-primary-500 px-4 py-2 text-sm font-medium"
        onClick={() => inputRef.current?.click()}
        disabled={isUploading}
      >
        {isUploading ? 'Uploading...' : 'Upload .zip'}
      </button>
      <input
        ref={inputRef}
        type="file"
        accept=".zip"
        className="hidden"
        onChange={(e) => onSelect(e.target.files?.[0])}
      />
      <p className="text-xs opacity-70">Accepted: .zip</p>
    </div>
  );
}









