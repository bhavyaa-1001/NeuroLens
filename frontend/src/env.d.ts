/// <reference types="vite/client" />

declare const __APP_VERSION__: string;
declare const __NODE_API_URL__: string;
declare const __FASTAPI_URL__: string;

interface ImportMetaEnv {
  readonly VITE_NODE_API_URL?: string;
  readonly VITE_FASTAPI_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}


